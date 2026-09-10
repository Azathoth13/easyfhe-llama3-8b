from __future__ import annotations

"""Weight data and ConstantBundle ownership for diagonal linear maps."""

import os
import time

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import synchronize_device as sync_device

from .config import LinearChunk, LinearOperatorConfig, LinearPlan
from .structured_weight import StructuredLinearWeight


def _cpu_packer(
    weight: np.ndarray,
    *,
    plan: LinearPlan,
    chunk: LinearChunk,
    slots: int,
    dtype: np.dtype,
    crypto_context,
):
    feature_at_slot = np.arange(int(slots), dtype=np.int64) // int(
        plan.token_lanes
    )
    values = np.empty(
        (int(chunk.giant_count) * int(plan.baby_steps), int(slots)),
        dtype=dtype,
    )
    row = 0
    for giant_relative in range(int(chunk.giant_count)):
        giant = int(chunk.giant_base) + giant_relative
        output_feature = (
            feature_at_slot - giant * int(plan.baby_steps)
        ) % int(plan.dimension)
        for baby in range(int(plan.baby_steps)):
            input_feature = (feature_at_slot + baby) % int(plan.dimension)
            values[row] = weight[output_feature, input_feature]
            row += 1
    tensor = crypto_context.tensor.from_numpy(values)
    if hasattr(tensor, "to"):
        tensor = tensor.to(crypto_context.device)
    return tensor


def _chunk_packer(
    weight: np.ndarray | StructuredLinearWeight,
    *,
    plan: LinearPlan,
    chunk: LinearChunk,
    dtype: np.dtype,
    crypto_context,
    use_gpu_packer: bool,
    structured_device_sources: tuple[object, ...] = (),
    structured_device_metadata: dict[str, object] | None = None,
):
    slots = int(crypto_context.max_slots)

    def pack(raw_weight, requested_slots, runtime_context):
        del runtime_context
        if int(requested_slots) != slots:
            raise ValueError(
                f"linear weight expects slots={slots}, got {requested_slots}."
            )
        if use_gpu_packer:
            if isinstance(weight, StructuredLinearWeight):
                from llama3fhe.operators.linear.gpu_packer import (
                    pack_structured_token_lane_bsgs_weight_chunk_triton_gpu,
                )

                if not structured_device_sources or structured_device_metadata is None:
                    raise RuntimeError(
                        "structured GPU weight was not prepared before packing."
                    )
                return pack_structured_token_lane_bsgs_weight_chunk_triton_gpu(
                    structured_device_sources,
                    structured_device_metadata,
                    slots=slots,
                    dimension=int(plan.dimension),
                    token_lanes=int(plan.token_lanes),
                    baby_steps=int(plan.baby_steps),
                    giant_base=int(chunk.giant_base),
                    giant_count=int(chunk.giant_count),
                    device=str(crypto_context.device),
                )

            from llama3fhe.operators.linear.gpu_packer import (
                pack_token_lane_bsgs_weight_chunk_triton_gpu,
            )

            return pack_token_lane_bsgs_weight_chunk_triton_gpu(
                raw_weight,
                slots=slots,
                dimension=int(plan.dimension),
                token_lanes=int(plan.token_lanes),
                baby_steps=int(plan.baby_steps),
                giant_base=int(chunk.giant_base),
                giant_count=int(chunk.giant_count),
                device=str(crypto_context.device),
            )
        physical_weight = (
            weight.materialize(dtype=dtype)
            if isinstance(weight, StructuredLinearWeight)
            else weight
        )
        return _cpu_packer(
            physical_weight,
            plan=plan,
            chunk=chunk,
            slots=slots,
            dtype=dtype,
            crypto_context=crypto_context,
        )

    return pack


class LinearWeight:
    """One square physical weight and the constants derived from it.

    This object deliberately has no BSGS/operator configuration.  The caller
    supplies a :class:`LinearPlan` and :class:`LinearOperatorConfig` whenever a
    plaintext chunk is requested.  Persistent ConstantBundles are keyed by
    that physical plan, so one weight can safely be reused by different
    operator schedules.
    """

    def __init__(
        self,
        weight: np.ndarray | StructuredLinearWeight,
        *,
        dtype: np.dtype,
        crypto_context,
    ) -> None:
        self.dtype = np.dtype(dtype)
        self.crypto_context = crypto_context
        self._structured_weight = (
            weight if isinstance(weight, StructuredLinearWeight) else None
        )
        if self._structured_weight is None:
            weight = np.asarray(weight)
            if weight.ndim != 2 or int(weight.shape[0]) != int(weight.shape[1]):
                raise ValueError(
                    "linear weight must be one square physical page, got "
                    f"{weight.shape}."
                )
            self._weight = np.ascontiguousarray(weight, dtype=self.dtype)
            self._dimension = int(self._weight.shape[0])
        else:
            if (
                self._structured_weight.is_complex
                and not np.issubdtype(self.dtype, np.complexfloating)
            ):
                raise TypeError(
                    "structured weight with an imaginary component requires "
                    f"complex output dtype, got {self.dtype}."
                )
            self._weight = None
            self._dimension = int(self._structured_weight.dimension)
        self._host_source = None
        self._device_source = None
        self._host_structured_sources: tuple[object, ...] = ()
        self._device_structured_sources: tuple[object, ...] = ()
        self._device_structured_metadata: dict[str, object] = {}
        self._fallback_physical_weight: np.ndarray | None = None
        self._bundles: dict[tuple[object, ...], object] = {}
        self._transient_cache_totals: dict[str, int] = {}
        self._closed = False
        self.setup_seconds = 0.0

    @property
    def dimension(self) -> int:
        return int(self._dimension)

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def prepare(self, operator_config: LinearOperatorConfig) -> float:
        """Materialize the configured raw source, returning host wall time."""

        if self.closed:
            raise RuntimeError("linear weight is closed.")
        start = time.perf_counter()
        if self._structured_weight is None:
            if self._host_source is None:
                self._host_source = self.crypto_context.tensor.from_numpy(self._weight)
            if str(operator_config.raw_weight_policy) == "device":
                if self._device_source is None:
                    self._device_source = (
                        self._host_source.to(self.crypto_context.device)
                        if hasattr(self._host_source, "to")
                        else self._host_source
                    )
        else:
            if not self._host_structured_sources:
                self._host_structured_sources = tuple(
                    self.crypto_context.tensor.from_numpy(source)
                    for source in self._structured_weight.sources
                )
                self._host_source = self._host_structured_sources[0]
            if str(operator_config.raw_weight_policy) == "device":
                if not self._device_structured_sources:
                    self._device_structured_sources = tuple(
                        source.to(self.crypto_context.device)
                        if hasattr(source, "to")
                        else source
                        for source in self._host_structured_sources
                    )
                    self._device_source = self._device_structured_sources[0]
                if not self._device_structured_metadata:
                    for name in (
                        "real_source",
                        "real_row",
                        "real_scale",
                        "imag_source",
                        "imag_row",
                        "imag_scale",
                        "column",
                        "column_scale",
                    ):
                        host = self.crypto_context.tensor.from_numpy(
                            getattr(self._structured_weight, name)
                        )
                        self._device_structured_metadata[name] = (
                            host.to(self.crypto_context.device)
                            if hasattr(host, "to")
                            else host
                        )
        sync_device(self.crypto_context.device)
        elapsed = float(time.perf_counter() - start)
        self.setup_seconds += elapsed
        return elapsed

    def _source(self, operator_config: LinearOperatorConfig):
        if str(operator_config.raw_weight_policy) == "device":
            if self._device_source is None:
                self.prepare(operator_config)
            return self._device_source
        if self._host_source is None:
            self.prepare(operator_config)
        return self._host_source

    @staticmethod
    def _bundle_key(
        plan: LinearPlan,
        chunk: LinearChunk,
        operator_config: LinearOperatorConfig,
    ) -> tuple[object, ...]:
        return (
            *plan.cache_key,
            int(chunk.index),
            str(operator_config.constant_cache_mode),
            str(operator_config.raw_weight_policy),
        )

    @staticmethod
    def _bundle_kwargs(
        vector: object,
        chunk: LinearChunk,
        operator_config: LinearOperatorConfig,
    ) -> dict[str, object]:
        return {
            "vectors": {chunk.name: vector},
            "cache_mode": str(operator_config.constant_cache_mode),
        }

    def _new_bundle(
        self,
        plan: LinearPlan,
        chunk: LinearChunk,
        operator_config: LinearOperatorConfig,
    ):
        source = self._source(operator_config)
        use_gpu_packer = bool(
            str(operator_config.raw_weight_policy) == "device"
            and str(self.crypto_context.device).startswith("cuda")
            and os.environ.get("EASYFHE_TOKEN_LANE_GPU_PACK", "1") != "0"
        )
        packing_weight = (
            self._structured_weight
            if self._structured_weight is not None
            else self._weight
        )
        if isinstance(packing_weight, StructuredLinearWeight) and not use_gpu_packer:
            if self._fallback_physical_weight is None:
                self._fallback_physical_weight = packing_weight.materialize(
                    dtype=self.dtype
                )
            packing_weight = self._fallback_physical_weight
        packer = _chunk_packer(
            packing_weight,
            plan=plan,
            chunk=chunk,
            dtype=self.dtype,
            crypto_context=self.crypto_context,
            use_gpu_packer=use_gpu_packer,
            structured_device_sources=self._device_structured_sources,
            structured_device_metadata=(
                self._device_structured_metadata
                if self._structured_weight is not None
                else None
            ),
        )
        if hasattr(self.crypto_context.fhe, "UnpackedRaw"):
            vector = self.crypto_context.fhe.UnpackedRaw(source, packer)
        else:
            packed = packer(
                source,
                int(self.crypto_context.max_slots),
                self.crypto_context.context,
            )
            vector = self.crypto_context.fhe.PackedRaw(packed)
        return self.crypto_context.fhe.ConstantBundle(
            **self._bundle_kwargs(vector, chunk, operator_config)
        )

    def plaintext(
        self,
        plan: LinearPlan,
        chunk: LinearChunk,
        *,
        level: int,
        operator_config: LinearOperatorConfig,
    ):
        if self.closed:
            raise RuntimeError("linear weight is closed.")
        if int(plan.dimension) != self.dimension:
            raise ValueError(
                f"linear plan dimension={plan.dimension} does not match "
                f"weight dimension={self.dimension}."
            )
        key = self._bundle_key(plan, chunk, operator_config)
        bundle = self._bundles.get(key)
        if bundle is None:
            bundle = self._new_bundle(plan, chunk, operator_config)
            if str(operator_config.constant_cache_mode) != "none":
                self._bundles[key] = bundle
        cur_limbs = int(self.crypto_context.context.L) - int(level)
        plaintext = bundle.plaintext(
            chunk.name,
            state=fhe.CipherState(
                cur_limbs=cur_limbs,
                scale_degree=1,
                scaling_factor=self.crypto_context.context.scale_at(
                    cur_limbs
                ),
            ),
            slots=int(self.crypto_context.max_slots),
            context=self.crypto_context.context,
            is_ext=(str(operator_config.hoist_strategy).lower() != "normal"),
            cache=True,
        )
        if str(operator_config.constant_cache_mode) == "none":
            for name, value in bundle.cache_info().items():
                if isinstance(value, (int, np.integer)) and not isinstance(
                    value, bool
                ):
                    self._transient_cache_totals[name] = int(
                        self._transient_cache_totals.get(name, 0)
                    ) + int(value)
        return plaintext

    def cache_info(self) -> dict[str, object]:
        if self.closed:
            raise RuntimeError("linear weight is closed.")
        infos = [bundle.cache_info() for bundle in self._bundles.values()]
        totals: dict[str, object] = {"bundle_count": len(infos)}
        numeric_keys = {
            name
            for info in infos
            for name, value in info.items()
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        }
        numeric_keys.update(self._transient_cache_totals)
        for name in numeric_keys:
            totals[name] = int(self._transient_cache_totals.get(name, 0)) + sum(
                int(info.get(name, 0)) for info in infos
            )
        for name in (
            "middle_entries",
            "plain_entries",
            "middle_bytes",
            "plain_bytes",
            "total_bytes",
            "middle_hits",
            "middle_misses",
            "plain_hits",
            "plain_misses",
        ):
            totals.setdefault(name, 0)
        return totals

    def clear_cache(self) -> None:
        for bundle in self._bundles.values():
            bundle.clear_cache()
        self._bundles.clear()

    def close(self) -> None:
        self.clear_cache()
        self._transient_cache_totals.clear()
        released_ids: set[int] = set()
        for value in (
            *self._device_structured_metadata.values(),
            *self._device_structured_sources,
            *self._host_structured_sources,
        ):
            if value is not None and id(value) not in released_ids:
                release_if_supported(value)
                released_ids.add(id(value))
        if self._device_source is not None and id(self._device_source) not in released_ids:
            release_if_supported(self._device_source)
            released_ids.add(id(self._device_source))
        if self._host_source is not None and id(self._host_source) not in released_ids:
            release_if_supported(self._host_source)
        self._device_source = None
        self._host_source = None
        self._device_structured_metadata.clear()
        self._device_structured_sources = ()
        self._host_structured_sources = ()
        self._weight = None
        self._structured_weight = None
        self._fallback_physical_weight = None
        self._closed = True

    def __enter__(self) -> "LinearWeight":
        if self.closed:
            raise RuntimeError("cannot enter a closed linear weight.")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()
