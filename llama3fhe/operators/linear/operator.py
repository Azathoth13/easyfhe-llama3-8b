from __future__ import annotations

"""Operator-owned BSGS execution for prepared diagonal linear weights."""

import time
from dataclasses import dataclass

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import synchronize_device as sync_device

from .config import LinearOperatorConfig, LinearPlan, linear_encode_reuse_enabled
from .structured_weight import StructuredLinearWeight
from .weight import LinearWeight

_EMPTY_PROFILE = {
    "weight_prepare_cpu": 0.0,
    "weight_upload": 0.0,
    "baby_precompute": 0.0,
    "baby_precompute_enqueue": 0.0,
    "baby_cache_hits": 0.0,
    "baby_cache_misses": 0.0,
    "pack_encode": 0.0,
    "online": 0.0,
    "rescale": 0.0,
}


def _release_plaintext_tree(value: object) -> None:
    if isinstance(value, (tuple, list)):
        for item in value:
            _release_plaintext_tree(item)
        return
    release_if_supported(value)


@dataclass
class _BabyEntry:
    #: The cipher this basis belongs to. Kept so a recycled ``id()`` cannot
    #: alias a stale entry onto a different ciphertext.
    source: object
    level: int
    rotations: object


class LinearOperator:
    """A reusable BSGS schedule with an operator-local baby cache."""

    def __init__(
        self,
        *,
        crypto_context,
        token_lanes: int,
        config: LinearOperatorConfig | None = None,
    ) -> None:
        self.crypto_context = crypto_context
        self.token_lanes = int(token_lanes)
        if self.token_lanes <= 0:
            raise ValueError("token_lanes must be positive.")
        self.config = LinearOperatorConfig() if config is None else config
        self._baby_cache: dict[tuple[int, tuple[int, ...]], _BabyEntry] = {}
        self._baby_hits = 0
        self._baby_misses = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def plan(self, dimension: int) -> LinearPlan:
        return self.config.plan(
            dimension=int(dimension), token_lanes=int(self.token_lanes)
        )

    def _baby_cache_key(
        self, cipher: object, plan: LinearPlan
    ) -> tuple[int, tuple[int, ...]]:
        return (
            id(cipher),
            plan.cache_key + (int(plan.baby_anchor_step),),
        )

    def _cached_baby_rotations(self, cipher: object, plan: LinearPlan):
        """Return the cached baby basis for ``cipher``, or None on a miss."""

        key = self._baby_cache_key(cipher, plan)
        level = int(self.crypto_context.level_for_cipher(cipher))
        entry = self._baby_cache.get(key)
        if entry is not None and entry.source is cipher and entry.level == level:
            self._baby_hits += 1
            return entry.rotations
        if entry is not None:
            self._release_baby_rotations(entry.rotations)
            del self._baby_cache[key]
        self._baby_misses += 1
        return None

    def _prepare_and_store_baby_rotations(
        self, cipher: object, plan: LinearPlan
    ) -> object:
        """Build the hoisted baby basis for one cipher and cache it."""

        rotations = fhe.prepare_hoisted_baby_rotations(
            cipher,
            plan.baby_offsets,
            self.crypto_context.context,
            strategy=str(self.config.hoist_strategy),
            baby_anchor_step=int(plan.baby_anchor_step),
        )
        self._baby_cache[self._baby_cache_key(cipher, plan)] = _BabyEntry(
            source=cipher,
            level=int(self.crypto_context.level_for_cipher(cipher)),
            rotations=rotations,
        )
        return rotations

    @staticmethod
    def _release_baby_rotations(rotations: object) -> None:
        for block in tuple(rotations):
            release_if_supported(block)

    def clear_baby_cache(self) -> None:
        for entry in self._baby_cache.values():
            self._release_baby_rotations(entry.rotations)
        self._baby_cache.clear()

    def cache_info(self) -> dict[str, int]:
        return {
            "baby_entries": len(self._baby_cache),
            "baby_hits": int(self._baby_hits),
            "baby_misses": int(self._baby_misses),
        }

    def prepare_baby_rotations(
        self,
        input_ciphers: tuple[object, ...] | list[object],
        *,
        dimension: int,
    ) -> dict[str, float]:
        """Populate the reusable baby basis before plaintext preparation.

        The cache is keyed by source object, level and physical BSGS plan, so
        later chunks and output pages borrow the same basis without
        recomputation. ``baby_precompute`` is the device wall time;
        ``baby_precompute_enqueue`` is the Python enqueue wall time, reported
        separately because the two diverge once the backend queues work.
        """

        if self.closed:
            raise RuntimeError("linear operator is closed.")
        input_ciphers = tuple(input_ciphers)
        profile = {
            "baby_precompute": 0.0,
            "baby_precompute_enqueue": 0.0,
            "baby_cache_hits": 0.0,
            "baby_cache_misses": 0.0,
        }
        if not input_ciphers or not bool(self.config.reuse_baby_rotations):
            return profile

        plan = self.plan(int(dimension))
        levels = {
            int(self.crypto_context.level_for_cipher(cipher))
            for cipher in input_ciphers
        }
        if len(levels) != 1:
            raise ValueError("linear inputs must enter at one common level.")

        hits_before = self._baby_hits
        misses_before = self._baby_misses
        missing = [
            cipher
            for cipher in input_ciphers
            if self._cached_baby_rotations(cipher, plan) is None
        ]

        start = time.perf_counter()
        for cipher in missing:
            self._prepare_and_store_baby_rotations(cipher, plan)
        enqueue_seconds = time.perf_counter() - start
        if missing:
            sync_device(self.crypto_context.device)
        profile["baby_precompute_enqueue"] = float(enqueue_seconds)
        profile["baby_precompute"] = float(
            time.perf_counter() - start if missing else 0.0
        )
        profile["baby_cache_hits"] = float(self._baby_hits - hits_before)
        profile["baby_cache_misses"] = float(
            self._baby_misses - misses_before
        )
        return profile

    def apply(
        self,
        input_ciphers: tuple[object, ...] | list[object],
        weight: LinearWeight,
    ) -> tuple[tuple[object, ...], dict[str, float]]:
        if self.closed:
            raise RuntimeError("linear operator is closed.")
        input_ciphers = tuple(input_ciphers)
        if not input_ciphers:
            return (), dict(_EMPTY_PROFILE)
        if weight.closed:
            raise RuntimeError("cannot apply a closed linear weight.")
        if weight.crypto_context is not self.crypto_context:
            raise ValueError("linear operator and weight must share one context.")

        plan = self.plan(weight.dimension)
        levels = {
            int(self.crypto_context.level_for_cipher(cipher))
            for cipher in input_ciphers
        }
        if len(levels) != 1:
            raise ValueError("linear inputs must enter at one common level.")
        input_level = levels.pop()
        strategy = str(self.config.hoist_strategy)
        stage = dict(_EMPTY_PROFILE)
        stage["weight_upload"] = float(weight.prepare(self.config))

        accumulators: list[object | None] = [None] * len(input_ciphers)
        outputs: list[object] = []
        cached_babies: list[object | None] = [None] * len(input_ciphers)
        # Always forward the configured two-parameter schedule.  When
        # Bstep <= anchorstep EasyFHE naturally creates one fast block; when
        # Bstep > anchorstep it chains anchors with the single anchor key.
        baby_anchor_step = int(plan.baby_anchor_step)
        try:
            if bool(self.config.reuse_baby_rotations):
                hits_before = self._baby_hits
                misses_before = self._baby_misses
                for index, cipher in enumerate(input_ciphers):
                    cached_babies[index] = self._cached_baby_rotations(
                        cipher, plan
                    )
                stage["baby_cache_hits"] = float(
                    self._baby_hits - hits_before
                )
                stage["baby_cache_misses"] = float(
                    self._baby_misses - misses_before
                )

            for chunk in plan.chunks:
                chunk_name = f"chunk_{int(chunk.index)}"
                reuse_encode = linear_encode_reuse_enabled()
                shared_plaintexts = None
                chunk_pack_encode = 0.0
                if reuse_encode:
                    start = time.perf_counter()
                    shared_plaintexts = weight.plaintext(
                        plan,
                        chunk,
                        level=input_level,
                        operator_config=self.config,
                    )
                    sync_device(self.crypto_context.device)
                    chunk_pack_encode = time.perf_counter() - start

                start = time.perf_counter()
                baby_event_start = baby_event_end = None
                baby_enqueue_seconds = 0.0
                missing_baby_indices = tuple(
                    index
                    for index, rotations in enumerate(cached_babies)
                    if rotations is None
                )
                if (
                    bool(self.config.reuse_baby_rotations)
                    and missing_baby_indices
                ):
                    if str(self.crypto_context.device).startswith("cuda"):
                        import easyfhe

                        baby_event_start = easyfhe.cuda.Event(
                            enable_timing=True
                        )
                        baby_event_end = easyfhe.cuda.Event(
                            enable_timing=True
                        )
                        baby_event_start.record()
                    baby_start = time.perf_counter()
                    for index in missing_baby_indices:
                        cached_babies[index] = (
                            self._prepare_and_store_baby_rotations(
                                input_ciphers[index], plan
                            )
                        )
                    baby_enqueue_seconds = time.perf_counter() - baby_start
                    if baby_event_end is not None:
                        baby_event_end.record()

                for index, cipher in enumerate(input_ciphers):
                    result = None
                    owned_plaintexts = None
                    try:
                        if reuse_encode:
                            plaintexts = shared_plaintexts
                        else:
                            encode_start = time.perf_counter()
                            owned_plaintexts = weight.plaintext(
                                plan,
                                chunk,
                                level=input_level,
                                operator_config=self.config,
                            )
                            sync_device(self.crypto_context.device)
                            chunk_pack_encode += (
                                time.perf_counter() - encode_start
                            )
                            plaintexts = owned_plaintexts
                        baby_rotations = cached_babies[index]
                        if (
                            bool(self.config.reuse_baby_rotations)
                            and baby_rotations is None
                        ):
                            raise RuntimeError(
                                "linear baby cache was not populated before MAC"
                            )
                        result = fhe.hoisted_mac_sum(
                            cipher,
                            plan.baby_offsets,
                            plaintexts,
                            plan.giant_offset,
                            int(chunk.giant_count),
                            self.crypto_context.context,
                            strategy=strategy,
                            baby_anchor_step=baby_anchor_step,
                            baby_rotations=baby_rotations,
                        )

                        if int(chunk.giant_base):
                            shifted = fhe.homo_rotate(
                                result,
                                int(chunk.giant_base) * plan.giant_offset,
                                self.crypto_context.context,
                            )
                            release_if_supported(result)
                            result = shifted
                        current = accumulators[index]
                        if current is None:
                            accumulators[index] = result
                            result = None
                        else:
                            combined = fhe.homo_add(
                                current, result, self.crypto_context.context
                            )
                            release_if_supported(current)
                            release_if_supported(result)
                            accumulators[index] = combined
                            result = None
                    finally:
                        release_if_supported(result)
                        if owned_plaintexts is not None:
                            _release_plaintext_tree(owned_plaintexts)
                sync_device(self.crypto_context.device)
                chunk_total = time.perf_counter() - start
                if baby_event_start is not None:
                    baby_precompute = float(
                        baby_event_start.elapsed_time(baby_event_end) / 1000.0
                    )
                else:
                    baby_precompute = float(baby_enqueue_seconds)
                stage["baby_precompute"] += baby_precompute
                stage["baby_precompute_enqueue"] += float(
                    baby_enqueue_seconds
                )
                encode_in_online = 0.0 if reuse_encode else float(
                    chunk_pack_encode
                )
                chunk_online = max(
                    0.0, chunk_total - baby_precompute - encode_in_online
                )
                stage["pack_encode"] += chunk_pack_encode
                stage[f"pack_encode.{chunk_name}"] = float(chunk_pack_encode)
                stage["online"] += chunk_online
                stage[f"online.{chunk_name}"] = float(chunk_online)

            start = time.perf_counter()
            for index, accumulator in enumerate(accumulators):
                if accumulator is None:
                    raise RuntimeError(
                        f"linear produced no accumulator for input {index}."
                    )
                try:
                    output = fhe.rescale(
                        accumulator, self.crypto_context.context
                    )
                finally:
                    release_if_supported(accumulator)
                    accumulators[index] = None
                outputs.append(output)
            sync_device(self.crypto_context.device)
            stage["rescale"] = time.perf_counter() - start

            result = tuple(outputs)
            outputs = []
            return result, {
                name: float(seconds) for name, seconds in stage.items()
            }
        finally:
            for accumulator in accumulators:
                release_if_supported(accumulator)
            for output in outputs:
                release_if_supported(output)

    def project(
        self,
        input_ciphers: tuple[object, ...] | list[object],
        weight: np.ndarray | StructuredLinearWeight,
        *,
        dtype: np.dtype,
    ) -> tuple[tuple[object, ...], dict[str, float]]:
        weight_prepare_start = time.perf_counter()
        with LinearWeight(
            weight,
            dtype=np.dtype(dtype),
            crypto_context=self.crypto_context,
        ) as prepared:
            weight_prepare_seconds = time.perf_counter() - weight_prepare_start
            outputs, profile = self.apply(input_ciphers, prepared)
            profile["weight_prepare_cpu"] = float(weight_prepare_seconds)
            return outputs, profile

    def close(self) -> None:
        self.clear_baby_cache()
        self._closed = True

    def __enter__(self) -> "LinearOperator":
        if self.closed:
            raise RuntimeError("cannot enter a closed linear operator.")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()
