from __future__ import annotations

import os

"""Transformer-layer definition and encrypted model execution."""

import time
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from . import graph
from .backend import plaintext_cache_snapshot, release_all
from .config import Llama3CKKSConfig

# Re-exported: the layer contract types live beside evaluate() in
# graph.py, but callers have always imported them from here.
from .graph import LayerApproximations, LayerWeights
from .layouts.attention import AttentionPairLayout
from .layouts.feature_major import FeatureMajorPrefillLayout
from .operators.nonlinear.bootstrap import BootstrapConfig, BootstrapOperator
from .schedule import LayerSchedule


def prepare_bootstrap_operator(crypto_context, schedule: LayerSchedule):
    """Prepare the one bootstrap program a run shares across layers.

    This is the single place a BootstrapConfig is built from a schedule;
    the application and the in-forward fallback both use it, so every
    bootstrap knob on LayerSchedule is honored on every path.
    """

    depth = int(crypto_context.context.L) - 1
    return BootstrapOperator.prepare(
        crypto_context,
        BootstrapConfig(
            log_slots=int(schedule.bootstrap_log_slots),
            level_budget=tuple(
                int(value) for value in schedule.bootstrap_level_budget
            ),
            output_levels=int(schedule.bootstrap_post_levels),
            iterations=int(schedule.softmax_bootstrap_iterations),
            precision_bits=int(schedule.softmax_bootstrap_precision),
            mode=os.environ.get("EASYFHE_BOOTSTRAP_MODE", "modraise_first"),
            evict_rotation_cache=bool(
                schedule.evict_rotation_cuda_cache_before_softmax_bootstrap
            ),
            deep_level_budget=(
                (3, 3) if schedule.softmax_main_deep_bootstrap else None
            ),
            deep_output_levels=(
                # Program depth = approx(10) + b1 + b2, so a (3,3) budget
                # fits two more output levels than (4,4) in the same chain.
                depth - 16 if schedule.softmax_main_deep_bootstrap else None
            ),
            slim_log_slots=schedule.rmsnorm2_slim_log_slots,
        ),
    )


@dataclass(frozen=True)
class TransformerLayer:
    """One indexed layer with no context, keygen, encryption or I/O effects."""

    index: int
    weights: LayerWeights
    approximations: LayerApproximations
    config: Llama3CKKSConfig
    feature_layout: FeatureMajorPrefillLayout = field(
        default_factory=FeatureMajorPrefillLayout
    )
    attention_layout: AttentionPairLayout = field(
        default_factory=AttentionPairLayout
    )
    rope_theta: float = 500000.0

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("layer index must be non-negative.")
        if not float(self.rope_theta) > 0:
            raise ValueError("rope_theta must be positive.")

    def forward(
        self,
        hidden_states: np.ndarray,
        *,
        crypto_context,
        input_ciphers: tuple[object, ...] | list[object],
        bootstrap_operator=None,
        schedule: LayerSchedule | None = None,
        refresh_input: bool = False,
        attention_token_valid_mask: np.ndarray | None = None,
        verify: bool = False,
        expected_output: np.ndarray | None = None,
    ) -> graph.LayerResult:
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        expected_shape = (
            int(self.feature_layout.seq_len),
            int(self.feature_layout.hidden_dim),
        )
        if hidden_states.shape != expected_shape:
            raise ValueError(
                f"layer hidden-state stub must have shape {expected_shape}, "
                f"got {hidden_states.shape}."
            )
        return graph.evaluate(
            hidden_states,
            layer_idx=int(self.index),
            weights=self.weights,
            approximations=self.approximations,
            attention_layout=self.attention_layout,
            config=self.config,
            crypto_context=crypto_context,
            input_ciphers=input_ciphers,
            bootstrap_operator=bootstrap_operator,
            schedule=schedule,
            refresh_input=bool(refresh_input),
            attention_token_valid_mask=attention_token_valid_mask,
            rope_theta=float(self.rope_theta),
            verify=bool(verify),
            expected_output=expected_output,
        )


@dataclass
class ModelResult:
    ciphers: tuple[object, ...]
    wall_seconds: float
    bootstrap_setup_seconds: float
    bootstrap_calls: int
    native_bootstrap_calls: int
    per_layer: list[dict[str, object]]

    def release(self) -> None:
        release_all(self.ciphers)
        self.ciphers = ()


@dataclass(frozen=True)
class Llama3Model:
    """A fixed-size model that consumes a streaming iterable of layer objects."""

    config: Llama3CKKSConfig
    num_layers: int = 32
    feature_layout: FeatureMajorPrefillLayout = field(
        default_factory=FeatureMajorPrefillLayout
    )
    attention_layout: AttentionPairLayout = field(
        default_factory=AttentionPairLayout
    )

    def __post_init__(self) -> None:
        if int(self.num_layers) <= 0:
            raise ValueError("num_layers must be positive.")

    def required_rotations(
        self,
        schedule: LayerSchedule | None = None,
        *,
        bootstrap_deep_level_budget: tuple[int, int] | None = None,
    ) -> tuple[int, ...]:
        return graph.required_rotations(
            self.feature_layout,
            self.attention_layout,
            schedule=schedule,
            log_n=int(self.config.simulator.logN),
            bootstrap_deep_level_budget=bootstrap_deep_level_budget,
        )

    def forward(
        self,
        hidden_states: np.ndarray,
        *,
        crypto_context,
        input_ciphers: tuple[object, ...] | list[object],
        layers: Iterable[TransformerLayer],
        bootstrap_operator: BootstrapOperator | None = None,
        schedule: LayerSchedule | None = None,
        refresh_first_layer: bool = False,
        attention_token_valid_mask: np.ndarray | None = None,
        progress_callback=None,
        diagnostic_callback=None,
    ) -> ModelResult:
        """Evaluate all layers without decrypting an inter-layer boundary.

        Layer zero consumes the caller's encrypted embedding.  Every later
        layer refreshes the previous FM output and fuses that refresh into its
        input RMSNorm.  The supplied layer iterable may therefore stream one
        checkpoint layer at a time.
        """

        schedule = LayerSchedule() if schedule is None else schedule
        # An absolute Layer0 alignment target cannot be reused after a
        # bootstrap has raised the modulus chain.
        layer_schedule = replace(schedule, qk_output_level=None)
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        expected_shape = (
            int(self.feature_layout.seq_len),
            int(self.feature_layout.hidden_dim),
        )
        if hidden_states.shape != expected_shape:
            raise ValueError(
                f"model hidden input must have shape {expected_shape}, "
                f"got {hidden_states.shape}."
            )
        current = tuple(input_ciphers)
        if len(current) != int(self.feature_layout.cipher_count):
            raise ValueError(
                f"model expects {self.feature_layout.cipher_count} input "
                f"ciphers, got {len(current)}."
            )

        owns_current = False
        owns_bootstrap = bootstrap_operator is None
        iterator = iter(layers)
        per_layer: list[dict[str, object]] = []
        setup_seconds = 0.0
        total_start = time.perf_counter()
        shape_stub = np.zeros(expected_shape, dtype=np.float32)
        try:
            if bootstrap_operator is None:
                bootstrap_operator = prepare_bootstrap_operator(
                    crypto_context, schedule
                )
                setup_seconds = float(bootstrap_operator.setup_seconds)
            bootstrap_calls_before = int(bootstrap_operator.calls)
            native_bootstrap_calls_before = int(
                bootstrap_operator.native_calls
            )

            first_layer_index: int | None = None
            for expected_index in range(int(self.num_layers)):
                try:
                    layer = next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        f"layer stream ended at {expected_index}; expected "
                        f"{self.num_layers} layers."
                    ) from exc
                if first_layer_index is None:
                    first_layer_index = int(layer.index)
                elif int(layer.index) != first_layer_index + expected_index:
                    raise ValueError(
                        "layer stream must be consecutive, expected index "
                        f"{first_layer_index + expected_index}, got "
                        f"{layer.index}."
                    )
                start = time.perf_counter()
                cache_before = plaintext_cache_snapshot(crypto_context)
                layer_result = None
                try:
                    layer_result = layer.forward(
                        hidden_states if expected_index == 0 else shape_stub,
                        crypto_context=crypto_context,
                        input_ciphers=current,
                        bootstrap_operator=bootstrap_operator,
                        schedule=layer_schedule,
                        refresh_input=(
                            expected_index > 0 or bool(refresh_first_layer)
                        ),
                        attention_token_valid_mask=attention_token_valid_mask,
                        verify=False,
                    )
                    compute_finished = time.perf_counter()
                    record_compute = getattr(
                        iterator, "record_compute_window", None
                    )
                    if callable(record_compute):
                        record_compute(expected_index, start, compute_finished)
                    next_ciphers = tuple(layer_result.ciphers)
                    if len(next_ciphers) != int(
                        self.feature_layout.cipher_count
                    ):
                        raise RuntimeError(
                            f"layer {expected_index} returned {len(next_ciphers)} "
                            "ciphertexts; the canonical feature-major boundary "
                            f"requires {self.feature_layout.cipher_count}."
                        )
                    layer_result.ciphers = ()
                    if owns_current:
                        release_all(current)
                    current = next_ciphers
                    owns_current = True
                    cache_after = plaintext_cache_snapshot(crypto_context)
                    per_layer.append(
                        {
                            "layer": expected_index,
                            "wall_seconds": float(compute_finished - start),
                            "kernel_wall_seconds": float(
                                layer_result.wall_seconds
                            ),
                            "stage_seconds": dict(layer_result.stage_seconds),
                            "nested_stage_seconds": dict(
                                layer_result.nested_stage_seconds
                            ),
                            "output_levels": list(layer_result.output_levels),
                            "softmax_bootstrap_calls": int(
                                layer_result.softmax_bootstrap_calls
                            ),
                            "residual_bootstrap_calls": int(
                                layer_result.residual_bootstrap_calls
                            ),
                            "softmax_native_bootstrap_calls": int(
                                layer_result.softmax_native_bootstrap_calls
                            ),
                            "residual_native_bootstrap_calls": int(
                                layer_result.residual_native_bootstrap_calls
                            ),
                            "plaintext_cache": {
                                "before": cache_before,
                                "after": cache_after,
                                "delta": {
                                    name: int(cache_after[name])
                                    - int(cache_before[name])
                                    for name in (
                                        "entries",
                                        "bytes",
                                        "hits",
                                        "misses",
                                        "skips",
                                    )
                                },
                            },
                        }
                    )
                    if callable(diagnostic_callback):
                        per_layer[-1]["value_diagnostic"] = dict(
                            diagnostic_callback(expected_index, current, layer)
                        )
                    if callable(progress_callback):
                        progress_callback(dict(per_layer[-1]))
                finally:
                    if layer_result is not None:
                        layer_result.release()

            output = current
            owns_current = False
            return ModelResult(
                ciphers=output,
                wall_seconds=float(time.perf_counter() - total_start),
                bootstrap_setup_seconds=float(setup_seconds),
                bootstrap_calls=(
                    int(bootstrap_operator.calls) - bootstrap_calls_before
                ),
                native_bootstrap_calls=(
                    int(bootstrap_operator.native_calls)
                    - native_bootstrap_calls_before
                ),
                per_layer=per_layer,
            )
        finally:
            close = getattr(iterator, "close", None)
            try:
                if callable(close):
                    close()
            finally:
                try:
                    if owns_current:
                        release_all(current)
                finally:
                    if owns_bootstrap and bootstrap_operator is not None:
                        bootstrap_operator.release()


__all__ = [
    "LayerApproximations",
    "LayerSchedule",
    "LayerWeights",
    "Llama3Model",
    "ModelResult",
    "TransformerLayer",
]
