from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import synchronize_device as sync_device
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout

from . import (
    LLAMA3_LINEAR_SCHEDULES,
    LinearOperatorConfig,
    linear_rotations,
    pair_real_ciphers,
)
from .qkv_kernel import (
    ComplexTokenDiagonalQKVResult,
    qkv_fhe_complex_token_diagonal,
)


@dataclass(frozen=True)
class FeatureMajorTokenPairComplexity:
    """Level-free Re/Im pairing at a canonical feature-major boundary."""

    input_cipher_count: int
    output_cipher_count: int
    additions: int
    imults: int
    multiplicative_depth: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


@dataclass
class FeatureMajorQKVResult:
    """QKV carriers projected directly from canonical real feature-major CTs."""

    projection: ComplexTokenDiagonalQKVResult
    pair_complexity: FeatureMajorTokenPairComplexity
    wall_seconds: float
    stage_seconds: dict[str, float]

    @property
    def carrier_ciphers(self) -> tuple[object, ...]:
        return self.projection.carrier_ciphers

    @property
    def output_layout(self):
        return self.projection.output_layout

    @property
    def carrier_width(self) -> int:
        return int(self.projection.carrier_width)

    @property
    def output_levels(self) -> tuple[int, ...]:
        return self.projection.output_levels

    @property
    def max_abs_diff(self) -> float | None:
        return self.projection.max_abs_diff

    def release(self) -> None:
        self.projection.release()


def token_pair_complexity(
    layout: FeatureMajorPrefillLayout,
) -> FeatureMajorTokenPairComplexity:
    if int(layout.cipher_count) % 2:
        raise ValueError(
            "complex-token pairing requires an even feature-major cipher count."
        )
    pairs = int(layout.cipher_count) // 2
    return FeatureMajorTokenPairComplexity(
        input_cipher_count=int(layout.cipher_count),
        output_cipher_count=pairs,
        additions=pairs,
        imults=pairs,
        multiplicative_depth=0,
    )


def qkv_rotations(
    layout: FeatureMajorPrefillLayout,
    *,
    operator_config: LinearOperatorConfig | None = None,
    extra_rotations: tuple[int, ...] | list[int] = (),
) -> tuple[int, ...]:
    """Rotation-key union for pairing plus feature-major diagonal QKV.

    Pairing itself uses only an ``i`` automorphism-free scalar operation and
    additions.  The returned set therefore consists of projection BSGS keys
    and the conjugation key used by the two-sided complex evaluation.
    """

    rotations = set(
        linear_rotations(
            dimension=int(layout.hidden_dim),
            slots=int(layout.slots),
            token_lanes=int(layout.tokens_per_cipher),
            include_conjugation=True,
            operator_config=(
                LLAMA3_LINEAR_SCHEDULES.qkv
                if operator_config is None
                else operator_config
            ),
        )
    )
    rotations.update(int(value) for value in extra_rotations if int(value))
    return tuple(sorted(rotations))


def pair_feature_major_ciphers(
    input_ciphers: tuple[object, ...] | list[object],
    *,
    layout: FeatureMajorPrefillLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Pack adjacent real token shards as ``even + i*odd`` without a level."""

    input_ciphers = tuple(input_ciphers)
    static = token_pair_complexity(layout)
    if len(input_ciphers) != static.input_cipher_count:
        raise ValueError(
            f"feature-major pairing expects {static.input_cipher_count} inputs, "
            f"got {len(input_ciphers)}."
        )
    levels = {
        int(crypto_context.level_for_cipher(cipher)) for cipher in input_ciphers
    }
    if len(levels) != 1:
        raise ValueError("feature-major pairing inputs must have one common level.")

    return pair_real_ciphers(input_ciphers, crypto_context=crypto_context)


def qkv_fhe(
    hidden_states: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    attention_layout: AttentionPairLayout,
    layout: FeatureMajorPrefillLayout | None = None,
    config: Llama3CKKSConfig | None = None,
    operator_config: LinearOperatorConfig | None = None,
    extra_rotations: tuple[int, ...] | list[int] = (),
    crypto_context=None,
    input_ciphers: tuple[object, ...] | list[object] | None = None,
    verify: bool = True,
    function_prefix: str = "model.layers.0.self_attn.feature_major_qkv",
    input_column_scale: np.ndarray | None = None,
) -> FeatureMajorQKVResult:
    """Run heterogeneous QKV from the long-lived real FM contract.

    The canonical boundary remains real.  Adjacent eight-token ciphertexts
    are paired only for the linear projection, after which the existing
    heterogeneous Q/K-feature and V/V complex carrier meanings take over.
    No layout transform and no multiplicative level are spent on the input
    pairing.
    """

    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.qkv
        if operator_config is None
        else operator_config
    )

    config = Llama3CKKSConfig() if config is None else config
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be rank two, got shape {hidden_states.shape}."
        )
    slots = 1 << (int(config.simulator.logN) - 1)
    layout = (
        FeatureMajorPrefillLayout(
            seq_len=int(hidden_states.shape[0]),
            hidden_dim=int(hidden_states.shape[1]),
            slots=slots,
        )
        if layout is None
        else layout
    )
    expected_shape = (int(layout.seq_len), int(layout.hidden_dim))
    if hidden_states.shape != expected_shape:
        raise ValueError(
            f"hidden_states must have shape {expected_shape}, got {hidden_states.shape}."
        )
    if int(layout.slots) != int(slots):
        raise ValueError(
            "feature-major layout/config slot mismatch: "
            f"layout.slots={layout.slots}, config slots={slots}."
        )
    if int(attention_layout.slots) != int(layout.slots):
        raise ValueError(
            "feature-major/attention slot mismatch: "
            f"{layout.slots} != {attention_layout.slots}."
        )
    pair_complexity = token_pair_complexity(layout)

    if crypto_context is None or input_ciphers is None:
        raise ValueError(
            "qkv_fhe requires an application-owned context "
            "and encrypted feature-major inputs."
        )

    if int(crypto_context.max_slots) != int(layout.slots):
        raise ValueError(
            "feature-major layout/context slot mismatch: "
            f"layout.slots={layout.slots}, context.max_slots={crypto_context.max_slots}."
        )

    wall_start = time.perf_counter()
    stage_seconds = {
        "context_setup": 0.0,
        "input_pair": 0.0,
    }
    owns_inputs = False
    real_inputs: tuple[object, ...] = ()
    paired_inputs: tuple[object, ...] = ()
    projection: ComplexTokenDiagonalQKVResult | None = None
    success = False
    try:
        real_inputs = tuple(input_ciphers)

        start = time.perf_counter()
        paired_inputs = pair_feature_major_ciphers(
            real_inputs, layout=layout, crypto_context=crypto_context
        )
        sync_device(crypto_context.device)
        stage_seconds["input_pair"] = time.perf_counter() - start

        projection = qkv_fhe_complex_token_diagonal(
            hidden_states,
            q_weight,
            k_weight,
            v_weight,
            attention_layout=attention_layout,
            config=config,
            operator_config=operator_config,
            extra_rotations=tuple(int(value) for value in extra_rotations),
            crypto_context=crypto_context,
            input_ciphers=paired_inputs,
            verify=bool(verify),
            function_prefix=f"{function_prefix}.projection",
            input_column_scale=input_column_scale,
        )
        for name, seconds in projection.stage_seconds.items():
            stage_seconds[f"projection.{name}"] = float(seconds)

        success = True
        return FeatureMajorQKVResult(
            projection=projection,
            pair_complexity=pair_complexity,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={str(key): float(value) for key, value in stage_seconds.items()},
        )
    finally:
        for cipher in paired_inputs:
            release_if_supported(cipher)
        if owns_inputs:
            for cipher in real_inputs:
                release_if_supported(cipher)
        if not success and projection is not None:
            projection.release()
