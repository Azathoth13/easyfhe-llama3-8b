from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import synchronize_device as sync_device
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout

from ...approx.rmsnorm import is_rmsn_weight, rmsnorm_poly_numpy
from ..nonlinear.polynomial import eval_chebyshev_series_cipher
from ..primitives import (
    mul_cipher_rescale as _mul_cipher_rescale,
)
from ..primitives import (
    mul_plain_rescale as _mul_plain_rescale,
)
from ..primitives import (
    square_rescale as _square_rescale,
)


@dataclass(frozen=True)
class SlimRefresh:
    """Policy for refreshing the slim rsqrt input when its budget runs out.

    The four values only mean anything together, so a caller enables the
    behavior by supplying this object and disables it by passing ``None``
    instead of nulling individual parameters.
    """

    bootstrap_operator: object
    minimum_output_limbs: int
    iterations: int = 2
    precision_bits: int = 14
    use_slim_program: bool = False

    def __post_init__(self) -> None:
        if int(self.minimum_output_limbs) <= 1:
            raise ValueError("minimum_output_limbs must exceed one.")
        if int(self.iterations) not in (1, 2):
            raise ValueError("slim refresh iterations must be one or two.")
        if int(self.precision_bits) <= 0:
            raise ValueError("slim refresh precision_bits must be positive.")


@dataclass(frozen=True)
class FeatureMajorRMSNormComplexity:
    input_cipher_count: int
    mean_square_ct_ct_multiplications: int
    mean_square_relinearizations: int
    mean_square_rotations: int
    slim_pack_pt_ct_multiplications: int
    slim_pack_rotations: int
    slim_pack_additions: int
    rsqrt_polynomial_evaluations: int
    rsqrt_polynomial_degree: int
    slim_extract_pt_ct_multiplications: int
    slim_extract_rotations: int
    scale_broadcast_rotations: int
    scale_broadcast_additions: int
    output_ct_ct_multiplications: int
    output_relinearizations: int
    gamma_pt_ct_multiplications: int
    explicit_multiplicative_depth_excluding_polynomial: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


@dataclass
class FeatureMajorRMSNormResult:
    ciphers: tuple[object, ...]
    complexity: FeatureMajorRMSNormComplexity
    wall_seconds: float
    stage_seconds: dict[str, float]
    output_levels: tuple[int, ...]
    deferred_output_scale: np.ndarray | None = None
    # Level/limb counts from a fused bootstrap refresh. Separate from
    # stage_seconds because these are not durations.
    level_schedule: dict[str, int] = field(default_factory=dict)
    refreshed_input_ciphers: tuple[object, ...] = ()
    bootstrap_output_levels: tuple[int, ...] = ()
    slim_bootstrap_output_levels: tuple[int, ...] = ()
    output: np.ndarray | None = None
    expected: np.ndarray | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None

    def release(self) -> None:
        for cipher in self.ciphers:
            release_if_supported(cipher)
        for cipher in self.refreshed_input_ciphers:
            release_if_supported(cipher)
        self.ciphers = ()
        self.refreshed_input_ciphers = ()


@dataclass
class FeatureMajorResidualResult:
    ciphers: tuple[object, ...]
    wall_seconds: float
    output_levels: tuple[int, ...]
    #: Pairs that entered at different levels and needed a modulus alignment.
    #: A runtime observation, unlike the static residual counts, which
    #: graph.operation_audit derives from the layout.
    modulus_aligned_pairs: int = 0
    output: np.ndarray | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None

    def release(self) -> None:
        for cipher in self.ciphers:
            release_if_supported(cipher)


def rmsnorm_rotations(
    layout: FeatureMajorPrefillLayout,
) -> tuple[int, ...]:
    """Rotation keys for feature reduction and slim-scale broadcast."""

    stride = int(layout.tokens_per_cipher)
    offsets = {
        stride * (1 << bit)
        for bit in range(int(math.log2(int(layout.hidden_dim))))
    }
    rotations = offsets | {-offset for offset in offsets}
    return tuple(sorted(int(offset) for offset in rotations if int(offset)))


def rmsnorm_complexity(
    layout: FeatureMajorPrefillLayout,
    *,
    polynomial_degree: int,
    apply_gamma: bool = True,
) -> FeatureMajorRMSNormComplexity:
    """Static count for the direct-row-mask slim RMSNorm schedule."""

    ciphers = int(layout.cipher_count)
    reduction_steps = int(math.log2(int(layout.hidden_dim)))
    return FeatureMajorRMSNormComplexity(
        input_cipher_count=ciphers,
        mean_square_ct_ct_multiplications=ciphers,
        mean_square_relinearizations=ciphers,
        mean_square_rotations=ciphers * reduction_steps,
        # Reduced values already repeat over every feature row.  Source c is
        # selected directly into slim row c, so no pack rotation is needed.
        slim_pack_pt_ct_multiplications=ciphers,
        slim_pack_rotations=0,
        slim_pack_additions=max(0, ciphers - 1),
        rsqrt_polynomial_evaluations=1,
        rsqrt_polynomial_degree=int(polynomial_degree),
        slim_extract_pt_ct_multiplications=ciphers,
        # A selected scale starts in row c and cyclic doubling broadcasts it
        # over the complete feature orbit; it need not first move to row zero.
        slim_extract_rotations=0,
        scale_broadcast_rotations=ciphers * reduction_steps,
        scale_broadcast_additions=ciphers * reduction_steps,
        output_ct_ct_multiplications=ciphers,
        output_relinearizations=ciphers,
        gamma_pt_ct_multiplications=ciphers if bool(apply_gamma) else 0,
        # square + slim pack mask + slim extract + input*scale (+ gamma)
        explicit_multiplicative_depth_excluding_polynomial=4
        + int(bool(apply_gamma)),
    )


def _plaintext(values: np.ndarray, *, name: str, cipher, crypto_context):
    return crypto_context.plaintext(
        np.asarray(values, dtype=np.float64),
        name=str(name),
        level=int(crypto_context.level_for_cipher(cipher)),
        slots=int(crypto_context.max_slots),
        dtype=np.float64,
    )


def _fold_feature_orbit(
    cipher,
    *,
    layout: FeatureMajorPrefillLayout,
    crypto_context,
    ascending: bool,
):
    """Rotate-and-add over the feature orbit, doubling the stride each step.

    ``ascending`` sums every feature into each slot (the mean-square
    reduction); descending broadcasts one slim row back over the whole orbit.
    The two directions are the same log2(hidden_dim) rotation tree with the
    rotation sign flipped, and both consume ``cipher`` only if they own it.
    """

    current = cipher
    owns_current = False
    stride = int(layout.tokens_per_cipher)
    sign = 1 if bool(ascending) else -1
    try:
        for bit in range(int(math.log2(int(layout.hidden_dim)))):
            rotated = crypto_context.fhe.homo_rotate(
                current, sign * stride * (1 << bit), crypto_context.context
            )
            try:
                next_current = crypto_context.fhe.homo_add(
                    current, rotated, crypto_context.context
                )
            finally:
                release_if_supported(rotated)
                if owns_current:
                    release_if_supported(current)
            current = next_current
            owns_current = True
        return current
    except Exception:
        if owns_current:
            release_if_supported(current)
        raise


def _slim_row_mask(
    layout: FeatureMajorPrefillLayout,
    cipher_index: int,
    *,
    value: float,
) -> np.ndarray:
    values = np.zeros((int(layout.slots),), dtype=np.float64)
    for local_token in range(layout.active_tokens(cipher_index)):
        values[layout.slot(int(cipher_index), local_token)] = float(value)
    return values


def _mapped_active_offset(
    layout: FeatureMajorPrefillLayout, value: float
) -> np.ndarray:
    values = np.zeros((int(layout.slots),), dtype=np.float64)
    for row in range(int(layout.cipher_count)):
        for local_token in range(layout.active_tokens(row)):
            values[layout.slot(row, local_token)] = float(value)
    return values


def _add_at_common_level(left, right, *, crypto_context):
    left_level = int(crypto_context.level_for_cipher(left))
    right_level = int(crypto_context.level_for_cipher(right))
    if left_level == right_level:
        return crypto_context.fhe.homo_add(
            left, right, crypto_context.context
        )
    if left_level < right_level:
        aligned = crypto_context.fhe.align_to(
            left, right.state, crypto_context.context
        )
        try:
            return crypto_context.fhe.homo_add(
                aligned, right, crypto_context.context
            )
        finally:
            if aligned is not left:
                release_if_supported(aligned)
    aligned = crypto_context.fhe.align_to(
        right, left.state, crypto_context.context
    )
    try:
        return crypto_context.fhe.homo_add(
            left, aligned, crypto_context.context
        )
    finally:
        if aligned is not right:
            release_if_supported(aligned)


def residual_add(
    left_ciphers: tuple[object, ...] | list[object],
    right_ciphers: tuple[object, ...] | list[object],
    *,
    layout: FeatureMajorPrefillLayout,
    crypto_context,
    verify: bool = False,
    expected_left: np.ndarray | None = None,
    expected_right: np.ndarray | None = None,
) -> FeatureMajorResidualResult:
    """Add two canonical feature-major streams without changing layout."""

    left_ciphers = tuple(left_ciphers)
    right_ciphers = tuple(right_ciphers)
    expected_count = int(layout.cipher_count)
    if len(left_ciphers) != expected_count or len(right_ciphers) != expected_count:
        raise ValueError(
            "feature-major residual add expects "
            f"{expected_count} ciphers on each side, got "
            f"{len(left_ciphers)} and {len(right_ciphers)}."
        )
    if bool(verify) and (expected_left is None or expected_right is None):
        raise ValueError("verify=True requires expected_left and expected_right.")

    start = time.perf_counter()
    outputs: list[object] = []
    try:
        alignment_count = 0
        for left, right in zip(left_ciphers, right_ciphers, strict=True):
            alignment_count += int(
                crypto_context.level_for_cipher(left)
            ) != int(crypto_context.level_for_cipher(right))
            outputs.append(
                _add_at_common_level(
                    left, right, crypto_context=crypto_context
                )
            )
        sync_device(crypto_context.device)
        wall_seconds = time.perf_counter() - start

        output = None
        max_abs_diff = mean_abs_diff = None
        if bool(verify):
            decoded = np.stack(
                [
                    np.asarray(crypto_context.decrypt(cipher)).real
                    for cipher in outputs
                ]
            )
            output = layout.unpack(decoded)
            expected = np.asarray(expected_left, dtype=np.float32) + np.asarray(
                expected_right, dtype=np.float32
            )
            difference = output - expected
            max_abs_diff = float(np.max(np.abs(difference)))
            mean_abs_diff = float(np.mean(np.abs(difference)))

        return FeatureMajorResidualResult(
            ciphers=tuple(outputs),
            wall_seconds=float(wall_seconds),
            modulus_aligned_pairs=int(alignment_count),
            output_levels=tuple(
                sorted(
                    {
                        int(crypto_context.level_for_cipher(cipher))
                        for cipher in outputs
                    }
                )
            ),
            output=output,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
        )
    except Exception:
        for cipher in outputs:
            release_if_supported(cipher)
        raise


def rmsnorm_fhe(
    input_ciphers: tuple[object, ...] | list[object],
    weight: np.ndarray,
    *,
    coeffs: dict,
    layout: FeatureMajorPrefillLayout,
    crypto_context,
    input_value_scale: float = 1.0,
    defer_output_scale: bool = False,
    slim_refresh: SlimRefresh | None = None,
    verify: bool = False,
    expected_input: np.ndarray | None = None,
    function_prefix: str = "model.layers.0.feature_major_rmsnorm",
) -> FeatureMajorRMSNormResult:
    """Ciphertext-only RMSNorm while preserving canonical feature-major slots.

    All 128 token mean squares are packed into one slim ciphertext, so the
    expensive inverse-square-root polynomial is evaluated exactly once.
    """

    input_ciphers = tuple(input_ciphers)
    if len(input_ciphers) != int(layout.cipher_count):
        raise ValueError(
            "feature-major RMSNorm expects "
            f"{layout.cipher_count} input ciphers, got {len(input_ciphers)}."
        )
    if int(layout.cipher_count) > int(layout.hidden_dim):
        raise ValueError(
            "slim feature-major RMSNorm requires cipher_count <= hidden_dim."
        )
    if int(layout.slots) != int(crypto_context.max_slots):
        raise ValueError(
            "feature-major RMSNorm requires layout.slots == context.max_slots, "
            f"got {layout.slots}/{crypto_context.max_slots}."
        )

    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    apply_gamma = not is_rmsn_weight(weight)
    if apply_gamma and weight.shape != (int(layout.hidden_dim),):
        raise ValueError(
            f"RMSNorm gamma must have width {layout.hidden_dim}, got {weight.shape}."
        )
    if bool(verify) and expected_input is None:
        raise ValueError("verify=True requires expected_input.")
    input_value_scale = float(input_value_scale)
    if not np.isfinite(input_value_scale) or input_value_scale <= 0.0:
        raise ValueError(
            "input_value_scale must be finite and positive, got "
            f"{input_value_scale}."
        )
    defer_output_scale = bool(defer_output_scale)
    if (
        not defer_output_scale
        and not apply_gamma
        and not math.isclose(input_value_scale, 1.0)
    ):
        raise ValueError(
            "scaled RMSNorm input requires gamma so the inverse public scale "
            "can be fused without another level."
        )

    coefficients = np.asarray(coeffs["coefficients"], dtype=np.float64)
    fit_lo, fit_hi = (float(value) for value in coeffs["fit_interval"])
    alpha = 2.0 / (fit_hi - fit_lo)
    beta = -1.0 - 2.0 * fit_lo / (fit_hi - fit_lo)
    output_scale = (
        weight.astype(np.float64, copy=False)
        if apply_gamma
        else np.ones((int(layout.hidden_dim),), dtype=np.float64)
    ) / input_value_scale
    apply_output_scale = not defer_output_scale and (
        apply_gamma or not math.isclose(input_value_scale, 1.0)
    )
    complexity = rmsnorm_complexity(
        layout,
        polynomial_degree=max(0, int(coefficients.size) - 1),
        apply_gamma=apply_output_scale,
    )

    wall_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    slim_terms: list[object] = []
    slim = mapped = polynomial = None
    slim_bootstrap_output_levels: tuple[int, ...] = ()
    broadcasts: list[object] = []
    outputs: list[object] = []
    success = False
    try:
        start = time.perf_counter()
        for cipher_index, cipher in enumerate(input_ciphers):
            squared = _square_rescale(cipher, crypto_context=crypto_context)
            reduced = _fold_feature_orbit(
                squared,
                layout=layout,
                crypto_context=crypto_context,
                ascending=True,
            )
            release_if_supported(squared)
            mask = _plaintext(
                _slim_row_mask(
                    layout,
                    cipher_index,
                    value=(
                        alpha
                        / float(layout.hidden_dim)
                        / (input_value_scale * input_value_scale)
                    ),
                ),
                name=f"{function_prefix}.slim_pack.{cipher_index}",
                cipher=reduced,
                crypto_context=crypto_context,
            )
            try:
                term = _mul_plain_rescale(
                    reduced, mask, crypto_context=crypto_context
                )
            finally:
                release_if_supported(mask)
                release_if_supported(reduced)
            slim_terms.append(term)
        for term in slim_terms:
            if slim is None:
                slim = term.deep_copy()
            else:
                next_slim = crypto_context.fhe.homo_add(
                    slim, term, crypto_context.context
                )
                release_if_supported(slim)
                slim = next_slim
        if slim is None:
            raise RuntimeError(
                "feature-major RMSNorm slim pack produced no ciphertext."
            )
        for term in slim_terms:
            release_if_supported(term)
        slim_terms = []
        sync_device(crypto_context.device)
        stage_seconds["mean_square_and_slim_pack"] = (
            time.perf_counter() - start
        )

        start = time.perf_counter()
        offset = _plaintext(
            _mapped_active_offset(layout, beta),
            name=f"{function_prefix}.mapped_beta",
            cipher=slim,
            crypto_context=crypto_context,
        )
        try:
            mapped = crypto_context.fhe.homo_add_pt(
                slim, offset, crypto_context.context
            )
        finally:
            release_if_supported(offset)
            release_if_supported(slim)
            slim = None
        polynomial = eval_chebyshev_series_cipher(
            mapped,
            coefficients,
            lower_bound=-1.0,
            upper_bound=1.0,
            crypto_context=crypto_context,
        )
        release_if_supported(mapped)
        mapped = None
        sync_device(crypto_context.device)
        stage_seconds["rsqrt_polynomial"] = time.perf_counter() - start
        remaining_depth = 2 + int(apply_output_scale)
        required_polynomial_limbs = (
            None
            if slim_refresh is None
            else int(slim_refresh.minimum_output_limbs) + remaining_depth
        )
        if (
            required_polynomial_limbs is not None
            and int(polynomial.state.cur_limbs) < required_polynomial_limbs
        ):
            bootstrap_start = time.perf_counter()
            previous = polynomial
            polynomial = slim_refresh.bootstrap_operator.refresh(
                previous,
                iterations=int(slim_refresh.iterations),
                precision_bits=int(slim_refresh.precision_bits),
                use_slim_program=bool(slim_refresh.use_slim_program),
            )
            if polynomial is not previous:
                release_if_supported(previous)
            slim_bootstrap_output_levels = (
                int(crypto_context.level_for_cipher(polynomial)),
            )
            stage_seconds["rsqrt_slim_bootstrap"] = (
                time.perf_counter() - bootstrap_start
            )
            excess_limbs = max(
                0,
                int(polynomial.state.cur_limbs)
                - int(required_polynomial_limbs),
            )
            if excess_limbs:
                drop_start = time.perf_counter()
                target_state = polynomial.state.replace(
                    cur_limbs=int(required_polynomial_limbs),
                    scale_degree=1,
                    scaling_factor=None,
                )
                dropped = crypto_context.fhe.align_to(
                    polynomial, target_state, crypto_context.context
                )
                if dropped is not polynomial:
                    release_if_supported(polynomial)
                polynomial = dropped
                stage_seconds["rsqrt_slim_bootstrap_modulus_drop"] = (
                    time.perf_counter() - drop_start
                )
        sync_device(crypto_context.device)

        start = time.perf_counter()
        for cipher_index in range(int(layout.cipher_count)):
            mask = _plaintext(
                _slim_row_mask(
                    layout, cipher_index, value=1.0
                ),
                name=f"{function_prefix}.slim_extract.{cipher_index}",
                cipher=polynomial,
                crypto_context=crypto_context,
            )
            try:
                selected = _mul_plain_rescale(
                    polynomial, mask, crypto_context=crypto_context
                )
            finally:
                release_if_supported(mask)
            broadcasts.append(
                _fold_feature_orbit(
                    selected,
                    layout=layout,
                    crypto_context=crypto_context,
                    ascending=False,
                )
            )
            release_if_supported(selected)
        release_if_supported(polynomial)
        polynomial = None
        sync_device(crypto_context.device)
        stage_seconds["slim_extract_broadcast"] = (
            time.perf_counter() - start
        )

        start = time.perf_counter()
        scaled = [
            _mul_cipher_rescale(
                left, right, crypto_context=crypto_context
            )
            for left, right in zip(input_ciphers, broadcasts, strict=True)
        ]
        for broadcast in broadcasts:
            release_if_supported(broadcast)
        broadcasts = []
        if apply_output_scale:
            gamma_values = np.repeat(
                output_scale,
                int(layout.tokens_per_cipher),
            )
            gamma = _plaintext(
                gamma_values,
                name=f"{function_prefix}.gamma",
                cipher=scaled[0],
                crypto_context=crypto_context,
            )
            try:
                outputs = [
                    _mul_plain_rescale(
                        cipher, gamma, crypto_context=crypto_context
                    )
                    for cipher in scaled
                ]
            finally:
                release_if_supported(gamma)
                for cipher in scaled:
                    release_if_supported(cipher)
        else:
            outputs = scaled
        sync_device(crypto_context.device)
        stage_seconds["scale_and_gamma"] = time.perf_counter() - start

        output = expected = None
        max_abs_diff = mean_abs_diff = None
        if bool(verify):
            start = time.perf_counter()
            decoded = np.stack(
                [
                    np.asarray(crypto_context.decrypt(cipher)).real
                    for cipher in outputs
                ]
            )
            output = layout.unpack(decoded)
            expected = rmsnorm_poly_numpy(
                np.asarray(expected_input, dtype=np.float32),
                coefficients,
                (fit_lo, fit_hi),
                skip_clamp=True,
            )
            if apply_output_scale:
                expected = (expected * weight).astype(np.float32, copy=False)
            elif defer_output_scale:
                expected = (expected * input_value_scale).astype(
                    np.float32, copy=False
                )
            difference = output - expected
            max_abs_diff = float(np.max(np.abs(difference)))
            mean_abs_diff = float(np.mean(np.abs(difference)))
            sync_device(crypto_context.device)
            stage_seconds["verify_decrypt"] = time.perf_counter() - start

        success = True
        return FeatureMajorRMSNormResult(
            ciphers=tuple(outputs),
            complexity=complexity,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={
                key: float(value) for key, value in stage_seconds.items()
            },
            output_levels=tuple(
                sorted(
                    {
                        int(crypto_context.level_for_cipher(cipher))
                        for cipher in outputs
                    }
                )
            ),
            deferred_output_scale=(
                output_scale.astype(np.float32, copy=False)
                if defer_output_scale
                else None
            ),
            slim_bootstrap_output_levels=slim_bootstrap_output_levels,
            output=output,
            expected=expected,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
        )
    finally:
        for term in slim_terms:
            release_if_supported(term)
        release_if_supported(slim)
        release_if_supported(mapped)
        release_if_supported(polynomial)
        for broadcast in broadcasts:
            release_if_supported(broadcast)
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)
