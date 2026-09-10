from __future__ import annotations

"""Fused residual refresh and feature-major RMSNorm."""

import time

from easyfhe import fhe

from llama3fhe.backend import release_if_supported

from ...approx.chebyshev import chebyshev_ps_mul_depth
from ..primitives import drop_levels
from .rmsnorm import (
    SlimRefresh,
    rmsnorm_complexity,
    rmsnorm_fhe,
)


def _half_rescale(cipher, *, crypto_context):
    encoded = fhe.encode_scalar(
        0.5,
        cur_limbs=int(cipher.state.cur_limbs),
        scale_degree=1,
        scaling_factor=cipher.state.scaling_factor,
        context=crypto_context.context,
    )
    return fhe.homo_mul_scalar_rescale(
        cipher, encoded, crypto_context.context
    )


def _pack_complex_pair(left, right, *, crypto_context):
    """Pack two real streams as ``left + i*right`` so one bootstrap serves both."""

    imaginary = fhe.homo_mul_i(right, crypto_context.context)
    try:
        return fhe.homo_add(left, imaginary, crypto_context.context)
    finally:
        release_if_supported(imaginary)


def _unpack_complex_pair(value, *, crypto_context):
    """Split a refreshed complex cipher into ``(2*real, 2*imaginary)``.

    The conjugate split inherently doubles both components; the caller decides
    whether to undo that with a public half-multiplication (``_half_rescale``)
    or to absorb the factor downstream, as fused RMSNorm does through
    ``input_value_scale``. ``value`` is not consumed.
    """

    conjugated = fhe.homo_rotate(
        value, int(crypto_context.context.M) - 1, crypto_context.context
    )
    try:
        real_twice = fhe.homo_add(value, conjugated, crypto_context.context)
        delta = fhe.homo_sub(conjugated, value, crypto_context.context)
        try:
            imaginary_twice = fhe.homo_mul_i(delta, crypto_context.context)
        except Exception:
            release_if_supported(real_twice)
            raise
        finally:
            release_if_supported(delta)
    finally:
        release_if_supported(conjugated)
    return real_twice, imaginary_twice


def adaptive_bootstrap_output_level_drop(
    *,
    current_limbs: int,
    requested_level_drop: int,
    rmsnorm_depth: int,
    minimum_output_limbs: int | None,
) -> int:
    """Clamp a cheap post-bootstrap modulus drop to its consumer budget."""

    current_limbs = int(current_limbs)
    requested_level_drop = int(requested_level_drop)
    rmsnorm_depth = int(rmsnorm_depth)
    if requested_level_drop < 0:
        raise ValueError("requested_level_drop must be non-negative.")
    if rmsnorm_depth < 0:
        raise ValueError("rmsnorm_depth must be non-negative.")
    if minimum_output_limbs is None:
        return requested_level_drop
    minimum_output_limbs = int(minimum_output_limbs)
    if minimum_output_limbs <= 1:
        raise ValueError("minimum_output_limbs must exceed one.")
    maximum_safe = max(
        0,
        current_limbs - rmsnorm_depth - minimum_output_limbs,
    )
    return min(requested_level_drop, maximum_safe)


def refresh_ciphers(
    input_ciphers,
    *,
    crypto_context,
    bootstrap_operator,
    iterations: int = 2,
    precision_bits: int = 14,
    function_prefix: str = "feature_major_refresh",
):
    """Refresh a real FM stream with pairwise complex bootstrap.

    Pairing halves the bootstrap calls.  The conjugate split produces twice
    each real value, so a final public 0.5 multiplication restores the exact
    canonical residual/linear boundary.
    """

    input_ciphers = tuple(input_ciphers)
    if len(input_ciphers) % 2:
        raise ValueError("complex-paired FM refresh requires an even stream.")
    outputs: list[object] = []
    try:
        for index in range(0, len(input_ciphers), 2):
            packed = _pack_complex_pair(
                input_ciphers[index],
                input_ciphers[index + 1],
                crypto_context=crypto_context,
            )
            try:
                value = bootstrap_operator.refresh(
                    packed,
                    iterations=int(iterations),
                    precision_bits=int(precision_bits),
                )
            finally:
                release_if_supported(packed)
            try:
                doubled = _unpack_complex_pair(
                    value, crypto_context=crypto_context
                )
            finally:
                release_if_supported(value)
            try:
                for component in doubled:
                    outputs.append(
                        _half_rescale(
                            component, crypto_context=crypto_context
                        )
                    )
            finally:
                for component in doubled:
                    release_if_supported(component)
        return tuple(outputs)
    except Exception:
        for cipher in outputs:
            release_if_supported(cipher)
        raise


def bootstrap_rmsnorm_fhe(
    input_ciphers,
    weight,
    *,
    coeffs,
    layout,
    crypto_context,
    bootstrap_operator,
    pair_inputs: bool = True,
    bootstrap_iterations: int = 2,
    bootstrap_precision_bits: int = 14,
    bootstrap_output_level_drop: int = 0,
    minimum_rmsnorm_output_limbs: int | None = None,
    return_refreshed_inputs: bool = True,
    defer_output_scale: bool = False,
    adaptive_slim_refresh: bool = False,
    use_slim_program: bool = False,
    function_prefix: str = "bootstrap_rmsnorm",
):
    """Refresh a real FM stream and immediately normalize it.

    Complex-pair refresh returns ``2*a`` and ``2*b``.  That implementation
    detail is absorbed by RMSNorm here, so neither the model nor the next
    operator observes or configures an ``input_value_scale`` contract.  A
    caller that keeps the pre-bootstrap identity branch alive may disable
    ``return_refreshed_inputs`` and avoid the otherwise residual-only copy
    (including one half-rescale per real stream in paired mode).
    """

    input_ciphers = tuple(input_ciphers)
    if bool(pair_inputs) and len(input_ciphers) % 2:
        raise ValueError("complex-paired bootstrap requires an even stream.")
    requested_level_drop = int(bootstrap_output_level_drop)
    if requested_level_drop < 0:
        raise ValueError("bootstrap_output_level_drop must be non-negative.")
    if minimum_rmsnorm_output_limbs is not None:
        minimum_rmsnorm_output_limbs = int(minimum_rmsnorm_output_limbs)
        if minimum_rmsnorm_output_limbs <= 1:
            raise ValueError(
                "minimum_rmsnorm_output_limbs must exceed one."
            )
    polynomial_depth = chebyshev_ps_mul_depth(coeffs["coefficients"])
    rmsnorm_depth = (
        rmsnorm_complexity(
            layout,
            polynomial_degree=len(coeffs["coefficients"]) - 1,
            apply_gamma=not bool(defer_output_scale),
        ).explicit_multiplicative_depth_excluding_polynomial
        + int(polynomial_depth)
    )

    def effective_level_drop(value) -> int:
        return adaptive_bootstrap_output_level_drop(
            current_limbs=int(value.state.cur_limbs),
            requested_level_drop=requested_level_drop,
            rmsnorm_depth=rmsnorm_depth,
            minimum_output_limbs=minimum_rmsnorm_output_limbs,
        )

    start = time.perf_counter()
    modulus_drop_seconds = 0.0
    effective_level_drops: list[int] = []
    refreshed: list[object] = []
    residual_inputs: list[object] = []
    result = None
    try:
        def refresh_and_drop(cipher):
            """Refresh one cipher, then apply the clamped modulus drop."""

            nonlocal modulus_drop_seconds
            value = bootstrap_operator.refresh(
                cipher,
                iterations=int(bootstrap_iterations),
                precision_bits=int(bootstrap_precision_bits),
            )
            level_drop = effective_level_drop(value)
            effective_level_drops.append(level_drop)
            if not level_drop:
                return value
            drop_start = time.perf_counter()
            try:
                dropped = drop_levels(
                    (value,), level_drop, crypto_context=crypto_context
                )[0]
            except Exception:
                release_if_supported(value)
                raise
            if dropped is not value:
                release_if_supported(value)
            modulus_drop_seconds += time.perf_counter() - drop_start
            return dropped

        if bool(pair_inputs):
            for index in range(0, len(input_ciphers), 2):
                packed = _pack_complex_pair(
                    input_ciphers[index],
                    input_ciphers[index + 1],
                    crypto_context=crypto_context,
                )
                try:
                    value = refresh_and_drop(packed)
                finally:
                    release_if_supported(packed)
                try:
                    refreshed.extend(
                        _unpack_complex_pair(
                            value, crypto_context=crypto_context
                        )
                    )
                finally:
                    release_if_supported(value)
            # The conjugate split doubles both components; RMSNorm absorbs the
            # factor through input_value_scale instead of spending a level.
            input_value_scale = 2.0
            if bool(return_refreshed_inputs):
                for cipher in refreshed:
                    residual_inputs.append(
                        _half_rescale(cipher, crypto_context=crypto_context)
                    )
        else:
            for cipher in input_ciphers:
                refreshed.append(refresh_and_drop(cipher))
            input_value_scale = 1.0
            if bool(return_refreshed_inputs):
                for cipher in refreshed:
                    residual_inputs.append(cipher.deep_copy())
        bootstrap_seconds = time.perf_counter() - start
        result = rmsnorm_fhe(
            tuple(refreshed),
            weight,
            coeffs=coeffs,
            layout=layout,
            crypto_context=crypto_context,
            input_value_scale=input_value_scale,
            defer_output_scale=bool(defer_output_scale),
            slim_refresh=(
                SlimRefresh(
                    bootstrap_operator=bootstrap_operator,
                    minimum_output_limbs=minimum_rmsnorm_output_limbs,
                    iterations=int(bootstrap_iterations),
                    precision_bits=int(bootstrap_precision_bits),
                    use_slim_program=bool(use_slim_program),
                )
                if adaptive_slim_refresh and minimum_rmsnorm_output_limbs
                else None
            ),
            verify=False,
            function_prefix=function_prefix,
        )
        result.stage_seconds = {
            "bootstrap": float(bootstrap_seconds - modulus_drop_seconds),
            "bootstrap_output_modulus_drop": float(modulus_drop_seconds),
            **result.stage_seconds,
        }
        # Level counts are not durations, so they stay out of stage_seconds:
        # every aggregator over that dict sums its values as seconds.
        result.level_schedule = {
            "bootstrap_output_level_drop_requested": int(requested_level_drop),
            "bootstrap_output_level_drop_effective": int(
                min(effective_level_drops, default=0)
            ),
            "minimum_rmsnorm_output_limbs": int(
                minimum_rmsnorm_output_limbs or 0
            ),
        }
        result.wall_seconds += float(bootstrap_seconds)
        result.bootstrap_output_levels = tuple(
            sorted(
                {
                    int(crypto_context.level_for_cipher(cipher))
                    for cipher in refreshed
                }
            )
        )
        result.refreshed_input_ciphers = tuple(residual_inputs)
        residual_inputs = []
        return result
    finally:
        for cipher in refreshed:
            release_if_supported(cipher)
        for cipher in residual_inputs:
            release_if_supported(cipher)


__all__ = [
    "adaptive_bootstrap_output_level_drop",
    "bootstrap_rmsnorm_fhe",
    "refresh_ciphers",
]
