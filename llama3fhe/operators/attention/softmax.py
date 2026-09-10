from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import synchronize_device as sync_device
from llama3fhe.layouts.attention import AttentionPairLayout

from ...approx.chebyshev import chebyshev_ps_mul_depth
from ..linear import pair_real_ciphers, split_complex_ciphers_twice
from ..nonlinear.bootstrap import BootstrapOperator
from ..primitives import (
    add_owned as _add_owned,
)
from ..primitives import (
    bundle_plain as _bundle_plain,
)
from ..primitives import (
    mul_cipher_rescale as _mul_cipher_rescale,
)
from ..primitives import (
    mul_plain_rescale as _mul_plain_rescale,
)
from ..primitives import (
    square_rescale as _square_rescale,
)
from .config import AttentionOperatorConfig


def _softmax_diagnostic_range(label: str, ciphers, *, crypto_context) -> None:
    """Print decrypted ranges only for an explicitly requested debug run."""

    if os.environ.get("LLAMA_FHE_STAGE_DIAGNOSTICS", "") != "1":
        return
    values = np.stack(
        [np.asarray(crypto_context.decrypt(cipher)) for cipher in ciphers]
    )
    finite = np.isfinite(values)
    finite_values = values[finite]
    levels = sorted(
        {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    )
    print(
        f"[softmax-diagnostic] stage={label} "
        f"finite={bool(np.all(finite))} "
        f"real_max={None if not finite_values.size else float(np.max(np.abs(finite_values.real)))} "
        f"imag_max={None if not finite_values.size else float(np.max(np.abs(finite_values.imag)))} "
        f"real_mean={None if not finite_values.size else float(np.mean(np.abs(finite_values.real)))} "
        f"levels={levels}",
        flush=True,
    )


@dataclass
class DeltaSoftmaxAlg2Result:
    probability_ciphers: tuple[object, ...]
    stage_seconds: dict[str, float]
    output_levels: tuple[int, ...]

    def release(self) -> None:
        for cipher in self.probability_ciphers:
            release_if_supported(cipher)


def delta_softmax_row_rotations(layout: AttentionPairLayout) -> tuple[int, ...]:
    stride = 2 * int(layout.seq_len)
    return tuple(stride * (1 << bit) for bit in range(int(layout.seq_len).bit_length() - 1))


def delta_causal_mask(
    layout: AttentionPairLayout,
    *,
    token_valid_mask: np.ndarray | None = None,
    dtype=np.float64,
) -> np.ndarray:
    """Public causal/padding mask in ``S_Delta/P_Delta`` slot order.

    Invalid query rows retain only their self slot.  Those rows are not model
    outputs, but a single public-safe entry keeps the inverse-square-root
    polynomial away from the undefined all-zero row while preventing padding
    keys from influencing any valid query.
    """

    seq_len = int(layout.seq_len)
    queries = np.arange(seq_len, dtype=np.int64)
    deltas = np.arange(seq_len, dtype=np.int64)
    keys = (queries[None, :] + deltas[:, None]) % seq_len
    valid = keys <= queries[None, :]
    if token_valid_mask is not None:
        token_valid = np.asarray(token_valid_mask, dtype=bool).reshape(-1)
        if token_valid.shape != (seq_len,):
            raise ValueError(
                "token_valid_mask must have shape "
                f"{(seq_len,)}, got {token_valid.shape}."
            )
        valid &= token_valid[keys]
        valid &= token_valid[None, :]
        invalid_queries = ~token_valid
        valid[0, invalid_queries] = True
    base = np.repeat(valid[:, :, None], 2, axis=2).reshape(-1).astype(dtype)
    return np.tile(base, int(layout.payload_repetitions))


def _pair_slice_mask(layout: AttentionPairLayout, pair_index: int) -> np.ndarray:
    pair_index = int(pair_index)
    if pair_index < 0 or pair_index >= int(layout.query_pair_count):
        raise ValueError(f"pair_index={pair_index} outside the query-pair range.")
    if pair_index >= int(layout.seq_len):
        raise ValueError("slim Delta packing requires query_pair_count <= seq_len.")
    base = np.zeros((int(layout.seq_len), int(layout.seq_len), 2), dtype=np.float64)
    base[pair_index, :, :] = 1.0
    return np.tile(base.reshape(-1), int(layout.payload_repetitions))


def build_delta_softmax_mask_bundle(
    layout: AttentionPairLayout,
    *,
    crypto_context,
    coefficient: float = 1.0,
    name_prefix: str = "softmax.slim",
):
    vectors: dict[str, np.ndarray] = {}
    for pair_index in range(int(layout.query_pair_count)):
        values = float(coefficient) * _pair_slice_mask(layout, pair_index)
        vectors[f"{name_prefix}.pair.{pair_index}"] = values.reshape(-1)
    return crypto_context.constant_bundle(
        vectors,
        cache_key=(
            "softmax.pair_masks",
            str(name_prefix),
            float(coefficient),
        ),
    )


def _single_vector_bundle(
    values: np.ndarray,
    *,
    name: str,
    crypto_context,
):
    return crypto_context.constant_bundle(
        {str(name): np.asarray(values, dtype=np.float64).reshape(-1)},
        cache_key=("softmax.vector", str(name)),
    )


def delta_slim_active_mask(layout: AttentionPairLayout, *, dtype=np.float64) -> np.ndarray:
    base = np.zeros((int(layout.seq_len), int(layout.seq_len), 2), dtype=dtype)
    base[: int(layout.query_pair_count), :, :] = 1.0
    return np.tile(base.reshape(-1), int(layout.payload_repetitions))


def _slim_fit_interval(poly_entry: dict) -> tuple[float, float]:
    """The Chebyshev fit interval of a slim rsqrt/exp polynomial entry."""

    cheb = poly_entry.get("chebyshev_fhe") or poly_entry
    lo, hi = cheb["fit_interval"]
    return float(lo), float(hi)


def _slim_safe_input(lo: float, hi: float) -> float:
    """A public value inside the fit interval for the inactive slim rows.

    Those rows carry no query, so any in-domain constant works; 0.25 is
    clamped into ``[lo, hi]`` rather than using an endpoint so the polynomial
    is never evaluated at the edge of its fit.
    """

    return min(max(0.25, lo), hi)


def _add_public_fill(cipher, values: np.ndarray, *, name: str, crypto_context):
    """Add a public vector to ``cipher`` at its own level; ``cipher`` is kept."""

    bundle = _single_vector_bundle(
        values, name=name, crypto_context=crypto_context
    )
    plain = _bundle_plain(bundle, name, cipher, crypto_context=crypto_context)
    try:
        return crypto_context.fhe.homo_add_pt(
            cipher, plain, crypto_context.context
        )
    finally:
        release_if_supported(plain)


def _sanitize_delta_slim_poly_input(
    slim_cipher,
    poly_entry: dict,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    name: str,
):
    """Remask the active slim rows, then fill the inactive ones.

    The general form, used when neither the direct fill nor the fused
    Chebyshev map is selected: it costs one plaintext multiplication because it
    does not assume the inactive rows are already zero.
    """

    active_mask = delta_slim_active_mask(layout)
    bundle = _single_vector_bundle(
        active_mask, name=f"{name}.active", crypto_context=crypto_context
    )
    plain = _bundle_plain(
        bundle, f"{name}.active", slim_cipher, crypto_context=crypto_context
    )
    try:
        active = _mul_plain_rescale(
            slim_cipher, plain, crypto_context=crypto_context
        )
    finally:
        release_if_supported(plain)
    lo, hi = _slim_fit_interval(poly_entry)
    try:
        return _add_public_fill(
            active,
            (1.0 - active_mask) * _slim_safe_input(lo, hi),
            name=f"{name}.inactive",
            crypto_context=crypto_context,
        )
    finally:
        release_if_supported(active)


def _fill_delta_slim_poly_input(
    slim_cipher,
    poly_entry: dict,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    name: str,
):
    """Fill already-zero inactive slim rows without another PT multiplication.

    ``delta_pack_row_sums_slim_fhe`` selects one disjoint destination row per
    query pair and leaves every inactive row zero.  Adding the public safe
    polynomial input there is therefore equivalent to remasking the active
    rows first, but consumes no multiplicative level.
    """

    lo, hi = _slim_fit_interval(poly_entry)
    inactive = 1.0 - delta_slim_active_mask(layout)
    return _add_public_fill(
        slim_cipher,
        inactive * _slim_safe_input(lo, hi),
        name=f"{name}.inactive",
        crypto_context=crypto_context,
    )


def _fill_mapped_delta_slim_poly_input(
    slim_cipher,
    poly_entry: dict,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    name: str,
):
    """Finish a Chebyshev-domain map already scaled in slim-pack masks.

    The slim-pack mask supplied the alpha factor, so only the beta offset
    remains for the active rows; the inactive rows get the mapped safe value.
    """

    active_mask = delta_slim_active_mask(layout)
    lo, hi = _slim_fit_interval(poly_entry)
    alpha = 2.0 / (hi - lo)
    beta = -1.0 - 2.0 * lo / (hi - lo)
    mapped_safe = alpha * _slim_safe_input(lo, hi) + beta
    return _add_public_fill(
        slim_cipher,
        active_mask * beta + (1.0 - active_mask) * mapped_safe,
        name=f"{name}.mapped_fill",
        crypto_context=crypto_context,
    )


def _apply_vector_mul_rescale(
    ciphers: tuple[object, ...],
    values: np.ndarray,
    *,
    name: str,
    crypto_context,
) -> tuple[object, ...]:
    bundle = _single_vector_bundle(
        values,
        name=name,
        crypto_context=crypto_context,
    )
    outputs: list[object] = []
    try:
        for cipher in ciphers:
            plain = _bundle_plain(
                bundle,
                name,
                cipher,
                crypto_context=crypto_context,
            )
            try:
                outputs.append(
                    _mul_plain_rescale(cipher, plain, crypto_context=crypto_context)
                )
            finally:
                release_if_supported(plain)
        return tuple(outputs)
    except Exception:
        for cipher in outputs:
            release_if_supported(cipher)
        raise


def _add_vector_chunks(
    ciphers: tuple[object, ...],
    values: np.ndarray,
    *,
    name: str,
    crypto_context,
    max_abs_chunk: float = 1.0,
) -> tuple[object, ...]:
    values = np.asarray(values, dtype=np.float64)
    chunk_count = max(1, int(math.ceil(float(np.max(np.abs(values))) / float(max_abs_chunk))))
    chunk = values / float(chunk_count)
    outputs = tuple(cipher.deep_copy() for cipher in ciphers)
    try:
        for index in range(chunk_count):
            bundle = _single_vector_bundle(
                chunk,
                name=f"{name}.{index}",
                crypto_context=crypto_context,
                )
            next_outputs: list[object] = []
            completed = False
            try:
                for cipher in outputs:
                    plain = _bundle_plain(
                        bundle,
                        f"{name}.{index}",
                        cipher,
                        crypto_context=crypto_context,
                    )
                    try:
                        next_outputs.append(
                            crypto_context.fhe.homo_add_pt(
                                cipher, plain, crypto_context.context
                            )
                        )
                    finally:
                        release_if_supported(plain)
                completed = True
            except Exception:
                for cipher in next_outputs:
                    release_if_supported(cipher)
                raise
            finally:
                for cipher in outputs:
                    release_if_supported(cipher)
                outputs = ()
            if completed:
                outputs = tuple(next_outputs)
        return outputs
    except Exception:
        for cipher in outputs:
            release_if_supported(cipher)
        raise


def delta_reduce_rows_fhe(
    value_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Sum all 128 Delta entries and broadcast each row sum over Delta."""

    if len(value_ciphers) != int(layout.query_pair_count):
        raise ValueError(
            f"Delta row reduction expects {layout.query_pair_count} ciphers, got {len(value_ciphers)}."
        )
    outputs: list[object] = []
    try:
        for cipher in value_ciphers:
            current = cipher.deep_copy()
            for rotation in delta_softmax_row_rotations(layout):
                rotated = crypto_context.fhe.homo_rotate(
                    current, int(rotation), crypto_context.context
                )
                try:
                    next_current = crypto_context.fhe.homo_add(
                        current, rotated, crypto_context.context
                    )
                finally:
                    release_if_supported(current)
                    release_if_supported(rotated)
                current = next_current
            outputs.append(current)
        return tuple(outputs)
    except Exception:
        for cipher in outputs:
            release_if_supported(cipher)
        raise


def delta_pack_row_sums_slim_fhe(
    reduced_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    masks=None,
    mask_name_prefix: str = "softmax.slim",
):
    """Pack every pair's row sums into a distinct Delta slice of one cipher."""

    if len(reduced_ciphers) != int(layout.query_pair_count):
        raise ValueError(
            f"slim pack expects {layout.query_pair_count} ciphers, got {len(reduced_ciphers)}."
        )
    bundle = (
        build_delta_softmax_mask_bundle(
            layout,
            crypto_context=crypto_context,
        )
        if masks is None
        else masks
    )
    accumulator = None
    try:
        for pair_index, cipher in enumerate(reduced_ciphers):
            plain = _bundle_plain(
                bundle,
                f"{mask_name_prefix}.pair.{pair_index}",
                cipher,
                crypto_context=crypto_context,
            )
            try:
                selected = _mul_plain_rescale(cipher, plain, crypto_context=crypto_context)
            finally:
                release_if_supported(plain)
            accumulator = (
                selected
                if accumulator is None
                else _add_owned(accumulator, selected, crypto_context=crypto_context)
            )
        if accumulator is None:
            raise RuntimeError("slim row-sum pack produced no output.")
        return accumulator
    except Exception:
        if accumulator is not None:
            release_if_supported(accumulator)
        raise


def delta_broadcast_slim_rows_fhe(
    slim_cipher,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    masks=None,
) -> tuple[object, ...]:
    """Extract each slim pair slice and broadcast it over all Delta blocks."""

    bundle = (
        build_delta_softmax_mask_bundle(layout, crypto_context=crypto_context)
        if masks is None
        else masks
    )
    stride = 2 * int(layout.seq_len)
    steps = delta_softmax_row_rotations(layout)
    outputs: list[object] = []
    source = slim_cipher
    expanded = None
    try:
        if int(slim_cipher.slots) != int(crypto_context.max_slots):
            raise ValueError(
                "slim broadcast requires the full Delta slot count."
            )
        for pair_index in range(int(layout.query_pair_count)):
            plain = _bundle_plain(
                bundle,
                f"softmax.slim.pair.{pair_index}",
                source,
                crypto_context=crypto_context,
            )
            try:
                selected = _mul_plain_rescale(source, plain, crypto_context=crypto_context)
            finally:
                release_if_supported(plain)
            if pair_index:
                current = crypto_context.fhe.homo_rotate(
                    selected, pair_index * stride, crypto_context.context
                )
                release_if_supported(selected)
            else:
                current = selected
            for positive_rotation in steps:
                rotated = crypto_context.fhe.homo_rotate(
                    current, -int(positive_rotation), crypto_context.context
                )
                try:
                    next_current = crypto_context.fhe.homo_add(
                        current, rotated, crypto_context.context
                    )
                finally:
                    release_if_supported(current)
                    release_if_supported(rotated)
                current = next_current
            outputs.append(current)
        return tuple(outputs)
    except Exception:
        for cipher in outputs:
            release_if_supported(cipher)
        raise
    finally:
        release_if_supported(expanded)


def _half_rescale(cipher, *, crypto_context):
    """Undo the 2x from ``split_complex_ciphers_twice`` after a paired bootstrap.

    The pre-exp checkpoint folds that 0.5 into the public score plaintext
    instead. Extra checkpoints after exp / Alg2 must apply it here; omitting
    it maps the probability stream to ~1e20.
    """

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


# Extra main-track Softmax bootstraps for the slim (depth-37 / post-19)
# context. Mid-layer k=2 fits in 19 limbs without them. High-k layers
# (0,1: k=4; 2,14: k=3; 31: k=5) would otherwise exhaust PairFM's 6-limb
# floor. Layer 31 also refreshes after exp, before round-1 λ: post-Alg2 is
# too late because λ is degree 512 on the slim program.
SLIM_POST_ALG2_LAYERS = frozenset({0, 1, 2, 14})
SLIM_POST_EXP_LAYERS = frozenset({31})
SLIM_MID_ALG2_LAYERS = frozenset({31})
_LEGACY_BOOTSTRAP_POST_LEVELS = 23


def _parse_layer_enable_env(raw: str, layer_idx: int) -> bool:
    text = raw.strip().lower()
    if text in {"", "false", "off", "none"}:
        return False
    if text in {"all", "true"}:
        return True
    return int(layer_idx) in {
        int(part) for part in text.split(",") if part.strip() != ""
    }


def extra_softmax_checkpoint_enabled(
    name: str,
    layer_idx: int,
    default_layers: frozenset[int],
    bootstrap_operator: BootstrapOperator | None,
) -> bool:
    """Return whether an extra Softmax checkpoint should run on ``layer_idx``.

    An explicit env var always wins (comma layer list, ``all``/``true``, or
    ``off``). ``1`` is layer 1, not a boolean. Unset env uses the slim
    schedule only when the bootstrap returns fewer than the legacy 23 levels.
    """

    raw = os.environ.get(name)
    if raw is not None:
        return _parse_layer_enable_env(raw, layer_idx)
    if bootstrap_operator is None:
        return False
    if int(bootstrap_operator.config.output_levels) >= _LEGACY_BOOTSTRAP_POST_LEVELS:
        return False
    return int(layer_idx) in default_layers


def _paired_checkpoint_refresh(
    y_ciphers: tuple[object, ...],
    *,
    bootstrap_operator: BootstrapOperator,
    label_prefix: str,
    main_bootstrap_iterations: int | None,
    checkpoint_precision_bits: int,
    match_single_pass_bootstrap_level: bool,
    crypto_context,
) -> tuple[object, ...]:
    """Pair, bootstrap, split, and undo the 2x conjugate scale."""

    refreshed_y: list[object] = []
    paired_y: tuple[object, ...] = ()
    try:
        paired_y = pair_real_ciphers(y_ciphers, crypto_context=crypto_context)
        for index, cipher in enumerate(paired_y):
            refreshed_y.append(
                bootstrap_operator.refresh(
                    cipher,
                    iterations=main_bootstrap_iterations,
                    precision_bits=int(checkpoint_precision_bits),
                    match_two_pass_level=bool(
                        match_single_pass_bootstrap_level
                    ),
                )
            )
    except Exception:
        for cipher in refreshed_y:
            release_if_supported(cipher)
        raise
    finally:
        for cipher in paired_y:
            release_if_supported(cipher)
    for cipher in y_ciphers:
        release_if_supported(cipher)
    split_y: tuple[object, ...] = ()
    halved_y: list[object] = []
    try:
        split_y = split_complex_ciphers_twice(
            tuple(refreshed_y), crypto_context=crypto_context
        )
        for cipher in split_y:
            halved_y.append(
                _half_rescale(cipher, crypto_context=crypto_context)
            )
    except Exception:
        for cipher in halved_y:
            release_if_supported(cipher)
        raise
    finally:
        for cipher in refreshed_y:
            release_if_supported(cipher)
        for cipher in split_y:
            release_if_supported(cipher)
    return tuple(halved_y)


def softmax_alg2_fhe_delta(
    score_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    layer_idx: int,
    config,
    crypto_context,
    operator_config: AttentionOperatorConfig | None = None,
    score_scale: float | None = None,
    bootstrap_operator: BootstrapOperator | None = None,
    refresh_before_exp: bool | None = None,
    refresh_after_exp: bool | None = None,
    token_valid_mask: np.ndarray | None = None,
) -> DeltaSoftmaxAlg2Result:
    """Run polynomial Alg2 directly on paired ``S_Delta`` ciphertexts.

    ``config`` is the fitted softmax artifact; ``operator_config`` is the
    execution policy. Lambda refresh is owned by this Softmax operator:
    supplying a prepared :class:`BootstrapOperator` makes the required Alg2
    bootstraps explicit without leaking callback scheduling into the model.

    The two refresh flags stay parameters because the caller resolves them per
    layer -- layer zero has no steady-state pre-exp checkpoint -- and pass
    ``None`` to accept the policy's own value.
    """

    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    if refresh_before_exp is None:
        refresh_before_exp = operator_config.refresh_softmax_exp_inputs
    if refresh_after_exp is None:
        refresh_after_exp = operator_config.refresh_softmax_exp_outputs
    main_bootstrap_use_deep = bool(operator_config.softmax_main_deep_bootstrap)
    pair_pre_exp_bootstrap = bool(
        operator_config.pair_softmax_pre_exp_bootstrap
    )
    reuse_round_square = bool(operator_config.reuse_softmax_round_square)
    direct_slim_inactive_fill = bool(
        operator_config.direct_softmax_slim_inactive_fill
    )
    fuse_slim_chebyshev_map = bool(
        operator_config.fuse_softmax_slim_chebyshev_map
    )
    main_bootstrap_iterations = operator_config.softmax_main_bootstrap_iterations
    lambda_bootstrap_iterations = (
        operator_config.softmax_lambda_bootstrap_iterations
    )
    match_single_pass_bootstrap_level = bool(
        operator_config.match_softmax_single_pass_bootstrap_level
    )
    checkpoint_precision_bits = int(
        operator_config.checkpoint_bootstrap_precision_bits
    )

    from ..nonlinear.elementwise import (
        eval_exp_cipher,
        eval_invsqrt_cipher,
    )

    if len(score_ciphers) != int(layout.query_pair_count):
        raise ValueError(
            f"Delta Alg2 expects {layout.query_pair_count} score ciphers, got {len(score_ciphers)}."
        )
    layer_idx = int(layer_idx)
    k_layer = int(config.k_for(layer_idx))
    score_scale = (
        1.0 / math.sqrt(float(layout.head_dim))
        if score_scale is None
        else float(score_scale)
    )
    valid = delta_causal_mask(
        layout, token_valid_mask=token_valid_mask
    )
    pair_pre_exp_bootstrap = bool(pair_pre_exp_bootstrap)
    reuse_round_square = bool(reuse_round_square)
    direct_slim_inactive_fill = bool(direct_slim_inactive_fill)
    fuse_slim_chebyshev_map = bool(fuse_slim_chebyshev_map)
    if pair_pre_exp_bootstrap and not bool(refresh_before_exp):
        raise ValueError(
            "pair_pre_exp_bootstrap requires refresh_before_exp=True."
        )
    masks = build_delta_softmax_mask_bundle(layout, crypto_context=crypto_context)
    slim_masks = masks
    stage_seconds: dict[str, float] = {}
    fine_profile = os.environ.get("LLAMA_FHE_FINE_PROFILE", "") == "1"

    def fine_mark(name: str, started: float) -> float:
        """Close one opt-in, synchronization-delimited profile interval."""

        sync_device(crypto_context.device)
        finished = time.perf_counter()
        stage_seconds[str(name)] = float(finished - started)
        return finished

    y_ciphers: tuple[object, ...] = ()
    success = False
    try:
        start = time.perf_counter()
        exp_entry = config.layer_exp[layer_idx]
        score_multiplier = score_scale / float(2**k_layer)
        public_shift = -float(config.U_hat_for(layer_idx)) / float(2**k_layer)
        invalid_value = float(config.stats_min_for(layer_idx)) / float(2**k_layer)
        # In the paired checkpoint path each complex split returns
        # ``2*Re, 2*Im``.  Pre-map and halve the two real streams inside the
        # score plaintext that already consumes this level.  The split then
        # recovers the exact Chebyshev-domain input, so both the extra 0.5
        # rescale and the polynomial domain-map level disappear.
        if pair_pre_exp_bootstrap:
            fit_lo, fit_hi = (float(value) for value in exp_entry["fit_interval"])
            map_alpha = 2.0 / (fit_hi - fit_lo)
            map_beta = -1.0 - 2.0 * fit_lo / (fit_hi - fit_lo)
            score_multiplier *= 0.5 * map_alpha
            public_shift = 0.5 * (map_alpha * public_shift + map_beta)
            invalid_value *= 0.5 * map_alpha
        y_ciphers = _apply_vector_mul_rescale(
            score_ciphers,
            valid * score_multiplier,
            name="softmax.delta.valid_scale",
            crypto_context=crypto_context,
        )
        shifted: list[object] = []
        for cipher in y_ciphers:
            encoded = crypto_context.fhe.encode_scalar(
                public_shift,
                cur_limbs=int(cipher.state.cur_limbs),
                scale_degree=int(cipher.state.scale_degree),
                scaling_factor=cipher.state.scaling_factor,
                context=crypto_context.context,
            )
            shifted.append(
                crypto_context.fhe.homo_add_scalar(
                    cipher, encoded, crypto_context.context
                )
            )
        for cipher in y_ciphers:
            release_if_supported(cipher)
        y_ciphers = tuple(shifted)
        invalid_correction = (
            (1.0 - valid)
            * invalid_value
        )
        corrected = _add_vector_chunks(
            y_ciphers,
            invalid_correction,
            name="softmax.delta.invalid",
            crypto_context=crypto_context,
        )
        for cipher in y_ciphers:
            release_if_supported(cipher)
        y_ciphers = corrected
        _softmax_diagnostic_range(
            "shift_scale_mask", y_ciphers, crypto_context=crypto_context
        )
        stage_seconds["shift_scale_mask"] = time.perf_counter() - start

        if bool(refresh_before_exp) and bool(refresh_after_exp):
            raise ValueError(
                "Softmax main stream cannot be refreshed both before and after exp."
            )
        if bool(refresh_before_exp):
            if bootstrap_operator is None:
                raise ValueError(
                    "refresh_before_exp requires a prepared bootstrap operator."
                )
            start = time.perf_counter()
            refreshed_y: list[object] = []
            paired_y: tuple[object, ...] = ()
            try:
                checkpoint_inputs = y_ciphers
                if pair_pre_exp_bootstrap:
                    paired_y = pair_real_ciphers(
                        y_ciphers, crypto_context=crypto_context
                    )
                    checkpoint_inputs = paired_y
                for index, cipher in enumerate(checkpoint_inputs):
                    refreshed_y.append(
                        bootstrap_operator.refresh(
                            cipher,
                            iterations=main_bootstrap_iterations,
                            precision_bits=int(checkpoint_precision_bits),
                            match_two_pass_level=bool(
                                match_single_pass_bootstrap_level
                            ),
                            use_deep_program=bool(main_bootstrap_use_deep),
                        )
                    )
            except Exception:
                for cipher in refreshed_y:
                    release_if_supported(cipher)
                raise
            finally:
                for cipher in paired_y:
                    release_if_supported(cipher)
            for cipher in y_ciphers:
                release_if_supported(cipher)
            if pair_pre_exp_bootstrap:
                y_ciphers = ()
                try:
                    y_ciphers = split_complex_ciphers_twice(
                        tuple(refreshed_y), crypto_context=crypto_context
                    )
                finally:
                    for cipher in refreshed_y:
                        release_if_supported(cipher)
            else:
                y_ciphers = tuple(refreshed_y)
            _softmax_diagnostic_range(
                "pre_exp_bootstrap", y_ciphers, crypto_context=crypto_context
            )
            stage_seconds["pre_exp_bootstrap"] = (
                time.perf_counter() - start
            )

        start = time.perf_counter()
        exp_outputs: list[object] = []
        for cipher in y_ciphers:
            exp_outputs.append(
                eval_exp_cipher(
                    cipher,
                    exp_entry,
                    crypto_context=crypto_context,
                    input_is_chebyshev_mapped=pair_pre_exp_bootstrap,
                )
            )
        # A normal polynomial evaluation consumes its input while mapping the
        # domain.  The paired path has already performed that map publicly,
        # so release the still-owned split inputs explicitly.
        if pair_pre_exp_bootstrap:
            for cipher in y_ciphers:
                release_if_supported(cipher)
        y_ciphers = tuple(exp_outputs)
        masked_exp = _apply_vector_mul_rescale(
            y_ciphers,
            valid,
            name="softmax.delta.post_exp_valid",
            crypto_context=crypto_context,
        )
        for cipher in y_ciphers:
            release_if_supported(cipher)
        y_ciphers = masked_exp
        _softmax_diagnostic_range(
            "exp", y_ciphers, crypto_context=crypto_context
        )
        stage_seconds["exp"] = time.perf_counter() - start

        if extra_softmax_checkpoint_enabled(
            "LLAMA_FHE_SOFTMAX_POST_EXP_BOOTSTRAP",
            layer_idx,
            SLIM_POST_EXP_LAYERS,
            bootstrap_operator,
        ):
            if bootstrap_operator is None:
                raise ValueError(
                    "post-exp bootstrap requires a prepared bootstrap operator."
                )
            start = time.perf_counter()
            y_ciphers = _paired_checkpoint_refresh(
                y_ciphers,
                bootstrap_operator=bootstrap_operator,
                label_prefix="alg2.post_exp",
                main_bootstrap_iterations=main_bootstrap_iterations,
                checkpoint_precision_bits=int(checkpoint_precision_bits),
                match_single_pass_bootstrap_level=bool(
                    match_single_pass_bootstrap_level
                ),
                crypto_context=crypto_context,
            )
            _softmax_diagnostic_range(
                "post_exp_bootstrap",
                y_ciphers,
                crypto_context=crypto_context,
            )
            stage_seconds["post_exp_bootstrap"] = (
                time.perf_counter() - start
            )

        if bool(refresh_after_exp):
            if bootstrap_operator is None:
                raise ValueError(
                    "refresh_after_exp requires a prepared bootstrap operator."
                )
            start = time.perf_counter()
            refreshed_y: list[object] = []
            try:
                for index, cipher in enumerate(y_ciphers):
                    refreshed_y.append(
                        bootstrap_operator.refresh(
                            cipher,
                            iterations=main_bootstrap_iterations,
                            precision_bits=int(checkpoint_precision_bits),
                            match_two_pass_level=bool(
                                match_single_pass_bootstrap_level
                            ),
                        )
                    )
            except Exception:
                for cipher in refreshed_y:
                    release_if_supported(cipher)
                raise
            for cipher in y_ciphers:
                release_if_supported(cipher)
            y_ciphers = tuple(refreshed_y)
            stage_seconds["post_exp_bootstrap"] = (
                time.perf_counter() - start
            )

        rough = config.invsqrt_alg2["j1_rough"]
        precise = config.invsqrt_alg2["j2_precise"]
        for round_index in range(1, k_layer + 1):
            if fine_profile:
                sync_device(crypto_context.device)
            round_start = time.perf_counter()
            fine_started = round_start
            squared_for_sum = tuple(
                _square_rescale(cipher, crypto_context=crypto_context)
                for cipher in y_ciphers
            )
            reduced = delta_reduce_rows_fhe(
                squared_for_sum,
                layout=layout,
                crypto_context=crypto_context,
            )
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.square_reduce",
                    fine_started,
                )
            _softmax_diagnostic_range(
                f"round_{round_index}.row_sum",
                reduced,
                crypto_context=crypto_context,
            )
            if reuse_round_square:
                previous_y = y_ciphers
                y_ciphers = ()
                for cipher in previous_y:
                    release_if_supported(cipher)
            else:
                for cipher in squared_for_sum:
                    release_if_supported(cipher)
            poly = config.poly_for_layer(
                rough if round_index == 1 else precise,
                layer_idx,
            )
            mapped_mask_prefix = f"softmax.slim.mapped.{round_index}"
            mapped_masks = None
            if fuse_slim_chebyshev_map:
                cheb = poly.get("chebyshev_fhe") or poly
                lo, hi = (float(value) for value in cheb["fit_interval"])
                mapped_masks = build_delta_softmax_mask_bundle(
                    layout,
                    crypto_context=crypto_context,
                    coefficient=2.0 / (hi - lo),
                    name_prefix=mapped_mask_prefix,
                )
            slim = delta_pack_row_sums_slim_fhe(
                reduced,
                layout=layout,
                crypto_context=crypto_context,
                masks=(
                    mapped_masks if mapped_masks is not None else slim_masks
                ),
                mask_name_prefix=(
                    mapped_mask_prefix
                    if mapped_masks is not None
                    else "softmax.slim"
                ),
            )
            for cipher in reduced:
                release_if_supported(cipher)

            sanitize_slim = (
                _fill_mapped_delta_slim_poly_input
                if fuse_slim_chebyshev_map
                else _fill_delta_slim_poly_input
                if direct_slim_inactive_fill
                else _sanitize_delta_slim_poly_input
            )
            sanitized_slim = sanitize_slim(
                slim,
                poly,
                layout=layout,
                crypto_context=crypto_context,
                name=f"softmax.delta.j{round_index}.slim",
            )
            release_if_supported(slim)
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.slim_pack",
                    fine_started,
                )
            _softmax_diagnostic_range(
                f"round_{round_index}.slim_input",
                (sanitized_slim,),
                crypto_context=crypto_context,
            )
            # A streamed layer can reach this compact boundary with too few
            # limbs for the deep inverse-square-root polynomial.  Refresh its
            # input only when needed; the normal Layer0 path retains the
            # original post-polynomial refresh schedule.
            polynomial_depth = chebyshev_ps_mul_depth(poly["coefficients"])
            total_polynomial_depth = polynomial_depth + (
                0 if fuse_slim_chebyshev_map else 1
            )
            if (
                bootstrap_operator is not None
                and int(sanitized_slim.state.cur_limbs)
                <= int(total_polynomial_depth) + 1
            ):
                previous = sanitized_slim
                sanitized_slim = bootstrap_operator.refresh(
                    previous,
                    iterations=lambda_bootstrap_iterations,
                    match_two_pass_level=bool(
                        match_single_pass_bootstrap_level
                    ),
                )
                if sanitized_slim is not previous:
                    release_if_supported(previous)
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.slim_input_refresh",
                    fine_started,
                )
            lambda_cipher = eval_invsqrt_cipher(
                sanitized_slim,
                poly,
                crypto_context=crypto_context,
                input_is_chebyshev_mapped=fuse_slim_chebyshev_map,
            )
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.lambda_polynomial",
                    fine_started,
                )
            _softmax_diagnostic_range(
                f"round_{round_index}.lambda_poly",
                (lambda_cipher,),
                crypto_context=crypto_context,
            )
            if bootstrap_operator is not None:
                previous = lambda_cipher
                lambda_cipher = bootstrap_operator.refresh(
                    previous,
                    iterations=lambda_bootstrap_iterations,
                    match_two_pass_level=bool(
                        match_single_pass_bootstrap_level
                    ),
                )
                if lambda_cipher is not previous:
                    release_if_supported(previous)
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.lambda_refresh",
                    fine_started,
                )
            _softmax_diagnostic_range(
                f"round_{round_index}.lambda_refreshed",
                (lambda_cipher,),
                crypto_context=crypto_context,
            )
            lambda_for_broadcast = lambda_cipher
            if reuse_round_square:
                lambda_for_broadcast = _square_rescale(
                    lambda_cipher, crypto_context=crypto_context
                )
                release_if_supported(lambda_cipher)
            lambda_pairs = delta_broadcast_slim_rows_fhe(
                lambda_for_broadcast,
                layout=layout,
                crypto_context=crypto_context,
                masks=masks,
            )
            release_if_supported(lambda_for_broadcast)
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.lambda_square_broadcast",
                    fine_started,
                )

            if reuse_round_square:
                squared = tuple(
                    _mul_cipher_rescale(
                        left, right, crypto_context=crypto_context
                    )
                    for left, right in zip(
                        squared_for_sum, lambda_pairs, strict=True
                    )
                )
                for cipher in squared_for_sum + lambda_pairs:
                    release_if_supported(cipher)
            else:
                scaled = tuple(
                    _mul_cipher_rescale(
                        left, right, crypto_context=crypto_context
                    )
                    for left, right in zip(
                        y_ciphers, lambda_pairs, strict=True
                    )
                )
                for cipher in y_ciphers + lambda_pairs:
                    release_if_supported(cipher)
                squared = tuple(
                    _square_rescale(cipher, crypto_context=crypto_context)
                    for cipher in scaled
                )
                for cipher in scaled:
                    release_if_supported(cipher)
            # ``y_ciphers`` was causally masked immediately after exp.
            # Multiplication by a row scalar and squaring cannot turn an
            # exact invalid zero into model data, so keep the squared stream
            # directly instead of spending one PT-CT level per round on the
            # same public mask.  The post-exp mask remains the sole explicit
            # causal-zero enforcement before the four Alg2 rounds.
            y_ciphers = squared
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.main_update",
                    fine_started,
                )
            _softmax_diagnostic_range(
                f"round_{round_index}.output",
                y_ciphers,
                crypto_context=crypto_context,
            )
            stage_seconds[f"alg2_round_{round_index}"] = (
                time.perf_counter() - round_start
            )
            mid_round = (int(k_layer) + 1) // 2
            if (
                round_index == mid_round
                and mid_round < int(k_layer)
                and extra_softmax_checkpoint_enabled(
                    "LLAMA_FHE_SOFTMAX_MID_ALG2_BOOTSTRAP",
                    layer_idx,
                    SLIM_MID_ALG2_LAYERS,
                    bootstrap_operator,
                )
            ):
                if bootstrap_operator is None:
                    raise ValueError(
                        "mid-Alg2 bootstrap requires a prepared bootstrap "
                        "operator."
                    )
                start = time.perf_counter()
                y_ciphers = _paired_checkpoint_refresh(
                    y_ciphers,
                    bootstrap_operator=bootstrap_operator,
                    label_prefix=f"alg2.mid_round_{round_index}",
                    main_bootstrap_iterations=main_bootstrap_iterations,
                    checkpoint_precision_bits=int(checkpoint_precision_bits),
                    match_single_pass_bootstrap_level=bool(
                        match_single_pass_bootstrap_level
                    ),
                    crypto_context=crypto_context,
                )
                _softmax_diagnostic_range(
                    f"mid_alg2_bootstrap_after_round_{round_index}",
                    y_ciphers,
                    crypto_context=crypto_context,
                )
                stage_seconds["mid_alg2_bootstrap"] = (
                    time.perf_counter() - start
                )


        if extra_softmax_checkpoint_enabled(
            "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP",
            layer_idx,
            SLIM_POST_ALG2_LAYERS,
            bootstrap_operator,
        ):
            if bootstrap_operator is None:
                raise ValueError(
                    "post-Alg2 bootstrap requires a prepared bootstrap operator."
                )
            _softmax_diagnostic_range(
                "pre_post_alg2_bootstrap",
                y_ciphers,
                crypto_context=crypto_context,
            )
            start = time.perf_counter()
            y_ciphers = _paired_checkpoint_refresh(
                y_ciphers,
                bootstrap_operator=bootstrap_operator,
                label_prefix="alg2.post_alg2",
                main_bootstrap_iterations=main_bootstrap_iterations,
                checkpoint_precision_bits=int(checkpoint_precision_bits),
                match_single_pass_bootstrap_level=bool(
                    match_single_pass_bootstrap_level
                ),
                crypto_context=crypto_context,
            )
            _softmax_diagnostic_range(
                "post_alg2_bootstrap",
                y_ciphers,
                crypto_context=crypto_context,
            )
            stage_seconds["post_alg2_bootstrap"] = (
                time.perf_counter() - start
            )

        success = True
        return DeltaSoftmaxAlg2Result(
            probability_ciphers=y_ciphers,
            stage_seconds={name: float(value) for name, value in stage_seconds.items()},
            output_levels=tuple(
                sorted({int(crypto_context.level_for_cipher(cipher)) for cipher in y_ciphers})
            ),
        )
    finally:
        if not success:
            for cipher in y_ciphers:
                release_if_supported(cipher)
