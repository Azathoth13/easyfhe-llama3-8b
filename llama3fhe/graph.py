from __future__ import annotations

"""One persistent-feature-major encrypted transformer-layer graph.

The long-lived tensor remains real feature-major across both residual paths::

    FM -> RMSNorm -> FM QKV -> Q_C/K_Delta/V_Delta
       -> Alg2 Softmax/PV -> O_C -> PairFM -> FM W_O -> residual
       -> paired Meta-BTS -> FM RMSNorm -> FM SwiGLU -> residual -> FM

PairFM is private to the attention output epilogue. The complete graph uses a
ciphertext-only Alg2/Meta-BTS schedule and never hides decrypt/re-encrypt
refreshes behind a smaller nominal depth.
"""

import os
import time
from dataclasses import dataclass

import easyfhe
import numpy as np

from .approx.chebyshev import chebyshev_ps_mul_depth
from .backend import (
    cuda_memory_snapshot,
    release_all,
    trim_device_allocator_cache,
)
from .config import Llama3CKKSConfig
from .layouts.attention import AttentionPairLayout
from .layouts.feature_major import FeatureMajorPrefillLayout
from .layouts.linear import LinearCarrierLayout
from .operators.attention.audit import (
    delta_pv_operation_counts,
    delta_qk_operation_counts,
    delta_softmax_alg2_complexity,
    pipeline_static_audit,
)
from .operators.attention.config import AttentionOperatorConfig
from .operators.attention.operator import (
    attention_fhe_from_qkv_carriers,
    attention_to_w_o_rotations,
)
from .operators.attention.output import (
    pairfm_output_complexity,
)
from .operators.linear import (
    LinearOperatorConfig,
    linear_rotations,
)
from .operators.linear.mlp import (
    mlp_complexity,
    mlp_fhe,
    mlp_rotations,
)
from .operators.linear.output import w_o_fhe
from .operators.linear.qkv import qkv_fhe, qkv_rotations
from .operators.nonlinear.bootstrap import BootstrapOperator
from .operators.norm.refresh import (
    bootstrap_rmsnorm_fhe,
    refresh_ciphers,
)
from .operators.norm.rmsnorm import (
    SlimRefresh,
    residual_add,
    rmsnorm_complexity,
    rmsnorm_fhe,
    rmsnorm_rotations,
)
from .operators.primitives import drop_levels
from .schedule import LayerSchedule


@dataclass(frozen=True)
class LayerWeights:
    """One layer's plaintext weights, exactly as the checkpoint stores them."""

    input_layernorm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    post_attention_layernorm: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray


@dataclass(frozen=True)
class LayerApproximations:
    """The polynomial approximations selected for one layer."""

    input_norm: dict
    post_attention_norm: dict
    silu: dict
    softmax: object


@dataclass
class LayerResult:
    ciphers: tuple[object, ...]
    wall_seconds: float
    stage_seconds: dict[str, float]
    nested_stage_seconds: dict[str, object]
    softmax_bootstrap_calls: int
    residual_bootstrap_calls: int
    softmax_native_bootstrap_calls: int
    residual_native_bootstrap_calls: int
    output_levels: tuple[int, ...]
    output: np.ndarray | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None

    def release(self) -> None:
        release_all(self.ciphers)
        self.ciphers = ()


def _cipher_levels(ciphers, *, crypto_context) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(crypto_context.level_for_cipher(cipher))
                for cipher in tuple(ciphers)
            }
        )
    )


def _cipher_runtime_state(cipher, *, crypto_context) -> dict[str, object]:
    """Return the complete metadata relevant to a ciphertext addition."""

    scaling_factor = cipher.state.scaling_factor
    return {
        "level": int(crypto_context.level_for_cipher(cipher)),
        "cur_limbs": int(cipher.state.cur_limbs),
        "scale_degree": int(cipher.state.scale_degree),
        "scaling_factor": (
            None if scaling_factor is None else float(scaling_factor)
        ),
        "slots": int(cipher.slots),
        "is_ext": bool(cipher.is_ext),
        "batch_size": int(getattr(cipher, "batch_size", 1)),
        "component_count": int(len(cipher.cv)),
    }


def _cipher_stream_state_audit(ciphers, *, crypto_context) -> dict[str, object]:
    """Compact identical full-state rows without losing their multiplicity."""

    groups: list[dict[str, object]] = []
    for cipher in tuple(ciphers):
        state = _cipher_runtime_state(cipher, crypto_context=crypto_context)
        for group in groups:
            comparable = {
                key: value
                for key, value in group.items()
                if key != "count"
            }
            if comparable == state:
                group["count"] = int(group["count"]) + 1
                break
        else:
            groups.append({**state, "count": 1})
    return {
        "cipher_count": int(sum(int(group["count"]) for group in groups)),
        "unique_states": groups,
    }


def _residual_state_audit(
    left_ciphers,
    right_ciphers,
    *,
    crypto_context,
) -> dict[str, object]:
    """Validate structural ownership boundaries and report add alignment.

    Different levels/scales are expected here: ``residual_add``
    explicitly aligns the higher-limb operand to the lower-limb operand.  The
    fields that cannot be repaired by that alignment must already match.
    """

    left_ciphers = tuple(left_ciphers)
    right_ciphers = tuple(right_ciphers)
    if len(left_ciphers) != len(right_ciphers):
        raise ValueError(
            "residual state audit requires equal streams, got "
            f"{len(left_ciphers)} and {len(right_ciphers)}."
        )
    alignment_pairs = 0
    for index, (left, right) in enumerate(
        zip(left_ciphers, right_ciphers, strict=True)
    ):
        for field in ("slots", "is_ext"):
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if left_value != right_value:
                raise ValueError(
                    f"residual pair {index}: {field} mismatch: "
                    f"{left_value} != {right_value}."
                )
        alignment_pairs += int(left.state != right.state)
    return {
        "left": _cipher_stream_state_audit(
            left_ciphers, crypto_context=crypto_context
        ),
        "right": _cipher_stream_state_audit(
            right_ciphers, crypto_context=crypto_context
        ),
        "state_alignment_pairs": int(alignment_pairs),
    }


def _select_residual_identity(
    original_ciphers,
    refreshed_ciphers,
    *,
    preserve_prebootstrap: bool,
    refresh_was_run: bool,
) -> tuple[tuple[object, ...], str]:
    """Borrow, rather than copy, the selected residual identity stream."""

    original_ciphers = tuple(original_ciphers)
    refreshed_ciphers = tuple(refreshed_ciphers)
    if bool(preserve_prebootstrap) or not bool(refresh_was_run):
        return original_ciphers, "prebootstrap"
    if len(refreshed_ciphers) != len(original_ciphers):
        raise ValueError(
            "bootstrap-refreshed residual stream has the wrong size: "
            f"{len(refreshed_ciphers)} != {len(original_ciphers)}."
        )
    return refreshed_ciphers, "bootstrap_refreshed"


def _linear_config(
    schedule: LayerSchedule,
    max_plaintext_rows: int,
    hoist_strategy: str,
) -> LinearOperatorConfig:
    """The linear execution policy for one projection boundary."""

    return LinearOperatorConfig(
        baby_steps=int(schedule.linear_baby_steps),
        baby_anchor_step=int(schedule.linear_anchor_step),
        max_plaintext_rows=int(max_plaintext_rows),
        hoist_strategy=str(hoist_strategy),
    )


def required_rotations(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    schedule: LayerSchedule | None = None,
    log_n: int = 16,
    bootstrap_deep_level_budget: tuple[int, int] | None = None,
) -> tuple[int, ...]:
    """Exact rotation-key union for the complete feature-major Layer0.

    ``bootstrap_deep_level_budget`` stays a separate parameter: key
    planning must include the deep budget's CtoS/StoC union when the deep
    main bootstrap is enabled, while a layer's own rotation accounting
    never does.
    """

    schedule = LayerSchedule() if schedule is None else schedule
    # The stages below read the schedule's decisions as plain locals;
    # binding them here keeps the object at the boundary only.
    qkv_hoist_strategy = schedule.qkv_hoist_strategy
    output_hoist_strategy = schedule.output_hoist_strategy
    mlp_hoist_strategy = schedule.mlp_hoist_strategy
    qkv_max_plaintext_rows = schedule.qkv_max_plaintext_rows
    output_max_plaintext_rows = schedule.output_max_plaintext_rows
    mlp_max_plaintext_rows = schedule.mlp_max_plaintext_rows
    sparse_max_baby_offsets = schedule.sparse_max_baby_offsets
    shear_baby_steps = schedule.shear_baby_steps
    bootstrap_log_slots = schedule.bootstrap_log_slots
    bootstrap_level_budget = schedule.bootstrap_level_budget
    rmsnorm2_slim_log_slots = schedule.rmsnorm2_slim_log_slots
    pv_complex_delta_folding = schedule.pv_complex_delta_folding
    pv_prefold_after_softmax = schedule.pv_prefold_after_softmax

    # One config per linear boundary, assembled from the schedule exactly
    # once. The key planner and the kernel must read the same object: a
    # partially rebuilt config plans a different key union and the missing
    # key only surfaces as a crash deep inside a rotation.
    qkv_linear = _linear_config(
        schedule, qkv_max_plaintext_rows, qkv_hoist_strategy
    )
    output_linear = _linear_config(
        schedule, output_max_plaintext_rows, output_hoist_strategy
    )
    mlp_linear = _linear_config(
        schedule, mlp_max_plaintext_rows, mlp_hoist_strategy
    )

    rotations = set(
        qkv_rotations(feature_layout, operator_config=qkv_linear)
    )
    rotations.update(
        attention_to_w_o_rotations(
            LinearCarrierLayout(
                seq_len=int(feature_layout.seq_len),
                dimension=int(feature_layout.hidden_dim),
                slots=int(feature_layout.slots),
            ),
            feature_layout,
            attention_layout,
            operator_config=AttentionOperatorConfig(
                sparse_max_baby_offsets=int(sparse_max_baby_offsets),
                shear_baby_steps=int(shear_baby_steps),
                pv_complex_delta_folding=bool(pv_complex_delta_folding),
                pv_prefold_after_softmax=bool(pv_prefold_after_softmax),
            ),
            log_n=int(log_n),
        )
    )
    rotations.update(
        linear_rotations(
            dimension=int(feature_layout.hidden_dim),
            slots=int(feature_layout.slots),
            token_lanes=int(feature_layout.tokens_per_cipher),
            include_conjugation=True,
            operator_config=output_linear,
        )
    )
    rotations.update(rmsnorm_rotations(feature_layout))
    rotations.update(
        mlp_rotations(feature_layout, operator_config=mlp_linear)
    )
    rotations.update(
        BootstrapOperator.required_rotations(
            log_n=int(log_n),
            log_slots=int(bootstrap_log_slots),
            level_budget=tuple(int(value) for value in bootstrap_level_budget),
            deep_level_budget=(
                None
                if bootstrap_deep_level_budget is None
                else tuple(int(v) for v in bootstrap_deep_level_budget)
            ),
            slim_log_slots=rmsnorm2_slim_log_slots,
        )
    )
    return tuple(sorted(int(rotation) for rotation in rotations if int(rotation)))


def operation_audit(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    schedule: LayerSchedule | None = None,
    intermediate_width: int = 14336,
    input_rmsnorm_degree: int = 16,
    post_attention_rmsnorm_degree: int = 32,
    silu_degree: int = 48,
    silu_multiplicative_depth: int = 6,
) -> dict[str, object]:
    """Static component audit without pretending polynomial internals are free."""

    schedule = LayerSchedule() if schedule is None else schedule
    # The stages below read the schedule's decisions as plain locals;
    # binding them here keeps the object at the boundary only.
    linear_baby_steps = schedule.linear_baby_steps
    qkv_hoist_strategy = schedule.qkv_hoist_strategy
    output_hoist_strategy = schedule.output_hoist_strategy
    mlp_hoist_strategy = schedule.mlp_hoist_strategy
    qkv_max_plaintext_rows = schedule.qkv_max_plaintext_rows
    output_max_plaintext_rows = schedule.output_max_plaintext_rows
    mlp_max_plaintext_rows = schedule.mlp_max_plaintext_rows
    sparse_max_baby_offsets = schedule.sparse_max_baby_offsets
    shear_baby_steps = schedule.shear_baby_steps
    pv_complex_delta_folding = schedule.pv_complex_delta_folding
    pv_prefold_after_softmax = schedule.pv_prefold_after_softmax
    pair_softmax_pre_exp_bootstrap = schedule.pair_softmax_pre_exp_bootstrap
    reuse_softmax_round_square = schedule.reuse_softmax_round_square
    direct_softmax_slim_inactive_fill = schedule.direct_softmax_slim_inactive_fill
    fuse_softmax_slim_chebyshev_map = schedule.fuse_softmax_slim_chebyshev_map
    adaptive_rmsnorm_slim_refresh = schedule.adaptive_rmsnorm_slim_refresh
    rmsnorm2_slim_log_slots = schedule.rmsnorm2_slim_log_slots
    batched_sparse_attention_input = schedule.batched_sparse_attention_input
    fuse_rmsnorm_output_scale = schedule.fuse_rmsnorm_output_scale
    fuse_silu_domain_map = schedule.fuse_silu_domain_map
    preserve_prebootstrap_residual = schedule.preserve_prebootstrap_residual
    softmax_main_bootstrap_iterations = schedule.softmax_main_bootstrap_iterations
    softmax_main_deep_bootstrap = schedule.softmax_main_deep_bootstrap
    softmax_lambda_bootstrap_iterations = schedule.softmax_lambda_bootstrap_iterations
    match_softmax_single_pass_bootstrap_level = schedule.match_softmax_single_pass_bootstrap_level

    attention_input = pipeline_static_audit(
        feature_layout,
        attention_layout,
        qkv_operator_config=_linear_config(
            schedule, qkv_max_plaintext_rows, qkv_hoist_strategy
        ),
        sparse_max_baby_offsets=int(sparse_max_baby_offsets),
        shear_baby_steps=int(shear_baby_steps),
        include_scores=False,
    )
    qk = delta_qk_operation_counts(
        attention_layout,
        complex_feature_folding=True,
        key_features_pre_folded=True,
        query_features_pre_scaled=True,
        query_features_pre_folded=True,
    )
    pv = delta_pv_operation_counts(
        attention_layout,
        complex_delta_folding=bool(pv_complex_delta_folding),
        probability_pre_folded=bool(pv_prefold_after_softmax),
        value_pre_folded=bool(pv_prefold_after_softmax),
    )
    pairfm = pairfm_output_complexity(
        feature_layout,
        attention_layout,
    )
    input_rms = rmsnorm_complexity(
        feature_layout,
        polynomial_degree=int(input_rmsnorm_degree),
        apply_gamma=not bool(fuse_rmsnorm_output_scale),
    )
    post_rms = rmsnorm_complexity(
        feature_layout,
        polynomial_degree=int(post_attention_rmsnorm_degree),
        apply_gamma=not bool(fuse_rmsnorm_output_scale),
    )
    softmax = delta_softmax_alg2_complexity(
        attention_layout,
        reuse_round_square=bool(reuse_softmax_round_square),
        direct_slim_inactive_fill=bool(
            direct_softmax_slim_inactive_fill
        ),
        fuse_slim_chebyshev_map=bool(
            fuse_softmax_slim_chebyshev_map
        ),
    )
    mlp = mlp_complexity(
        feature_layout,
        intermediate_width=int(intermediate_width),
        silu_polynomial_degree=int(silu_degree),
        silu_multiplicative_depth=int(silu_multiplicative_depth),
        baby_steps=int(linear_baby_steps),
        max_plaintext_rows=int(mlp_max_plaintext_rows),
        reuse_baby_rotations_across_outputs=True,
        fuse_silu_domain_map=bool(fuse_silu_domain_map),
    )
    w_rows = int(feature_layout.hidden_dim)
    w_ptct = w_rows * (int(feature_layout.cipher_count) // 2)
    residual_bootstrap_copies = (
        0
        if bool(preserve_prebootstrap_residual)
        else 2 * int(feature_layout.cipher_count)
    )
    return {
        "shape": {
            "seq_len": int(feature_layout.seq_len),
            "hidden_dim": int(feature_layout.hidden_dim),
            "intermediate_width": int(intermediate_width),
            "canonical_cipher_count": int(feature_layout.cipher_count),
        },
        "stages": {
            "input_rmsnorm": input_rms.to_dict(),
            "feature_major_qkv_to_attention": attention_input.to_dict(),
            "qk": qk,
            "softmax": softmax.to_dict(),
            "pv": pv,
            "pairfm_output_conversion": pairfm.to_dict(),
            "w_o": {
                "encoded_weight_rows": w_rows,
                "pt_ct_multiplications": w_ptct,
                "multiplicative_depth": 1,
                "max_plaintext_rows": int(output_max_plaintext_rows),
            },
            "post_attention_rmsnorm": post_rms.to_dict(),
            "mlp": mlp.to_dict(),
            "residuals": {
                "ciphertext_additions": 2 * int(feature_layout.cipher_count),
                "bootstrap_identity_scalar_multiplications": int(
                    residual_bootstrap_copies
                ),
                "bootstrap_identity_rescales": int(
                    residual_bootstrap_copies
                ),
            },
        },
        "settings": {
            "pv_complex_delta_folding": bool(pv_complex_delta_folding),
            "pv_prefold_after_softmax": bool(pv_prefold_after_softmax),
            "pair_softmax_pre_exp_bootstrap": bool(
                pair_softmax_pre_exp_bootstrap
            ),
            "reuse_softmax_round_square": bool(
                reuse_softmax_round_square
            ),
            "direct_softmax_slim_inactive_fill": bool(
                direct_softmax_slim_inactive_fill
            ),
            "fuse_softmax_slim_chebyshev_map": bool(
                fuse_softmax_slim_chebyshev_map
            ),
            "softmax_main_bootstrap_iterations": (
                None
                if softmax_main_bootstrap_iterations is None
                else int(softmax_main_bootstrap_iterations)
            ),
            "softmax_main_deep_bootstrap": bool(
                softmax_main_deep_bootstrap
            ),
            "softmax_lambda_bootstrap_iterations": (
                None
                if softmax_lambda_bootstrap_iterations is None
                else int(softmax_lambda_bootstrap_iterations)
            ),
            "match_softmax_single_pass_bootstrap_level": bool(
                match_softmax_single_pass_bootstrap_level
            ),
            "adaptive_rmsnorm_slim_refresh": bool(
                adaptive_rmsnorm_slim_refresh
            ),
            "rmsnorm2_slim_log_slots": (
                None
                if rmsnorm2_slim_log_slots is None
                else int(rmsnorm2_slim_log_slots)
            ),
            "batched_sparse_attention_input": bool(
                batched_sparse_attention_input
            ),
            "fuse_rmsnorm_output_scale": bool(
                fuse_rmsnorm_output_scale
            ),
            "fuse_silu_domain_map": bool(fuse_silu_domain_map),
            "preserve_prebootstrap_residual": bool(
                preserve_prebootstrap_residual
            ),
            "qkv_max_plaintext_rows": int(qkv_max_plaintext_rows),
            "output_max_plaintext_rows": int(output_max_plaintext_rows),
            "mlp_max_plaintext_rows": int(mlp_max_plaintext_rows),
            "qkv_hoist_strategy": str(qkv_hoist_strategy),
            "output_hoist_strategy": str(output_hoist_strategy),
            "mlp_hoist_strategy": str(mlp_hoist_strategy),
        },
    }


def evaluate(
    hidden_states: np.ndarray,
    *,
    layer_idx: int,
    weights: LayerWeights,
    approximations: LayerApproximations,
    attention_layout: AttentionPairLayout,
    config: Llama3CKKSConfig,
    crypto_context,
    input_ciphers: tuple[object, ...] | list[object],
    bootstrap_operator=None,
    schedule: LayerSchedule | None = None,
    refresh_input: bool = False,
    attention_token_valid_mask: np.ndarray | None = None,
    rope_theta: float = 500000.0,
    verify: bool = False,
    expected_output: np.ndarray | None = None,
) -> LayerResult:
    """Run one continuous ciphertext-only transformer layer in real FM."""

    schedule = LayerSchedule() if schedule is None else schedule
    # The stages below read the schedule's decisions as plain locals;
    # binding them here keeps the object at the boundary only.
    steady_state_attention_refresh = schedule.steady_state_attention_refresh
    refresh_softmax_exp_inputs = schedule.refresh_softmax_exp_inputs
    refresh_softmax_exp_outputs = schedule.refresh_softmax_exp_outputs
    softmax_main_deep_bootstrap = schedule.softmax_main_deep_bootstrap
    pair_softmax_pre_exp_bootstrap = schedule.pair_softmax_pre_exp_bootstrap
    reuse_softmax_round_square = schedule.reuse_softmax_round_square
    direct_softmax_slim_inactive_fill = schedule.direct_softmax_slim_inactive_fill
    fuse_softmax_slim_chebyshev_map = schedule.fuse_softmax_slim_chebyshev_map
    adaptive_rmsnorm_slim_refresh = schedule.adaptive_rmsnorm_slim_refresh
    batched_sparse_attention_input = schedule.batched_sparse_attention_input
    fuse_rmsnorm_output_scale = schedule.fuse_rmsnorm_output_scale
    fuse_silu_domain_map = schedule.fuse_silu_domain_map
    softmax_main_bootstrap_iterations = schedule.softmax_main_bootstrap_iterations
    softmax_lambda_bootstrap_iterations = schedule.softmax_lambda_bootstrap_iterations
    match_softmax_single_pass_bootstrap_level = schedule.match_softmax_single_pass_bootstrap_level
    pairfm_bootstrap_mode = schedule.pairfm_bootstrap_mode
    pairfm_post_refresh_input_limbs = schedule.pairfm_post_refresh_input_limbs
    qkv_max_plaintext_rows = schedule.qkv_max_plaintext_rows
    output_max_plaintext_rows = schedule.output_max_plaintext_rows
    mlp_max_plaintext_rows = schedule.mlp_max_plaintext_rows
    sparse_max_baby_offsets = schedule.sparse_max_baby_offsets
    shear_baby_steps = schedule.shear_baby_steps
    residual_bootstrap_level_drop = schedule.residual_bootstrap_level_drop
    bootstrap_log_slots = schedule.bootstrap_log_slots
    bootstrap_level_budget = schedule.bootstrap_level_budget
    rmsnorm2_slim_log_slots = schedule.rmsnorm2_slim_log_slots
    residual_bootstrap_iterations = schedule.residual_bootstrap_iterations
    input_residual_bootstrap_iterations = schedule.input_residual_bootstrap_iterations
    post_attention_residual_bootstrap_iterations = schedule.post_attention_residual_bootstrap_iterations
    residual_bootstrap_precision = schedule.residual_bootstrap_precision
    qkv_hoist_strategy = schedule.qkv_hoist_strategy
    output_hoist_strategy = schedule.output_hoist_strategy
    mlp_hoist_strategy = schedule.mlp_hoist_strategy
    mlp_input_level_drop = schedule.mlp_input_level_drop
    pair_residual_bootstrap = schedule.pair_residual_bootstrap
    preserve_prebootstrap_residual = schedule.preserve_prebootstrap_residual
    qk_output_level = schedule.qk_output_level
    attention_rotation_chunk_size = schedule.attention_rotation_chunk_size
    pv_complex_delta_folding = schedule.pv_complex_delta_folding
    pv_prefold_after_softmax = schedule.pv_prefold_after_softmax
    evict_rotation_cuda_cache_before_output_projection = schedule.evict_rotation_cuda_cache_before_output_projection
    evict_rotation_cuda_cache_before_mlp = schedule.evict_rotation_cuda_cache_before_mlp
    evict_rotation_cuda_cache_after_layer = schedule.evict_rotation_cuda_cache_after_layer
    profile_device_memory = schedule.profile_device_memory

    # The stage code below reads the individual tensors and coefficient
    # entries; bind them once here so the schedule keeps one weights object
    # and one approximations object at its boundary.
    input_norm_weight = weights.input_layernorm
    q_weight = weights.q_proj
    k_weight = weights.k_proj
    v_weight = weights.v_proj
    o_weight = weights.o_proj
    post_attention_norm_weight = weights.post_attention_layernorm
    gate_weight = weights.gate_proj
    up_weight = weights.up_proj
    down_weight = weights.down_proj
    input_norm_coeffs = approximations.input_norm
    post_attention_norm_coeffs = approximations.post_attention_norm
    silu_entry = approximations.silu
    softmax_config = approximations.softmax

    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    layer_idx = int(layer_idx)
    os.environ["LLAMA_FHE_CURRENT_LAYER"] = str(layer_idx)
    if bool(verify) and expected_output is None:
        raise ValueError("verify=True requires expected_output.")
    mlp_input_level_drop = int(mlp_input_level_drop)
    if mlp_input_level_drop < 0:
        raise ValueError("mlp_input_level_drop must be non-negative.")
    residual_bootstrap_level_drop = int(residual_bootstrap_level_drop)
    if residual_bootstrap_level_drop < 0:
        raise ValueError(
            "residual_bootstrap_level_drop must be non-negative."
        )

    steady_attention_default = (
        layer_idx > 0 or bool(steady_state_attention_refresh)
    )
    if refresh_softmax_exp_inputs is None:
        refresh_softmax_exp_inputs = steady_attention_default
    if bool(refresh_softmax_exp_inputs) and bool(
        refresh_softmax_exp_outputs
    ):
        raise ValueError(
            "Softmax main-stream bootstrap cannot run both before and after exp."
        )
    pairfm_bootstrap_mode = str(pairfm_bootstrap_mode).strip().lower()
    if pairfm_bootstrap_mode not in {"auto", "always", "off"}:
        raise ValueError(
            "pairfm_bootstrap_mode must be 'auto', 'always', or 'off'."
        )

    slots = 1 << (int(config.simulator.logN) - 1)
    layout = FeatureMajorPrefillLayout(
        seq_len=int(hidden_states.shape[0]),
        hidden_dim=int(hidden_states.shape[1]),
        slots=slots,
    )
    if int(attention_layout.slots) != slots:
        raise ValueError("feature-major and attention slot counts must match.")
    rotations = required_rotations(
        layout,
        attention_layout,
        schedule=schedule,
        log_n=int(config.simulator.logN),
    )
    if crypto_context is None or input_ciphers is None:
        raise ValueError(
            "layer evaluation requires an application-owned crypto context "
            "and encrypted feature-major inputs."
        )
    if bootstrap_operator is None:
        raise ValueError(
            "layer evaluation requires a prepared bootstrap operator; "
            "Llama3Model.forward prepares one when not supplied."
        )

    wall_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    nested: dict[str, object] = {
        "output_levels": {},
        "residual_identity": {
            "preserve_prebootstrap": bool(preserve_prebootstrap_residual),
        },
    }
    if bool(profile_device_memory):
        nested["device_memory"] = {}

    def record_device_memory(label: str) -> None:
        if not bool(profile_device_memory):
            return
        snapshot = cuda_memory_snapshot(crypto_context.device)
        snapshot["elapsed_seconds"] = float(time.perf_counter() - wall_start)
        nested["device_memory"][str(label)] = snapshot

    softmax_bootstrap_calls = 0
    softmax_native_bootstrap_calls = 0

    input_norm = qkv = attention = None
    first_residual = post_norm = mlp = final_residual = None
    qkv_inputs: tuple[object, ...] = ()
    owns_qkv_inputs = False
    projected: tuple[object, ...] = ()
    mlp_inputs: tuple[object, ...] = ()
    refreshed_layer_input: tuple[object, ...] = ()
    refreshed_attention_residual: tuple[object, ...] = ()

    def debug_range(label: str, ciphers, *, variance_interval=None) -> None:
        dump_dir = os.environ.get("LLAMA_FHE_STAGE_DUMP", "").strip()
        diag = os.environ.get("LLAMA_FHE_STAGE_DIAGNOSTICS", "") == "1"
        only = os.environ.get("LLAMA_FHE_STAGE_DIAGNOSTICS_LAYER", "").strip()
        layer_selected = not only or int(only) == int(layer_idx)
        print_this = diag and layer_selected
        dump_this = bool(dump_dir) and layer_selected
        if not print_this and not dump_this:
            return
        packed = np.stack(
            [
                np.asarray(crypto_context.decrypt(cipher), dtype=np.float64)
                for cipher in ciphers
            ]
        )
        values = layout.unpack(packed)
        finite = np.isfinite(values)
        finite_values = values[finite]
        limbs = sorted({int(cipher.state.cur_limbs) for cipher in ciphers})
        variance = np.mean(np.square(values, dtype=np.float64), axis=-1)
        finite_var = variance[np.isfinite(variance)]
        extra = f" limbs={limbs}"
        if finite_var.size:
            extra += (
                f" var=[{float(np.min(finite_var)):.4g},"
                f"{float(np.max(finite_var)):.4g}]"
            )
        if variance_interval is not None and finite_var.size:
            lo, hi = (float(v) for v in variance_interval)
            oob = float(np.mean((variance < lo) | (variance > hi)))
            extra += (
                f" rmsn_interval=[{lo:.4g},{hi:.4g}] var_oob_frac={oob:.4g}"
            )
        print(
            f"[stage-diagnostic] layer={layer_idx + 1} stage={label} "
            f"finite={bool(np.all(finite))} "
            f"max_abs={None if not finite_values.size else float(np.max(np.abs(finite_values)))} "
            f"mean_abs={None if not finite_values.size else float(np.mean(np.abs(finite_values)))}"
            f"{extra}",
            flush=True,
        )
        if dump_this:
            os.makedirs(dump_dir, exist_ok=True)
            np.save(
                os.path.join(dump_dir, f"l{layer_idx}_{label}.npy"),
                values.astype(np.float32, copy=False),
            )

    try:
        input_ciphers = tuple(input_ciphers)
        if len(input_ciphers) != int(layout.cipher_count):
            raise ValueError(
                f"layer expects {layout.cipher_count} FM inputs, got "
                f"{len(input_ciphers)}."
            )
        record_device_memory("layer_entry")
        debug_range(
            "layer_input",
            input_ciphers,
            variance_interval=input_norm_coeffs["fit_interval"],
        )

        layer_bootstrap_calls_before = int(bootstrap_operator.calls)
        layer_native_bootstrap_calls_before = int(
            bootstrap_operator.native_calls
        )
        record_device_memory("bootstrap_ready")

        start = time.perf_counter()
        # Minimum post-normalization budget for the QKV -> attention ->
        # pre-exp checkpoint schedule.  The score stream only needs 10
        # (QKV 1 + K adapter 4 + QK 2 + preprocessing 1 + bootstrap floor 2),
        # but the VALUE stream is never re-bootstrapped and must reach the
        # PairFM pairing at >= 6 limbs, or every steady layer pays an
        # 8-cipher bootstrap-to-23-drop-to-6 (~1.4s).  Three extra limbs
        # carry the value chain to that threshold: full model 600.4s ->
        # 560.3s (-6.7%), Paris fixture passes, peak memory unchanged.
        # LLAMA_FHE_ATTN_EXTRA_LIMBS adjusts the margin for experiments.
        minimum_attention_rmsnorm_limbs = 13 + int(
            os.environ.get("LLAMA_FHE_ATTN_EXTRA_LIMBS", "0")
        )
        if bool(refresh_input):
            # QKV(1) + K adapter(4) + QK(2) + score preprocessing(1),
            # followed by a bootstrap that still needs at least two limbs.
            input_norm = bootstrap_rmsnorm_fhe(
                input_ciphers,
                input_norm_weight,
                coeffs=input_norm_coeffs,
                layout=layout,
                crypto_context=crypto_context,
                bootstrap_operator=bootstrap_operator,
                pair_inputs=bool(pair_residual_bootstrap),
                bootstrap_iterations=int(
                    residual_bootstrap_iterations
                    if input_residual_bootstrap_iterations is None
                    else input_residual_bootstrap_iterations
                ),
                bootstrap_precision_bits=int(residual_bootstrap_precision),
                bootstrap_output_level_drop=(
                    int(residual_bootstrap_level_drop)
                ),
                minimum_rmsnorm_output_limbs=(
                    minimum_attention_rmsnorm_limbs
                ),
                return_refreshed_inputs=(
                    not bool(preserve_prebootstrap_residual)
                ),
                defer_output_scale=bool(fuse_rmsnorm_output_scale),
                adaptive_slim_refresh=bool(adaptive_rmsnorm_slim_refresh),
                function_prefix=(
                    f"model.layers.{layer_idx}.input_layernorm.feature_major"
                ),
            )
        else:
            input_norm = rmsnorm_fhe(
                input_ciphers,
                input_norm_weight,
                coeffs=input_norm_coeffs,
                layout=layout,
                crypto_context=crypto_context,
                verify=False,
                defer_output_scale=bool(fuse_rmsnorm_output_scale),
                slim_refresh=(
                    SlimRefresh(
                        bootstrap_operator=bootstrap_operator,
                        minimum_output_limbs=(
                            minimum_attention_rmsnorm_limbs
                        ),
                        iterations=int(residual_bootstrap_iterations),
                        precision_bits=int(residual_bootstrap_precision),
                    )
                    if adaptive_rmsnorm_slim_refresh
                    else None
                ),
                function_prefix=(
                    f"model.layers.{layer_idx}.input_layernorm.feature_major"
                ),
            )
        stage_seconds["input_rmsnorm"] = time.perf_counter() - start
        nested["input_rmsnorm"] = input_norm.stage_seconds
        nested["input_rmsnorm_level_schedule"] = dict(
            input_norm.level_schedule
        )
        nested["output_levels"]["input_rmsnorm"] = input_norm.output_levels
        nested["output_levels"]["input_rmsnorm_slim_bootstrap"] = tuple(
            input_norm.slim_bootstrap_output_levels
        )
        if bool(refresh_input):
            refreshed_layer_input = input_norm.refreshed_input_ciphers
            input_norm.refreshed_input_ciphers = ()
            if bool(preserve_prebootstrap_residual) and refreshed_layer_input:
                raise RuntimeError(
                    "pre-bootstrap residual preservation unexpectedly "
                    "materialized an input residual copy."
                )
            nested["output_levels"]["input_bootstrap_raw"] = tuple(
                input_norm.bootstrap_output_levels
            )
            nested["output_levels"]["input_residual_after_bootstrap"] = (
                _cipher_levels(
                    refreshed_layer_input, crypto_context=crypto_context
                )
            )
            if refreshed_layer_input:
                debug_range(
                    "refreshed_layer_input",
                    refreshed_layer_input,
                    variance_interval=input_norm_coeffs["fit_interval"],
                )
        debug_range("input_rmsnorm", input_norm.ciphers)
        record_device_memory("input_rmsnorm_complete")

        # Deferring the final RMSNorm output-scale multiplication saves one
        # rescale.  Modulus-drop that reclaimed limb before the dense QKV so
        # the large linear runs at the same modulus size as the unfused path.
        # Very high-degree per-layer RMSNorm polynomials can consume more than
        # the bootstrap's entire QKV headroom.  In that case refresh only the
        # normalized branch, then drop it to the exact attention budget; the
        # persistent residual continues to borrow the original L40 stream.
        qkv_input_limbs = min(
            int(cipher.state.cur_limbs) for cipher in input_norm.ciphers
        )
        refreshed_qkv_input = (
            qkv_input_limbs < minimum_attention_rmsnorm_limbs
        )
        requested_qkv_input_drop = int(
            bool(fuse_rmsnorm_output_scale)
        )
        maximum_safe_qkv_input_drop = max(
            0, qkv_input_limbs - minimum_attention_rmsnorm_limbs
        )
        scheduled_qkv_input_drop = min(
            requested_qkv_input_drop, maximum_safe_qkv_input_drop
        )
        qkv_input_level_drop = 0
        start = time.perf_counter()
        if refreshed_qkv_input:
            raised_qkv_inputs = refresh_ciphers(
                input_norm.ciphers,
                crypto_context=crypto_context,
                bootstrap_operator=bootstrap_operator,
                iterations=int(residual_bootstrap_iterations),
                precision_bits=int(residual_bootstrap_precision),
                function_prefix=(
                    f"model.layers.{layer_idx}.input_rmsnorm_qkv"
                ),
            )
            try:
                raised_qkv_limbs = min(
                    int(cipher.state.cur_limbs)
                    for cipher in raised_qkv_inputs
                )
                if raised_qkv_limbs < minimum_attention_rmsnorm_limbs:
                    raise RuntimeError(
                        "QKV emergency refresh did not restore its consumer "
                        f"budget: {raised_qkv_limbs} < "
                        f"{minimum_attention_rmsnorm_limbs}."
                    )
                qkv_input_level_drop = (
                    raised_qkv_limbs - minimum_attention_rmsnorm_limbs
                )
                qkv_inputs = drop_levels(
                    raised_qkv_inputs,
                    qkv_input_level_drop,
                    crypto_context=crypto_context,
                )
            finally:
                release_all(raised_qkv_inputs)
            owns_qkv_inputs = True
        elif scheduled_qkv_input_drop:
            qkv_input_level_drop = scheduled_qkv_input_drop
            qkv_inputs = drop_levels(
                input_norm.ciphers,
                qkv_input_level_drop,
                crypto_context=crypto_context,
            )
            owns_qkv_inputs = True
            if str(crypto_context.device).startswith("cuda"):
                easyfhe.cuda.synchronize()
        else:
            qkv_inputs = tuple(input_norm.ciphers)
        stage_seconds[
            "pre_qkv_bootstrap"
            if refreshed_qkv_input
            else "qkv_input_modulus_drop"
        ] = time.perf_counter() - start
        nested["qkv_input_modulus_drop"] = {
            "rmsnorm_scale_level_reclaimed": int(
                bool(fuse_rmsnorm_output_scale)
            ),
            "before_min_limbs": int(qkv_input_limbs),
            "minimum_required_limbs": int(
                minimum_attention_rmsnorm_limbs
            ),
            "requested": int(requested_qkv_input_drop),
            "maximum_safe": int(maximum_safe_qkv_input_drop),
            "scheduled": int(scheduled_qkv_input_drop),
            "applied": int(qkv_input_level_drop),
            "clamped": bool(
                not refreshed_qkv_input
                and scheduled_qkv_input_drop != requested_qkv_input_drop
            ),
            "overridden_by_emergency_bootstrap": bool(
                refreshed_qkv_input
            ),
            "after_emergency_bootstrap": bool(refreshed_qkv_input),
            "after_min_limbs": int(
                min(int(cipher.state.cur_limbs) for cipher in qkv_inputs)
            ),
        }
        if min(
            int(cipher.state.cur_limbs) for cipher in qkv_inputs
        ) < minimum_attention_rmsnorm_limbs:
            raise RuntimeError(
                "QKV input scheduling violated its consumer budget."
            )
        nested["output_levels"]["qkv_input"] = _cipher_levels(
            qkv_inputs, crypto_context=crypto_context
        )

        start = time.perf_counter()
        try:
            qkv = qkv_fhe(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                attention_layout=attention_layout,
                layout=layout,
                config=config,
                operator_config=_linear_config(
                    schedule, qkv_max_plaintext_rows, qkv_hoist_strategy
                ),
                extra_rotations=rotations,
                crypto_context=crypto_context,
                input_ciphers=qkv_inputs,
                verify=False,
                function_prefix=(
                    f"model.layers.{layer_idx}.self_attn.feature_major_qkv"
                ),
                input_column_scale=input_norm.deferred_output_scale,
            )
        finally:
            if owns_qkv_inputs:
                release_all(qkv_inputs)
            qkv_inputs = ()
            owns_qkv_inputs = False
        stage_seconds["qkv_projection"] = time.perf_counter() - start
        nested["qkv_projection"] = qkv.stage_seconds
        nested["output_levels"]["qkv_carriers"] = qkv.output_levels
        record_device_memory("qkv_projection_complete")
        input_norm.release()
        input_norm = None
        record_device_memory("qkv_input_released")

        softmax_calls_before = int(bootstrap_operator.calls)
        softmax_native_calls_before = int(bootstrap_operator.native_calls)
        softmax_cache_eviction_seconds_before = float(
            bootstrap_operator.cache_eviction_seconds
        )
        start = time.perf_counter()
        attention = attention_fhe_from_qkv_carriers(
            qkv.carrier_ciphers,
            carrier_layout=qkv.output_layout,
            feature_layout=layout,
            attention_layout=attention_layout,
            crypto_context=crypto_context,
            softmax_config=softmax_config,
            model_config=config,
            operator_config=AttentionOperatorConfig(
                rope_theta=float(rope_theta),
                sparse_max_baby_offsets=int(sparse_max_baby_offsets),
                shear_baby_steps=int(shear_baby_steps),
                # The score stream sheds exactly the limbs the entry budget
                # holds above the tight 10-limb Q/K schedule (the extras
                # exist only so the VALUE stream reaches the PairFM 6-limb
                # pairing threshold; the pre-exp bootstrap resets Q/K).
                qk_score_level_drop=max(
                    0, int(minimum_attention_rmsnorm_limbs) - 10
                ),
                rotation_mode="fast",
                rotation_chunk_size=int(attention_rotation_chunk_size),
                qk_output_level=qk_output_level,
                qk_complex_feature_folding=True,
                qk_key_features_pre_folded=True,
                qk_query_features_pre_scaled=True,
                qk_query_features_pre_folded=True,
                pv_complex_delta_folding=bool(pv_complex_delta_folding),
                pv_prefold_after_softmax=bool(pv_prefold_after_softmax),
                # Bootstrap only after the public causal mask, score scale,
                # and U-hat shift.  Raw QK scores can be hundreds in
                # magnitude and are outside the bootstrap identity domain.
                refresh_softmax_exp_inputs=bool(
                    refresh_softmax_exp_inputs
                ),
                refresh_softmax_exp_outputs=bool(
                    refresh_softmax_exp_outputs
                ),
                # The deep (3,3)/+2-level program only pays off where the
                # paired PairFM inputs would otherwise fall below their
                # 6-limb refresh threshold — the steady-layer profile.
                softmax_main_deep_bootstrap=bool(
                    softmax_main_deep_bootstrap and steady_attention_default
                ),
                # Pairing optimizes an existing pre-exp checkpoint.  Layer 0
                # has no such refresh by default, so the selected policy must
                # become a no-op instead of making that configuration invalid.
                pair_softmax_pre_exp_bootstrap=bool(
                    pair_softmax_pre_exp_bootstrap
                    and refresh_softmax_exp_inputs
                ),
                reuse_softmax_round_square=bool(
                    reuse_softmax_round_square
                ),
                direct_softmax_slim_inactive_fill=bool(
                    direct_softmax_slim_inactive_fill
                ),
                fuse_softmax_slim_chebyshev_map=bool(
                    fuse_softmax_slim_chebyshev_map
                ),
                batched_sparse_input_bsgs=bool(
                    batched_sparse_attention_input
                ),
                softmax_main_bootstrap_iterations=(
                    softmax_main_bootstrap_iterations
                ),
                softmax_lambda_bootstrap_iterations=(
                    softmax_lambda_bootstrap_iterations
                ),
                match_softmax_single_pass_bootstrap_level=bool(
                    match_softmax_single_pass_bootstrap_level
                ),
                refresh_output_before_pairfm=(
                    pairfm_bootstrap_mode == "always"
                ),
                pairfm_refresh_min_input_limbs=(
                    6 if pairfm_bootstrap_mode == "auto" else None
                ),
                pairfm_post_refresh_input_limbs=(
                    pairfm_post_refresh_input_limbs
                ),
                checkpoint_bootstrap_precision_bits=int(
                    residual_bootstrap_precision
                ),
            ),
            layer_idx=layer_idx,
            softmax_bootstrap=bootstrap_operator,
            token_valid_mask=attention_token_valid_mask,
            function_prefix=f"model.layers.{layer_idx}.self_attn.attention",
        )
        stage_seconds["attention"] = time.perf_counter() - start
        nested["attention"] = attention.stage_seconds
        nested["attention_input_scheduler"] = dict(
            attention.input_scheduler_stats
        )
        nested["output_levels"].update(
            {
                f"attention_{name}": tuple(levels)
                for name, levels in attention.level_trace.items()
            }
        )
        nested["output_levels"]["attention_w_o_input"] = (
            attention.output_levels
        )
        softmax_bootstrap_calls = (
            int(bootstrap_operator.calls) - softmax_calls_before
        )
        softmax_native_bootstrap_calls = (
            int(bootstrap_operator.native_calls)
            - softmax_native_calls_before
        )
        stage_seconds["softmax_bootstrap_rotation_cache_evict"] = float(
            bootstrap_operator.cache_eviction_seconds
            - softmax_cache_eviction_seconds_before
        )
        record_device_memory("attention_complete")
        qkv.release()
        qkv = None
        record_device_memory("qkv_released")

        if bool(evict_rotation_cuda_cache_before_output_projection):
            start = time.perf_counter()
            crypto_context.context.clear_cuda_rotation_cache(
                keep_rotations=BootstrapOperator.required_rotations(
                    log_n=int(config.simulator.logN),
                    log_slots=int(bootstrap_log_slots),
                    level_budget=tuple(
                        int(v) for v in bootstrap_level_budget
                    ),
                    slim_log_slots=rmsnorm2_slim_log_slots,
                ),
            )
            stage_seconds["pre_output_projection_rotation_cache_evict"] = (
                time.perf_counter() - start
            )
        start = time.perf_counter()
        trim_device_allocator_cache(crypto_context.device)
        stage_seconds["pre_output_projection_allocator_trim"] = (
            time.perf_counter() - start
        )
        record_device_memory("pre_output_projection_cache_ready")

        w_profile: dict[str, float] = {}
        start = time.perf_counter()
        projected = w_o_fhe(
            attention.w_o_input_ciphers,
            o_weight,
            feature_layout=layout,
            attention_layout=attention_layout,
            crypto_context=crypto_context,
            operator_config=_linear_config(
                schedule, output_max_plaintext_rows, output_hoist_strategy
            ),
            dtype=np.float64,
            profile=w_profile,
        )
        stage_seconds["output_projection"] = time.perf_counter() - start
        nested["output_projection"] = w_profile
        nested["output_levels"]["attention_w_o"] = _cipher_levels(
            projected, crypto_context=crypto_context
        )
        debug_range("attention_w_o", projected)
        record_device_memory("output_projection_complete")
        attention.release()
        attention = None
        record_device_memory("attention_released")

        start = time.perf_counter()
        first_identity, first_identity_source = _select_residual_identity(
            input_ciphers,
            refreshed_layer_input,
            preserve_prebootstrap=bool(preserve_prebootstrap_residual),
            refresh_was_run=bool(refresh_input),
        )
        first_identity_audit = _residual_state_audit(
            first_identity,
            projected,
            crypto_context=crypto_context,
        )
        first_residual = residual_add(
            first_identity,
            projected,
            layout=layout,
            crypto_context=crypto_context,
            verify=False,
        )
        stage_seconds["attention_residual"] = time.perf_counter() - start
        nested["output_levels"]["attention_residual"] = (
            first_residual.output_levels
        )
        nested["residual_identity"]["attention"] = {
            "source": first_identity_source,
            "borrowed": True,
            "materialized_bootstrap_copy": bool(refreshed_layer_input),
            **first_identity_audit,
            "output": _cipher_stream_state_audit(
                first_residual.ciphers, crypto_context=crypto_context
            ),
        }
        debug_range(
            "attention_residual",
            first_residual.ciphers,
            variance_interval=post_attention_norm_coeffs["fit_interval"],
        )
        record_device_memory("attention_residual_complete")
        release_all(projected)
        projected = ()
        release_all(refreshed_layer_input)
        refreshed_layer_input = ()
        start = time.perf_counter()
        # Suffix after the SiLU PS depth: gate(1)+up-mul(1)+down(1)+residual
        # match and a two-limb floor so the next layer can still bootstrap.
        # silu_priority_v1 Layer0: suffix 5 is finite in isolation but leaves
        # output_levels=[37] (1 limb) and the following input RMSNorm BS
        # dies on rescale. Keep 6 for streamed models.
        # Override with LLAMA_FHE_MLP_SUFFIX_LIMBS.
        mlp_suffix_limbs = int(os.environ.get("LLAMA_FHE_MLP_SUFFIX_LIMBS", "6"))
        if mlp_suffix_limbs < 3:
            raise ValueError("LLAMA_FHE_MLP_SUFFIX_LIMBS must be at least 3.")
        required_mlp_input_limbs = (
            chebyshev_ps_mul_depth(silu_entry["coefficients"])
            + mlp_suffix_limbs
            - int(bool(fuse_silu_domain_map))
        )
        post_norm = bootstrap_rmsnorm_fhe(
            first_residual.ciphers,
            post_attention_norm_weight,
            coeffs=post_attention_norm_coeffs,
            layout=layout,
            crypto_context=crypto_context,
            bootstrap_operator=bootstrap_operator,
            pair_inputs=bool(pair_residual_bootstrap),
            bootstrap_iterations=int(
                residual_bootstrap_iterations
                if post_attention_residual_bootstrap_iterations is None
                else post_attention_residual_bootstrap_iterations
            ),
            bootstrap_precision_bits=int(residual_bootstrap_precision),
            bootstrap_output_level_drop=(
                int(residual_bootstrap_level_drop)
            ),
            minimum_rmsnorm_output_limbs=required_mlp_input_limbs,
            return_refreshed_inputs=(
                not bool(preserve_prebootstrap_residual)
            ),
            defer_output_scale=bool(fuse_rmsnorm_output_scale),
            adaptive_slim_refresh=bool(adaptive_rmsnorm_slim_refresh),
            use_slim_program=rmsnorm2_slim_log_slots is not None,
            function_prefix=(
                f"model.layers.{layer_idx}.post_attention_layernorm.feature_major"
            ),
        )
        stage_seconds["post_attention_bootstrap_rmsnorm"] = (
            time.perf_counter() - start
        )
        nested["post_attention_bootstrap_rmsnorm"] = post_norm.stage_seconds
        nested["post_attention_rmsnorm_level_schedule"] = dict(
            post_norm.level_schedule
        )
        nested["output_levels"]["post_attention_rmsnorm"] = (
            post_norm.output_levels
        )
        nested["output_levels"]["post_rmsnorm_slim_bootstrap"] = tuple(
            post_norm.slim_bootstrap_output_levels
        )
        refreshed_attention_residual = post_norm.refreshed_input_ciphers
        post_norm.refreshed_input_ciphers = ()
        if bool(preserve_prebootstrap_residual) and refreshed_attention_residual:
            raise RuntimeError(
                "pre-bootstrap residual preservation unexpectedly "
                "materialized a post-attention residual copy."
            )
        nested["output_levels"]["post_attention_bootstrap_raw"] = tuple(
            post_norm.bootstrap_output_levels
        )
        nested["output_levels"]["attention_residual_after_bootstrap"] = (
            _cipher_levels(
                refreshed_attention_residual, crypto_context=crypto_context
            )
        )
        if refreshed_attention_residual:
            debug_range(
                "refreshed_attention_residual",
                refreshed_attention_residual,
                variance_interval=post_attention_norm_coeffs["fit_interval"],
            )
        debug_range("post_attention_rmsnorm", post_norm.ciphers)
        record_device_memory("post_attention_rmsnorm_complete")

        if bool(evict_rotation_cuda_cache_before_mlp):
            start = time.perf_counter()
            crypto_context.context.clear_cuda_rotation_cache()
            stage_seconds["pre_mlp_rotation_cache_evict"] = (
                time.perf_counter() - start
            )
        start = time.perf_counter()
        trim_device_allocator_cache(crypto_context.device)
        stage_seconds["pre_mlp_allocator_trim"] = time.perf_counter() - start
        record_device_memory("pre_mlp_cache_ready")

        start = time.perf_counter()
        minimum_mlp_input_limbs = min(
            int(cipher.state.cur_limbs) for cipher in post_norm.ciphers
        )
        refreshed_mlp_input = (
            minimum_mlp_input_limbs < required_mlp_input_limbs
        )
        requested_fusion_mlp_drop = (
            int(bool(fuse_rmsnorm_output_scale))
            + int(bool(fuse_silu_domain_map))
        )
        requested_mlp_input_drop = (
            int(mlp_input_level_drop) + requested_fusion_mlp_drop
        )
        maximum_safe_mlp_input_drop = max(
            0, minimum_mlp_input_limbs - required_mlp_input_limbs
        )
        scheduled_mlp_input_drop = min(
            requested_mlp_input_drop, maximum_safe_mlp_input_drop
        )
        applied_mlp_input_drop = 0
        if refreshed_mlp_input:
            raised_mlp_inputs = refresh_ciphers(
                post_norm.ciphers,
                crypto_context=crypto_context,
                bootstrap_operator=bootstrap_operator,
                iterations=int(residual_bootstrap_iterations),
                precision_bits=int(residual_bootstrap_precision),
                function_prefix=(
                    f"model.layers.{layer_idx}.post_rmsnorm_mlp"
                ),
            )
            try:
                raised_limbs = min(
                    int(cipher.state.cur_limbs)
                    for cipher in raised_mlp_inputs
                )
                applied_mlp_input_drop = max(
                    0, raised_limbs - required_mlp_input_limbs
                )
                mlp_inputs = drop_levels(
                    raised_mlp_inputs,
                    applied_mlp_input_drop,
                    crypto_context=crypto_context,
                )
            finally:
                release_all(raised_mlp_inputs)
        else:
            mlp_inputs = drop_levels(
                post_norm.ciphers,
                scheduled_mlp_input_drop,
                crypto_context=crypto_context,
            )
            applied_mlp_input_drop = scheduled_mlp_input_drop
        if str(crypto_context.device).startswith("cuda"):
            easyfhe.cuda.synchronize()
        stage_seconds[
            "pre_mlp_bootstrap"
            if refreshed_mlp_input
            else "mlp_input_modulus_drop"
        ] = time.perf_counter() - start
        nested["output_levels"]["mlp_input"] = _cipher_levels(
            mlp_inputs, crypto_context=crypto_context
        )
        minimum_scheduled_mlp_limbs = min(
            int(cipher.state.cur_limbs) for cipher in mlp_inputs
        )
        if minimum_scheduled_mlp_limbs < required_mlp_input_limbs:
            raise RuntimeError(
                "MLP input scheduling violated its consumer budget: "
                f"{minimum_scheduled_mlp_limbs} < "
                f"{required_mlp_input_limbs}."
            )
        nested["mlp_input_modulus_drop"] = {
            "explicit": int(mlp_input_level_drop),
            "rmsnorm_scale_level_reclaimed": int(
                bool(fuse_rmsnorm_output_scale)
            ),
            "silu_map_level_reclaimed": int(bool(fuse_silu_domain_map)),
            "requested": int(requested_mlp_input_drop),
            "maximum_safe": int(maximum_safe_mlp_input_drop),
            "scheduled": int(scheduled_mlp_input_drop),
            "applied": int(applied_mlp_input_drop),
            "clamped": bool(
                not refreshed_mlp_input
                and scheduled_mlp_input_drop != requested_mlp_input_drop
            ),
            "overridden_by_emergency_bootstrap": bool(refreshed_mlp_input),
            "after_emergency_bootstrap": bool(refreshed_mlp_input),
            "before_min_limbs": int(minimum_mlp_input_limbs),
            "after_min_limbs": int(minimum_scheduled_mlp_limbs),
            "required_input_limbs": int(required_mlp_input_limbs),
        }

        start = time.perf_counter()
        mlp = mlp_fhe(
            hidden_states,
            gate_weight,
            up_weight,
            down_weight,
            silu_entry=silu_entry,
            layout=layout,
            config=config,
            operator_config=_linear_config(
                schedule, mlp_max_plaintext_rows, mlp_hoist_strategy
            ),
            crypto_context=crypto_context,
            input_ciphers=mlp_inputs,
            verify=False,
            input_column_scale=post_norm.deferred_output_scale,
            fuse_silu_domain_map=bool(fuse_silu_domain_map),
        )
        stage_seconds["mlp"] = time.perf_counter() - start
        nested["mlp"] = mlp.stage_seconds
        nested["output_levels"]["mlp"] = mlp.output_levels
        debug_range("mlp", mlp.ciphers)
        record_device_memory("mlp_complete")
        release_all(mlp_inputs)
        mlp_inputs = ()
        post_norm.release()
        post_norm = None

        start = time.perf_counter()
        final_identity, final_identity_source = _select_residual_identity(
            first_residual.ciphers,
            refreshed_attention_residual,
            preserve_prebootstrap=bool(preserve_prebootstrap_residual),
            refresh_was_run=True,
        )
        final_identity_audit = _residual_state_audit(
            final_identity,
            mlp.ciphers,
            crypto_context=crypto_context,
        )
        final_residual = residual_add(
            final_identity,
            mlp.ciphers,
            layout=layout,
            crypto_context=crypto_context,
            verify=False,
        )
        stage_seconds["mlp_residual"] = time.perf_counter() - start
        nested["output_levels"]["layer_output"] = final_residual.output_levels
        nested["residual_identity"]["mlp"] = {
            "source": final_identity_source,
            "borrowed": True,
            "materialized_bootstrap_copy": bool(
                refreshed_attention_residual
            ),
            **final_identity_audit,
            "output": _cipher_stream_state_audit(
                final_residual.ciphers, crypto_context=crypto_context
            ),
        }
        debug_range("layer_output", final_residual.ciphers)
        record_device_memory("layer_residual_complete")
        mlp.release()
        mlp = None
        first_residual.release()
        first_residual = None
        release_all(refreshed_attention_residual)
        refreshed_attention_residual = ()
        record_device_memory("layer_output_live")

        if bool(evict_rotation_cuda_cache_after_layer):
            start = time.perf_counter()
            crypto_context.context.clear_cuda_rotation_cache()
            stage_seconds["post_layer_rotation_cache_evict"] = (
                time.perf_counter() - start
            )
        start = time.perf_counter()
        trim_device_allocator_cache(crypto_context.device)
        stage_seconds["post_layer_allocator_trim"] = time.perf_counter() - start
        record_device_memory("post_layer_cache_evict")

        output = None
        max_abs_diff = mean_abs_diff = None
        if bool(verify):
            start = time.perf_counter()
            packed = np.stack(
                [
                    np.asarray(crypto_context.decrypt(cipher), dtype=np.float64)
                    for cipher in final_residual.ciphers
                ]
            )
            output = layout.unpack(packed)
            difference = output - np.asarray(expected_output, dtype=np.float32)
            max_abs_diff = float(np.max(np.abs(difference)))
            mean_abs_diff = float(np.mean(np.abs(difference)))
            if str(crypto_context.device).startswith("cuda"):
                easyfhe.cuda.synchronize()
            stage_seconds["verify_decrypt"] = time.perf_counter() - start

        output_ciphers = final_residual.ciphers
        final_residual.ciphers = ()
        return LayerResult(
            ciphers=tuple(output_ciphers),
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={str(k): float(v) for k, v in stage_seconds.items()},
            nested_stage_seconds=nested,
            softmax_bootstrap_calls=int(softmax_bootstrap_calls),
            residual_bootstrap_calls=(
                int(bootstrap_operator.calls)
                - layer_bootstrap_calls_before
                - int(softmax_bootstrap_calls)
            ),
            softmax_native_bootstrap_calls=int(
                softmax_native_bootstrap_calls
            ),
            residual_native_bootstrap_calls=(
                int(bootstrap_operator.native_calls)
                - layer_native_bootstrap_calls_before
                - int(softmax_native_bootstrap_calls)
            ),
            output_levels=_cipher_levels(output_ciphers, crypto_context=crypto_context),
            output=output,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
        )
    finally:
        if final_residual is not None:
            final_residual.release()
        if mlp is not None:
            mlp.release()
        if post_norm is not None:
            post_norm.release()
        release_all(mlp_inputs)
        release_all(refreshed_attention_residual)
        if first_residual is not None:
            first_residual.release()
        release_all(refreshed_layer_input)
        release_all(projected)
        if attention is not None:
            attention.release()
        if qkv is not None:
            qkv.release()
        if owns_qkv_inputs:
            release_all(qkv_inputs)
        if input_norm is not None:
            input_norm.release()


__all__ = [
    "LayerResult",
    "LayerSchedule",
    "evaluate",
    "operation_audit",
    "required_rotations",
]
