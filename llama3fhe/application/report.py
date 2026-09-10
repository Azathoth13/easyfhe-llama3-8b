"""Report assembly: the shared schedule snapshot and the audit payload.

Both reports the runner can emit — the ``--audit-only`` payload and the full
execution payload — must state which schedule the run used. That statement
lives here exactly once, as ``schedule_report``; the two callers embed it
under their own key ("schedule" in the audit, inside "runtime" for a real
run) and add only what is genuinely theirs.
"""

from __future__ import annotations

from ..schedule import LayerSchedule
from .settings import RunSettings


def schedule_report(
    args, settings: RunSettings, schedule: LayerSchedule
) -> dict[str, object]:
    """Every schedule decision and flag a report should state, resolved.

    Values are read from the ``LayerSchedule`` the model actually receives —
    never re-derived from ``args`` — so the report cannot drift from the
    execution. Keys keep the CLI flags' names, which downstream tooling
    already parses.
    """

    return {
        "context_depth": settings.depth,
        "input_level": settings.input_level,
        "input_level_source": settings.input_level_source,
        "bootstrap_post_levels": schedule.bootstrap_post_levels,
        "softmax_bootstrap_position": settings.softmax_bootstrap_position,
        "steady_state_layer": bool(args.steady_state_layer),
        "residual_bootstrap_level_drop": (
            schedule.residual_bootstrap_level_drop
        ),
        "preserve_prebootstrap_residual": (
            schedule.preserve_prebootstrap_residual
        ),
        "pairfm_bootstrap_mode": schedule.pairfm_bootstrap_mode,
        "pairfm_post_refresh_input_limbs": (
            schedule.pairfm_post_refresh_input_limbs
        ),
        "linear_hoist_strategy": schedule.qkv_hoist_strategy,
        "attention_rotation_chunk_size": (
            schedule.attention_rotation_chunk_size
        ),
        "softmax_complex_pre_exp_bootstrap": (
            schedule.pair_softmax_pre_exp_bootstrap
        ),
        "softmax_reuse_round_square": schedule.reuse_softmax_round_square,
        "softmax_direct_slim_inactive_fill": (
            schedule.direct_softmax_slim_inactive_fill
        ),
        "softmax_fuse_slim_chebyshev_map": (
            schedule.fuse_softmax_slim_chebyshev_map
        ),
        "softmax_bootstrap_iterations": schedule.softmax_bootstrap_iterations,
        "softmax_main_bootstrap_iterations": (
            schedule.softmax_main_bootstrap_iterations
        ),
        "softmax_lambda_bootstrap_iterations": (
            schedule.softmax_lambda_bootstrap_iterations
        ),
        "softmax_single_pass_match_two_pass_level": (
            schedule.match_softmax_single_pass_bootstrap_level
        ),
        "softmax_bootstrap_precision": schedule.softmax_bootstrap_precision,
        "residual_bootstrap_iterations": (
            schedule.residual_bootstrap_iterations
        ),
        "input_residual_bootstrap_iterations": (
            schedule.input_residual_bootstrap_iterations
        ),
        "post_attention_residual_bootstrap_iterations": (
            schedule.post_attention_residual_bootstrap_iterations
        ),
        "residual_bootstrap_precision": (
            schedule.residual_bootstrap_precision
        ),
        "adaptive_rmsnorm_slim_refresh": (
            schedule.adaptive_rmsnorm_slim_refresh
        ),
        "rmsnorm2_slim_bootstrap_log_slots": (
            schedule.rmsnorm2_slim_log_slots
        ),
        "batched_sparse_attention_input": (
            schedule.batched_sparse_attention_input
        ),
        "fuse_rmsnorm_output_scale": schedule.fuse_rmsnorm_output_scale,
        "fuse_silu_domain_map": schedule.fuse_silu_domain_map,
        "profile_device_memory": schedule.profile_device_memory,
        "profile_operator_breakdown": bool(args.profile_operator_breakdown),
        "warmup": int(args.warmup),
        "warmup_scope": settings.warmup_scope,
    }


def audit_payload(
    args,
    settings: RunSettings,
    schedule: LayerSchedule,
    *,
    rotation_key_count: int,
    canonical_cipher_count: int,
    degree_summaries: dict[str, dict[str, object]],
    audit: dict[str, object],
) -> dict[str, object]:
    """The complete ``--audit-only`` report: structure, schedule, no GPU."""

    return {
        "backend": "feature_major",
        "mode": "audit",
        "model_scope": settings.model_scope,
        "num_layers": settings.num_layers,
        "rotation_key_count": int(rotation_key_count),
        "canonical_cipher_count": int(canonical_cipher_count),
        "inter_layer": "bootstrap fused into next input RMSNorm",
        "schedule": schedule_report(args, settings, schedule),
        "polynomial_degrees": degree_summaries,
        "audit": audit,
    }


__all__ = ["audit_payload", "schedule_report"]
