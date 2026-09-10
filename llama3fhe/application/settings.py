"""Resolving CLI arguments into the decisions a run is actually made of.

The parser accepts what a user may write; this module turns that into what the
run will do, once, before anything expensive happens. Two things motivate it:

*Validation belongs before setup.* Every value here is range-checked against the
modulus chain and the layer count, so an impossible combination fails in
milliseconds instead of after a minute of key generation.

*A decision should be described once.* These resolved values are what both the
``--audit-only`` report and the full run report have to state. Rendering them
from one object is why ``softmax_bootstrap_position`` is derived in a single
place instead of by the same three-way ternary written out in two report
builders.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import Llama3CKKSConfig
from ..schedule import LayerSchedule


@dataclass(frozen=True)
class RunSettings:
    """Every schedule decision a run makes, resolved and validated."""

    config: Llama3CKKSConfig
    model_scope: str
    num_layers: int
    depth: int
    input_level: int
    input_level_source: str
    bootstrap_post_levels: int
    pairfm_post_refresh_input_limbs: int | None
    auto_softmax_pre_refresh: bool
    warmup_scope: str
    requested_softmax_position: str

    @property
    def softmax_bootstrap_position(self) -> str:
        """The effective Softmax refresh position, with ``auto`` resolved.

        ``auto`` selects "pre" exactly when the layer has an input refresh whose
        budget the pre-exp checkpoint can share — a steady-state layer or any
        multi-layer run. A fresh single layer0 has no such refresh, so ``auto``
        resolves to "off" there.
        """
        if self.requested_softmax_position != "auto":
            return self.requested_softmax_position
        return "pre" if self.auto_softmax_pre_refresh else "off"

    def as_report(self) -> dict[str, object]:
        """The resolved schedule, in the shape both reports embed."""
        return {
            "context_depth": self.depth,
            "input_level": self.input_level,
            "input_level_source": self.input_level_source,
            "bootstrap_post_levels": self.bootstrap_post_levels,
            "softmax_bootstrap_position": self.softmax_bootstrap_position,
        }


def resolve_settings(args, config: Llama3CKKSConfig) -> RunSettings:
    """Validate ``args`` against ``config`` and return the settled decisions.

    ``config`` must already carry any ``--depth`` / ``--device`` override, since
    the modulus chain it describes is what every bound here is checked against.
    """

    num_layers = int(args.num_layers)
    if not 0 < num_layers <= 32:
        raise ValueError("num-layers must lie in [1, 32] for Llama-3-8B.")
    if int(args.warmup) < 0:
        raise ValueError("warmup must be non-negative.")
    if int(args.repetitions) <= 0:
        raise ValueError("repetitions must be positive.")
    if int(args.top_k) <= 0:
        raise ValueError("top-k must be positive.")
    if float(args.plaintext_cache_gb) < 0:
        raise ValueError("plaintext-cache-gb must be non-negative.")
    if float(args.plaintext_cache_max_source_mb) < 0:
        raise ValueError("plaintext-cache-max-source-mb must be non-negative.")
    if args.expect_token_id is not None and (
        bool(args.audit_only) or str(args.weights_source) != "real"
    ):
        raise ValueError("expect-token-id requires a real-weight model execution.")
    if str(args.weights_source) == "synthetic" and num_layers != 1:
        raise ValueError(
            "synthetic weights are a one-layer system preset; full-model "
            "execution requires a prepared checkpoint."
        )

    # ``auto`` warms the cheapest representative shape: one steady layer for a
    # multi-layer run, the whole (single-layer) model otherwise.
    warmup_scope = str(args.warmup_scope)
    if warmup_scope == "auto":
        warmup_scope = "steady-layer" if num_layers > 1 else "model"

    depth = int(config.simulator.maxLevelsRemaining)
    # (4,4) Meta-BTS program depth is 18, so the bootstrap returns depth-18
    # levels. Production is depth 37 → 19; --depth 41 rolls back to 23.
    bootstrap_post_levels = (
        int(depth) - 18
        if args.bootstrap_post_levels is None
        else int(args.bootstrap_post_levels)
    )
    if not 0 < bootstrap_post_levels <= depth:
        raise ValueError(
            "bootstrap-post-levels must be positive and no larger than depth."
        )

    if args.input_level is not None:
        input_level = int(args.input_level)
        input_level_source = "explicit"
    elif bool(args.steady_state_layer):
        # The selected schedule leaves two limbs at the persistent boundary:
        # max_limbs=(depth+1), hence L36 in the depth-37 context.  The first
        # operation below is the same bootstrap-fused RMSNorm as layers 1..31.
        input_level = depth - 1
        input_level_source = "steady_state_boundary"
    elif num_layers > 1:
        # Full-model input enters with the same limb budget produced by the
        # configured bootstrap.  Starting the first layer at L0 needlessly
        # carries all 38 limbs through large linears and exceeds A100 memory.
        input_level = depth + 1 - bootstrap_post_levels
        input_level_source = "bootstrap_output_budget"
    else:
        input_level = 0
        input_level_source = "single_layer_fresh"
    if not 0 <= input_level < depth:
        raise ValueError("input-level must lie inside the modulus chain.")

    raw_pairfm_limbs = args.pairfm_post_refresh_input_limbs
    if raw_pairfm_limbs is None:
        pairfm_post_refresh_input_limbs: int | None = 6
    elif int(raw_pairfm_limbs) == 0:
        pairfm_post_refresh_input_limbs = None
    else:
        pairfm_post_refresh_input_limbs = int(raw_pairfm_limbs)

    return RunSettings(
        config=config,
        model_scope="layer0" if num_layers == 1 else "full_model",
        num_layers=num_layers,
        depth=depth,
        input_level=input_level,
        input_level_source=input_level_source,
        bootstrap_post_levels=bootstrap_post_levels,
        pairfm_post_refresh_input_limbs=pairfm_post_refresh_input_limbs,
        auto_softmax_pre_refresh=bool(args.steady_state_layer or num_layers > 1),
        warmup_scope=warmup_scope,
        requested_softmax_position=str(args.softmax_bootstrap_position),
    )


def layer_schedule(args, settings: RunSettings) -> LayerSchedule:
    """The model-facing schedule, assembled once from the CLI and the run.

    Every knob the CLI exposes is written here exactly once; knobs the CLI
    does not expose keep the production defaults declared on
    :class:`LayerSchedule` itself.
    """

    position = settings.requested_softmax_position
    return LayerSchedule(
        steady_state_attention_refresh=settings.auto_softmax_pre_refresh,
        refresh_softmax_exp_inputs=(
            None if position == "auto" else position == "pre"
        ),
        refresh_softmax_exp_outputs=(
            False if position == "auto" else position == "post"
        ),
        preserve_prebootstrap_residual=bool(
            args.preserve_prebootstrap_residual
        ),
        softmax_main_deep_bootstrap=bool(args.softmax_deep_main_bootstrap),
        pair_softmax_pre_exp_bootstrap=bool(
            args.softmax_complex_pre_exp_bootstrap
        ),
        reuse_softmax_round_square=bool(args.softmax_reuse_round_square),
        direct_softmax_slim_inactive_fill=bool(
            args.softmax_direct_slim_inactive_fill
        ),
        fuse_softmax_slim_chebyshev_map=bool(
            args.softmax_fuse_slim_chebyshev_map
        ),
        softmax_main_bootstrap_iterations=(
            args.softmax_main_bootstrap_iterations
        ),
        softmax_lambda_bootstrap_iterations=(
            args.softmax_lambda_bootstrap_iterations
        ),
        match_softmax_single_pass_bootstrap_level=bool(
            args.softmax_single_pass_match_two_pass_level
        ),
        softmax_bootstrap_iterations=int(args.softmax_bootstrap_iterations),
        softmax_bootstrap_precision=int(args.softmax_bootstrap_precision),
        bootstrap_post_levels=settings.bootstrap_post_levels,
        bootstrap_level_budget=tuple(
            int(value) for value in settings.config.simulator.level_budget
        ),
        residual_bootstrap_level_drop=int(args.residual_bootstrap_level_drop),
        residual_bootstrap_iterations=int(args.residual_bootstrap_iterations),
        input_residual_bootstrap_iterations=(
            args.input_residual_bootstrap_iterations
        ),
        post_attention_residual_bootstrap_iterations=(
            args.post_attention_residual_bootstrap_iterations
        ),
        residual_bootstrap_precision=int(args.residual_bootstrap_precision),
        adaptive_rmsnorm_slim_refresh=bool(args.adaptive_rmsnorm_slim_refresh),
        rmsnorm2_slim_log_slots=(
            None
            if args.rmsnorm2_slim_bootstrap_log_slots is None
            else int(args.rmsnorm2_slim_bootstrap_log_slots)
        ),
        fuse_rmsnorm_output_scale=bool(args.fuse_rmsnorm_output_scale),
        fuse_silu_domain_map=bool(args.fuse_silu_domain_map),
        batched_sparse_attention_input=bool(
            args.batched_sparse_attention_input
        ),
        attention_rotation_chunk_size=int(args.attention_rotation_chunk_size),
        pairfm_bootstrap_mode=str(args.pairfm_bootstrap_mode),
        pairfm_post_refresh_input_limbs=(
            settings.pairfm_post_refresh_input_limbs
        ),
        qkv_hoist_strategy=str(args.linear_hoist_strategy),
        output_hoist_strategy=str(args.linear_hoist_strategy),
        mlp_hoist_strategy=str(args.linear_hoist_strategy),
        profile_device_memory=bool(args.profile_device_memory),
    )


__all__ = ["RunSettings", "layer_schedule", "resolve_settings"]
