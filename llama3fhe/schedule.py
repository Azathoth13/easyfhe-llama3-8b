"""The layer schedule: every knob a transformer layer's evaluation obeys.

One frozen object carries the complete schedule from the application to the
layer graph. Before it existed, the same ~45 decisions travelled as keyword
arguments through four signatures (``runner`` → ``Llama3Model.forward`` →
``TransformerLayer.forward`` → ``graph.evaluate``), and the two static audits
(:func:`graph.operation_audit`, :func:`graph.required_rotations`) re-declared
overlapping subsets with their own defaults. Now each consumer binds the
fields it reads, and a default constructed ``LayerSchedule()`` *is* the
production schedule.

Data stays out: ciphertexts, weights, layouts, the crypto context, and
per-layer values (``layer_idx``, ``refresh_input``) remain ordinary
parameters. This object holds only decisions that are constant across the
layers of one run.

The defaults are the production schedule, so a bare ``LayerSchedule()`` runs
the release configuration. The one field the CLI resolves rather than
defaults is ``steady_state_attention_refresh``: it depends on the layer count,
because layer zero has no steady-state pre-exp checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from .operators.linear import LLAMA3_LINEAR_SCHEDULES


@dataclass(frozen=True)
class LayerSchedule:
    """All run-constant evaluation decisions, with production defaults."""

    # -- refresh placement -------------------------------------------------
    #: Refresh QK scores before Softmax the way steady-state layers do.
    steady_state_attention_refresh: bool = False
    #: Tri-state: None lets the layer graph decide from the refresh context.
    refresh_softmax_exp_inputs: bool | None = None
    refresh_softmax_exp_outputs: bool = False
    #: Preserve the residual captured before the fused bootstrap.
    #: Borrow the pre-bootstrap residual instead of materializing a
    #: refreshed copy used by the release schedule.
    preserve_prebootstrap_residual: bool = True
    pair_residual_bootstrap: bool = True

    # -- softmax -----------------------------------------------------------
    softmax_main_deep_bootstrap: bool = False
    pair_softmax_pre_exp_bootstrap: bool = True
    reuse_softmax_round_square: bool = True
    direct_softmax_slim_inactive_fill: bool = False
    fuse_softmax_slim_chebyshev_map: bool = True
    #: One native pass for the main Softmax checkpoint; None would defer
    #: to the bootstrap program's own two.
    softmax_main_bootstrap_iterations: int | None = 1
    softmax_lambda_bootstrap_iterations: int | None = 1
    match_softmax_single_pass_bootstrap_level: bool = True
    softmax_bootstrap_iterations: int = 2
    softmax_bootstrap_precision: int = 8

    # -- bootstrap budgets -------------------------------------------------
    bootstrap_post_levels: int = 19
    bootstrap_log_slots: int = 15
    bootstrap_level_budget: tuple[int, int] = (4, 4)
    residual_bootstrap_level_drop: int = 0
    residual_bootstrap_iterations: int = 1
    input_residual_bootstrap_iterations: int | None = None
    post_attention_residual_bootstrap_iterations: int | None = None
    residual_bootstrap_precision: int = 14
    #: Sparse log_slots for the RMSN2 rsqrt slim refresh only. None keeps
    #: the full-slot program. RMSN1 recovery stays on the main program.
    rmsnorm2_slim_log_slots: int | None = None

    # -- rmsnorm / mlp fusions ---------------------------------------------
    adaptive_rmsnorm_slim_refresh: bool = True
    fuse_rmsnorm_output_scale: bool = True
    fuse_silu_domain_map: bool = True
    mlp_input_level_drop: int = 0

    # -- attention ---------------------------------------------------------
    batched_sparse_attention_input: bool = True
    #: Absolute Layer0 alignment target; a multi-layer run must clear it to
    #: None because a bootstrap has raised the modulus chain in between.
    qk_output_level: int | None = 17
    attention_rotation_chunk_size: int = 8
    pv_complex_delta_folding: bool = True
    pv_prefold_after_softmax: bool = False
    pairfm_bootstrap_mode: str = "auto"
    pairfm_post_refresh_input_limbs: int | None = 6

    # -- linear kernels ----------------------------------------------------
    linear_baby_steps: int = LLAMA3_LINEAR_SCHEDULES.qkv.baby_steps
    linear_anchor_step: int = LLAMA3_LINEAR_SCHEDULES.qkv.baby_anchor_step
    qkv_hoist_strategy: str = LLAMA3_LINEAR_SCHEDULES.qkv.hoist_strategy
    output_hoist_strategy: str = LLAMA3_LINEAR_SCHEDULES.w_o.hoist_strategy
    mlp_hoist_strategy: str = LLAMA3_LINEAR_SCHEDULES.gate_up.hoist_strategy
    qkv_max_plaintext_rows: int = LLAMA3_LINEAR_SCHEDULES.qkv.max_plaintext_rows
    output_max_plaintext_rows: int = (
        LLAMA3_LINEAR_SCHEDULES.w_o.max_plaintext_rows
    )
    mlp_max_plaintext_rows: int = (
        LLAMA3_LINEAR_SCHEDULES.gate_up.max_plaintext_rows
    )
    sparse_max_baby_offsets: int = 32
    shear_baby_steps: int = 16

    # -- memory / profiling ------------------------------------------------
    evict_rotation_cuda_cache_before_softmax_bootstrap: bool = False
    evict_rotation_cuda_cache_before_output_projection: bool = False
    evict_rotation_cuda_cache_before_mlp: bool = False
    evict_rotation_cuda_cache_after_layer: bool = False
    profile_device_memory: bool = False


__all__ = ["LayerSchedule"]
