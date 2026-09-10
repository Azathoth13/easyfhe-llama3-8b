from __future__ import annotations

"""Execution policy for the QKV-carrier to W_O-input attention boundary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionOperatorConfig:
    """Attention-owned schedule choices; layouts remain fixed contracts."""

    sparse_max_baby_offsets: int = 32
    batched_sparse_input_bsgs: bool = False
    sparse_input_batch_cap: int = 32
    sparse_input_fallback: bool = True
    shear_baby_steps: int = 16
    # Modulus levels the Q/K score stream sheds at its earliest split from
    # the value stream (qk_axis1).  The attention entry budget is sized for
    # the value stream's PairFM threshold; the score stream is reset by the
    # pre-exp softmax bootstrap and only needs the tight schedule, so the
    # extra entry limbs make its adapter/QK proportionally more expensive.
    qk_score_level_drop: int = 0
    rotation_mode: str = "fast"
    rotation_chunk_size: int = 16
    qk_output_level: int | None = 13
    qk_lazy_relinearization: bool = True
    pv_lazy_relinearization: bool = True
    qk_complex_feature_folding: bool = True
    qk_key_features_pre_folded: bool = True
    qk_query_features_pre_scaled: bool = True
    qk_query_features_pre_folded: bool = True
    pv_complex_delta_folding: bool = True
    pv_prefold_after_softmax: bool = False
    refresh_softmax_exp_inputs: bool = False
    refresh_softmax_exp_outputs: bool = False
    softmax_main_deep_bootstrap: bool = False
    pair_softmax_pre_exp_bootstrap: bool = False
    reuse_softmax_round_square: bool = True
    direct_softmax_slim_inactive_fill: bool = False
    fuse_softmax_slim_chebyshev_map: bool = True
    softmax_main_bootstrap_iterations: int | None = None
    softmax_lambda_bootstrap_iterations: int | None = 1
    match_softmax_single_pass_bootstrap_level: bool = True
    refresh_output_before_pairfm: bool = False
    pairfm_refresh_min_input_limbs: int | None = None
    pairfm_post_refresh_input_limbs: int | None = 6
    checkpoint_bootstrap_precision_bits: int = 14
    # Native Meta-BTS passes for the PairFM paired-input refresh. Two is what
    # this site inherited implicitly from BootstrapConfig.iterations before
    # the value was stated here; every other attention checkpoint runs one
    # pass (the validated 2026-08 default). The refresh only fires where the
    # PV output falls below the pairing threshold — in the release schedule
    # that is layer 31 alone (k=5), measured at 2.155 s, about half of it the
    # second pass. Lowering this to one has NOT been accuracy-validated at
    # the layer feeding the final logits.
    pairfm_refresh_bootstrap_iterations: int = 2
    pairfm_first_baby_steps: int = 16
    pairfm_block_baby_steps: int = 32
    rope_theta: float = 500000.0
    rope_start_pos: int = 0

    def __post_init__(self) -> None:
        if int(self.sparse_max_baby_offsets) <= 0:
            raise ValueError("sparse_max_baby_offsets must be positive.")
        if not 1 <= int(self.sparse_input_batch_cap) <= 32:
            raise ValueError("sparse_input_batch_cap must be in [1, 32].")
        if int(self.shear_baby_steps) <= 0:
            raise ValueError("shear_baby_steps must be positive.")
        if int(self.qk_score_level_drop) < 0:
            raise ValueError("qk_score_level_drop must be non-negative.")
        if int(self.rotation_chunk_size) <= 0:
            raise ValueError("rotation_chunk_size must be positive.")
        if int(self.pairfm_first_baby_steps) <= 0:
            raise ValueError("pairfm_first_baby_steps must be positive.")
        if int(self.pairfm_block_baby_steps) <= 0:
            raise ValueError("pairfm_block_baby_steps must be positive.")
        if float(self.rope_theta) <= 0:
            raise ValueError("rope_theta must be positive.")
        if int(self.pairfm_refresh_bootstrap_iterations) not in (1, 2):
            raise ValueError(
                "pairfm_refresh_bootstrap_iterations must be one or two."
            )
        if int(self.checkpoint_bootstrap_precision_bits) <= 0:
            raise ValueError(
                "checkpoint_bootstrap_precision_bits must be positive."
            )
        if bool(self.pv_prefold_after_softmax) and not bool(
            self.pv_complex_delta_folding
        ):
            raise ValueError(
                "pv_prefold_after_softmax requires pv_complex_delta_folding."
            )
        if bool(self.refresh_softmax_exp_inputs) and bool(
            self.refresh_softmax_exp_outputs
        ):
            raise ValueError(
                "Softmax bootstrap cannot be enabled on both sides of exp."
            )
        if bool(self.pair_softmax_pre_exp_bootstrap) and not bool(
            self.refresh_softmax_exp_inputs
        ):
            raise ValueError(
                "paired Softmax checkpoint requires a pre-exp bootstrap."
            )
        for name, value in (
            ("softmax_main_bootstrap_iterations", self.softmax_main_bootstrap_iterations),
            ("softmax_lambda_bootstrap_iterations", self.softmax_lambda_bootstrap_iterations),
        ):
            if value is not None and int(value) not in (1, 2):
                raise ValueError(f"{name} must be one or two when set.")
        if (
            self.pairfm_refresh_min_input_limbs is not None
            and int(self.pairfm_refresh_min_input_limbs) <= 1
        ):
            raise ValueError(
                "pairfm_refresh_min_input_limbs must exceed one."
            )
        if (
            self.pairfm_post_refresh_input_limbs is not None
            and int(self.pairfm_post_refresh_input_limbs) <= 1
        ):
            raise ValueError(
                "pairfm_post_refresh_input_limbs must exceed one."
            )
