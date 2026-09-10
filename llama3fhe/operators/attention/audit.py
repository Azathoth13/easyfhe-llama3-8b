from __future__ import annotations

"""Static cost and rotation-key audit for feature-major QKV preparation.

This module is intentionally execution-free.  Runtime orchestration belongs
to :mod:`linear.qkv`, :mod:`attention.input`, and :mod:`attention.operator`;
the audit merely composes their public cost models.
"""

from dataclasses import asdict, dataclass

from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.layouts.linear import LinearCarrierLayout

from ..linear.config import LLAMA3_LINEAR_SCHEDULES, LinearOperatorConfig
from ..linear.qkv import (
    qkv_rotations,
    token_pair_complexity,
)
from ..linear.qkv_kernel import complex_token_diagonal_qkv_complexity
from .input import (
    complex_token_attention_complexity,
    complex_token_attention_rotations,
)
from .softmax import delta_softmax_row_rotations


def delta_attention_rotations(
    layout: AttentionPairLayout,
    *,
    include_qk_feature_fold: bool = False,
    qk_key_features_pre_folded: bool = False,
    qk_query_features_pre_folded: bool = False,
    include_pv_delta_fold: bool = False,
    pv_probability_pre_folded: bool = False,
    pv_value_pre_folded: bool = False,
    log_n: int = 16,
) -> tuple[int, ...]:
    """Rotation-key set for direct Delta QK and PV."""

    seq_len = int(layout.seq_len)
    head_dim = int(layout.head_dim)
    qk_folded = bool(include_qk_feature_fold)
    key_pre_folded = bool(qk_key_features_pre_folded)
    query_pre_folded = bool(qk_query_features_pre_folded)
    pv_folded = bool(include_pv_delta_fold)
    probability_pre_folded = bool(pv_probability_pre_folded)
    value_pre_folded = bool(pv_value_pre_folded)
    if key_pre_folded and not qk_folded:
        raise ValueError("pre-folded K requires complex QK feature folding.")
    if query_pre_folded and not qk_folded:
        raise ValueError("pre-folded Q requires complex QK feature folding.")
    if (probability_pre_folded or value_pre_folded) and not pv_folded:
        raise ValueError("pre-folded P/V requires complex PV delta folding.")
    if (qk_folded and head_dim % 2) or (pv_folded and seq_len % 2):
        raise ValueError("complex QK/PV folding requires even reduction sizes.")

    rotations: set[int] = set()
    query_shifts = set(
        range(1, head_dim // 2 if qk_folded else head_dim)
    )
    probability_shifts = set(
        range(1, seq_len // 2 if pv_folded else seq_len)
    )
    if qk_folded and not query_pre_folded:
        query_shifts.add(head_dim // 2)
    if pv_folded and not probability_pre_folded:
        probability_shifts.add(seq_len // 2)
    for shift in query_shifts | probability_shifts:
        rotations.add(2 * seq_len * shift)  # Q/P feature-block shift

    key_shifts = range(
        1,
        head_dim // 2 if qk_folded and key_pre_folded else head_dim,
    )
    for shift in key_shifts:
        rotations.add(2 * shift * (seq_len - 1))  # K torus, no q wrap
        rotations.add(2 * (shift * (seq_len - 1) + seq_len))  # K torus, q wrap

    value_shifts = set(range(1, seq_len // 2 if pv_folded else seq_len))
    if pv_folded and not value_pre_folded:
        value_shifts.add(seq_len // 2)
    for shift in value_shifts:
        rotations.add(2 * shift)  # V q shift, no wrap
        rotations.add(-2 * (seq_len - shift))  # V q wrap

    if qk_folded or pv_folded:
        rotations.add((1 << (int(log_n) + 1)) - 1)  # CKKS conjugation
    return tuple(sorted(rotation for rotation in rotations if rotation))


def delta_qk_operation_counts(
    layout: AttentionPairLayout,
    *,
    complex_feature_folding: bool = False,
    key_features_pre_folded: bool = False,
    query_features_pre_scaled: bool = False,
    query_features_pre_folded: bool = False,
) -> dict[str, int]:
    """Static counts for the selected lazy-triplet Delta QK kernel."""

    queries = int(layout.query_pair_count)
    keys = int(layout.key_value_pair_count)
    dimensions = int(layout.head_dim)
    folded = bool(complex_feature_folding)
    key_pre_folded = bool(key_features_pre_folded)
    query_pre_scaled = bool(query_features_pre_scaled)
    query_pre_folded = bool(query_features_pre_folded)
    if key_pre_folded and not folded:
        raise ValueError("pre-folded K requires complex QK feature folding.")
    if query_pre_scaled and not folded:
        raise ValueError("pre-scaled Q requires complex QK feature folding.")
    if query_pre_folded and not folded:
        raise ValueError("pre-folded Q requires complex QK feature folding.")
    shifts = dimensions // 2 if folded else dimensions
    if folded and dimensions % 2:
        raise ValueError("complex QK feature folding requires even head_dim.")
    counts = {
        "ct_ct_multiplications": queries * shifts,
        "relinearizations": queries,
        "ct_ct_final_rescales": queries,
        "precomputed_input_alignments": queries + keys,
        "pt_ct_multiplications": 2 * keys * (shifts - 1),
        "rotations": (shifts - 1) * (queries + 2 * keys),
        "static_mask_plaintexts": 2 * (shifts - 1),
        "ciphertext_additions": queries * (shifts - 1)
        + keys * (shifts - 1),
        "multiplicative_depth_from_qk_inputs": 2,
    }
    if folded:
        # Scale/fold Q one level above the target.  K's two feature halves use
        # the normal masked torus shifts at the same target, so the fold adds no
        # multiplicative depth.  Extracting the real score uses a conjugation.
        counts["precomputed_input_alignments"] += keys
        counts["pt_ct_multiplications"] = queries + 2 * keys * (dimensions - 1)
        counts["rotations"] = (
            queries
            + queries * (shifts - 1)
            + 2 * keys * (dimensions - 1)
            + queries
        )
        counts["static_mask_plaintexts"] = 1 + 2 * (dimensions - 1)
        counts["ciphertext_additions"] = (
            queries
            + keys * (dimensions - 1)
            + keys * shifts
            + queries * (shifts - 1)
            + queries
        )
        if key_pre_folded:
            # K_C folding is fused before the existing shear. QK itself now
            # shifts only 64 already-folded K diagonals instead of separately
            # constructing both 64-feature halves.
            counts["precomputed_input_alignments"] = queries + 2 * keys
            counts["pt_ct_multiplications"] = queries + 2 * keys * (shifts - 1)
            counts["rotations"] = (
                queries
                + queries * (shifts - 1)
                + 2 * keys * (shifts - 1)
                + queries
            )
            counts["static_mask_plaintexts"] = 1 + 2 * (shifts - 1)
            counts["ciphertext_additions"] = (
                queries
                + keys * (shifts - 1)
                + queries * (shifts - 1)
                + queries
            )
        if query_pre_scaled:
            counts["pt_ct_multiplications"] -= queries
            counts["static_mask_plaintexts"] -= 1
        if query_pre_folded:
            counts["rotations"] -= queries
            counts["ciphertext_additions"] -= queries
    return counts


def delta_pv_operation_counts(
    layout: AttentionPairLayout,
    *,
    complex_delta_folding: bool = False,
    probability_pre_folded: bool = False,
    value_pre_folded: bool = False,
) -> dict[str, int]:
    """Static counts for the selected lazy-triplet Delta PV kernel."""

    queries = int(layout.query_pair_count)
    values = int(layout.key_value_pair_count)
    deltas = int(layout.seq_len)
    folded = bool(complex_delta_folding)
    probability_pre_folded = bool(probability_pre_folded)
    value_pre_folded = bool(value_pre_folded)
    if (probability_pre_folded or value_pre_folded) and not folded:
        raise ValueError("pre-folded P/V requires complex PV delta folding.")
    if folded and deltas % 2:
        raise ValueError("complex PV delta folding requires even seq_len.")
    shifts = deltas // 2 if folded else deltas
    counts = {
        "ct_ct_multiplications": queries * shifts,
        "relinearizations": queries,
        "ct_ct_final_rescales": queries,
        "precomputed_input_alignments": queries + 2 * values,
        "pt_ct_multiplications": 2 * values * (deltas - 1),
        "rotations": (deltas - 1) * (queries + 2 * values),
        "static_mask_plaintexts": 2 * (deltas - 1),
        "ciphertext_additions": queries * (deltas - 1)
        + values * (deltas - 1),
        "multiplicative_depth_from_pv_inputs": 2,
    }
    if folded:
        # The critical nonzero-shift path is deeper than the s=0 path:
        # optional V half-fold -> masked torus shift -> final CT-CT rescale.
        # The probability half-fold is parallel and therefore does not add
        # another critical level.
        counts["multiplicative_depth_from_pv_inputs"] = 3
        # One P scaling and one V half-row construction prepare the two
        # complex operands.  CKKS imults/conjugations are counted as rotations.
        counts["pt_ct_multiplications"] = queries + 2 * values * shifts
        counts["rotations"] = (
            queries  # P half-row rotate
            + queries  # P imult
            + queries * (shifts - 1)
            + 2 * values  # V half-row masked shift
            + values  # V imult
            + 2 * values * (shifts - 1)
            + queries  # final conjugation
        )
        counts["static_mask_plaintexts"] = 1 + 2 * shifts
        counts["ciphertext_additions"] = (
            queries  # P fold
            + values  # V half-row masked shift
            + values  # V fold
            + queries * (shifts - 1)
            + values * (shifts - 1)
            + queries  # final conjugate-add
        )
        if probability_pre_folded:
            counts["pt_ct_multiplications"] -= queries
            counts["rotations"] -= 2 * queries
            counts["static_mask_plaintexts"] -= 1
            counts["ciphertext_additions"] -= queries
        if value_pre_folded:
            counts["precomputed_input_alignments"] -= values
            counts["pt_ct_multiplications"] -= 2 * values
            counts["rotations"] -= 3 * values
            counts["static_mask_plaintexts"] -= 2
            counts["ciphertext_additions"] -= 2 * values
        if value_pre_folded:
            # Pre-folding V removes the first of the three critical levels;
            # the masked nonzero shift and CT-CT rescale remain.
            counts["multiplicative_depth_from_pv_inputs"] = 2
    return counts

@dataclass(frozen=True)
class FeatureMajorPipelineStaticAudit:
    canonical_input_cipher_count: int
    projection_input_cipher_count: int
    carrier_cipher_count: int
    query_cipher_count: int
    key_cipher_count: int
    value_cipher_count: int
    score_cipher_count: int
    rotation_key_count: int
    encoded_static_plaintext_rows: int
    pt_ct_multiplications: int
    ct_ct_multiplications: int
    rotation_operations_including_conjugations: int
    query_multiplicative_depth: int
    key_multiplicative_depth: int
    value_multiplicative_depth: int
    score_multiplicative_depth: int
    canonical_input_is_real: bool
    requires_input_layout_conversion: bool
    stages: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _carrier_layout(layout: FeatureMajorPrefillLayout) -> LinearCarrierLayout:
    return LinearCarrierLayout(
        seq_len=int(layout.seq_len),
        dimension=int(layout.hidden_dim),
        slots=int(layout.slots),
    )


def _carrier_width(layout: AttentionPairLayout) -> int:
    return int(layout.head_dim) * (
        int(layout.query_pair_count) + 2 * int(layout.key_value_pair_count)
    )


def _ring_log_n(slots: int) -> int:
    slots = int(slots)
    if slots <= 0 or slots & (slots - 1):
        raise ValueError(f"slots must be a positive power of two, got {slots}.")
    return slots.bit_length()


def _validate_layouts(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> None:
    if int(feature_layout.seq_len) != int(attention_layout.seq_len):
        raise ValueError("feature-major and attention sequence lengths must match.")
    if int(feature_layout.slots) != int(attention_layout.slots):
        raise ValueError("feature-major and attention slot counts must match.")


def pipeline_rotations(
    feature_major_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    qkv_operator_config: LinearOperatorConfig | None = None,
    sparse_max_baby_offsets: int = 32,
    shear_baby_steps: int = 16,
    include_scores: bool = True,
    extra_rotations: tuple[int, ...] | list[int] = (),
) -> tuple[int, ...]:
    """Return the exact rotation-key union for FM QKV and its adapter."""

    _validate_layouts(feature_major_layout, attention_layout)
    log_n = _ring_log_n(int(feature_major_layout.slots))
    qkv_operator_config = (
        LLAMA3_LINEAR_SCHEDULES.qkv
        if qkv_operator_config is None
        else qkv_operator_config
    )
    rotations = set(
        qkv_rotations(
            feature_major_layout, operator_config=qkv_operator_config
        )
    )
    rotations.update(
        complex_token_attention_rotations(
            _carrier_layout(feature_major_layout),
            attention_layout,
            sparse_max_baby_offsets=int(sparse_max_baby_offsets),
            shear_baby_steps=int(shear_baby_steps),
            log_n=log_n,
        )
    )
    if include_scores:
        rotations.update(
            delta_attention_rotations(
                attention_layout,
                include_qk_feature_fold=True,
                qk_key_features_pre_folded=True,
                qk_query_features_pre_folded=True,
                log_n=log_n,
            )
        )
    rotations.update(int(value) for value in extra_rotations if int(value))
    return tuple(sorted(int(value) for value in rotations if int(value)))


def pipeline_static_audit(
    feature_major_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    qkv_operator_config: LinearOperatorConfig | None = None,
    sparse_max_baby_offsets: int = 32,
    shear_baby_steps: int = 16,
    include_scores: bool = True,
) -> FeatureMajorPipelineStaticAudit:
    """Compose the public QKV, adapter, and optional QK cost models."""
    qkv_operator_config = (
        LLAMA3_LINEAR_SCHEDULES.qkv
        if qkv_operator_config is None
        else qkv_operator_config
    )

    _validate_layouts(feature_major_layout, attention_layout)
    pair = token_pair_complexity(feature_major_layout)
    qkv = complex_token_diagonal_qkv_complexity(
        seq_len=int(feature_major_layout.seq_len),
        dimension=int(feature_major_layout.hidden_dim),
        slots=int(feature_major_layout.slots),
        carrier_width=_carrier_width(attention_layout),
        baby_steps=int(qkv_operator_config.baby_steps),
        baby_anchor_step=int(qkv_operator_config.baby_anchor_step),
        max_plaintext_rows=int(qkv_operator_config.max_plaintext_rows),
        reuse_baby_rotations=bool(
            qkv_operator_config.reuse_baby_rotations
        ),
    )
    adapter = complex_token_attention_complexity(
        _carrier_layout(feature_major_layout),
        attention_layout,
        sparse_max_baby_offsets=int(sparse_max_baby_offsets),
        shear_baby_steps=int(shear_baby_steps),
    )
    qk = (
        delta_qk_operation_counts(
            attention_layout,
            complex_feature_folding=True,
            key_features_pre_folded=True,
            query_features_pre_scaled=True,
            query_features_pre_folded=True,
        )
        if include_scores
        else {}
    )
    rotations = pipeline_rotations(
        feature_major_layout,
        attention_layout,
        qkv_operator_config=qkv_operator_config,
        sparse_max_baby_offsets=int(sparse_max_baby_offsets),
        shear_baby_steps=int(shear_baby_steps),
        include_scores=bool(include_scores),
    )
    stages = {
        "real_feature_major_pair": pair.to_dict(),
        "heterogeneous_qkv": qkv.to_dict(),
        "carrier_to_attention": adapter.to_dict(),
    }
    if include_scores:
        stages["delta_qk"] = {str(key): int(value) for key, value in qk.items()}
    qk_depth = int(qk.get("multiplicative_depth_from_qk_inputs", 0))
    return FeatureMajorPipelineStaticAudit(
        canonical_input_cipher_count=int(feature_major_layout.cipher_count),
        projection_input_cipher_count=int(pair.output_cipher_count),
        carrier_cipher_count=int(qkv.output_cipher_count),
        query_cipher_count=int(attention_layout.query_pair_count),
        key_cipher_count=int(attention_layout.key_value_pair_count),
        value_cipher_count=int(attention_layout.key_value_pair_count),
        score_cipher_count=int(attention_layout.query_pair_count) if include_scores else 0,
        rotation_key_count=len(rotations),
        encoded_static_plaintext_rows=(
            int(qkv.encoded_weight_plaintexts)
            + int(adapter.encoded_static_plaintext_rows)
            + int(qk.get("static_mask_plaintexts", 0))
        ),
        pt_ct_multiplications=(
            int(qkv.pt_ct_multiplications)
            + int(adapter.pt_ct_multiplications)
            + int(qk.get("pt_ct_multiplications", 0))
        ),
        ct_ct_multiplications=int(qk.get("ct_ct_multiplications", 0)),
        rotation_operations_including_conjugations=(
            int(qkv.rotations_including_conjugations)
            + int(adapter.rotations)
            + int(adapter.conjugations)
            + int(qk.get("rotations", 0))
        ),
        query_multiplicative_depth=(
            int(qkv.multiplicative_depth) + int(adapter.query_multiplicative_depth)
        ),
        key_multiplicative_depth=(
            int(qkv.multiplicative_depth) + int(adapter.key_multiplicative_depth)
        ),
        value_multiplicative_depth=(
            int(qkv.multiplicative_depth) + int(adapter.value_multiplicative_depth)
        ),
        score_multiplicative_depth=(
            int(qkv.multiplicative_depth)
            + int(adapter.key_multiplicative_depth)
            + qk_depth
            if include_scores
            else 0
        ),
        canonical_input_is_real=True,
        requires_input_layout_conversion=False,
        stages=stages,
    )


__all__ = [
    "FeatureMajorPipelineStaticAudit",
    "pipeline_rotations",
    "pipeline_static_audit",
]


@dataclass(frozen=True)
class DeltaSoftmaxLayoutComplexity:
    row_reduce_rotations: int
    row_reduce_additions: int
    slim_pack_pt_ct_multiplications: int
    slim_pack_additions: int
    slim_broadcast_pt_ct_multiplications: int
    slim_broadcast_rotations: int
    slim_broadcast_additions: int
    slim_polynomial_evaluations: int = 1

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class DeltaSoftmaxAlg2Complexity:
    main_cipher_count: int
    rounds: int
    explicit_ct_ct_multiplications: int
    explicit_relinearizations: int
    explicit_ct_ct_rescales: int
    pt_ct_multiplications: int
    rotations: int
    ct_additions: int
    lambda_bootstraps: int
    exp_polynomial_evaluations: int
    exp_polynomial_degree: int
    rough_invsqrt_evaluations: int
    rough_invsqrt_degree: int
    precise_invsqrt_evaluations: int
    precise_invsqrt_degree: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


def delta_softmax_alg2_complexity(
    layout: AttentionPairLayout,
    *,
    rounds: int = 4,
    exp_degree: int = 17,
    rough_invsqrt_degree: int = 192,
    precise_invsqrt_degree: int = 96,
    reuse_round_square: bool = False,
    direct_slim_inactive_fill: bool = False,
    fuse_slim_chebyshev_map: bool = False,
) -> DeltaSoftmaxAlg2Complexity:
    """Static non-polynomial operation audit for Delta-native Layer0 Alg2."""

    pairs = int(layout.query_pair_count)
    rounds = int(rounds)
    plumbing = delta_softmax_layout_complexity(layout)
    explicit_ct_ct_per_round = (
        2 * pairs + 1 if bool(reuse_round_square) else 3 * pairs
    )
    # Per round: slim-pack, optional inactive-slot remask, and slim broadcast.
    # Slim pack already writes only the active query-pair rows, so the direct
    # fill path can add the public inactive polynomial value without masking
    # and rescaling the active rows a second time.  The
    # initial score scale/mask and the single post-exp causal mask add
    # 2*pairs.  Once post-exp invalid slots are zero, the Alg2 update
    # ``y <- (y * lambda)^2`` preserves those zeros, so remasking every round
    # is algebraically redundant.
    pt_ct_per_round = (
        plumbing.slim_pack_pt_ct_multiplications
        + (
            0
            if bool(direct_slim_inactive_fill)
            or bool(fuse_slim_chebyshev_map)
            else 1
        )
        + plumbing.slim_broadcast_pt_ct_multiplications
    )
    rotations_per_round = (
        plumbing.row_reduce_rotations + plumbing.slim_broadcast_rotations
    )
    additions_per_round = (
        plumbing.row_reduce_additions
        + plumbing.slim_pack_additions
        + plumbing.slim_broadcast_additions
    )
    return DeltaSoftmaxAlg2Complexity(
        main_cipher_count=pairs,
        rounds=rounds,
        explicit_ct_ct_multiplications=rounds * explicit_ct_ct_per_round,
        explicit_relinearizations=rounds * explicit_ct_ct_per_round,
        explicit_ct_ct_rescales=rounds * explicit_ct_ct_per_round,
        pt_ct_multiplications=2 * pairs + rounds * pt_ct_per_round,
        rotations=rounds * rotations_per_round,
        ct_additions=rounds * additions_per_round,
        lambda_bootstraps=rounds,
        exp_polynomial_evaluations=pairs,
        exp_polynomial_degree=int(exp_degree),
        rough_invsqrt_evaluations=1 if rounds else 0,
        rough_invsqrt_degree=int(rough_invsqrt_degree),
        precise_invsqrt_evaluations=max(0, rounds - 1),
        precise_invsqrt_degree=int(precise_invsqrt_degree),
    )


def delta_softmax_rotations(layout: AttentionPairLayout) -> tuple[int, ...]:
    rotations = set(delta_softmax_row_rotations(layout))
    rotations.update(-rotation for rotation in tuple(rotations))
    stride = 2 * int(layout.seq_len)
    rotations.update(pair * stride for pair in range(1, int(layout.query_pair_count)))
    return tuple(sorted(rotation for rotation in rotations if rotation))


def delta_softmax_layout_complexity(layout: AttentionPairLayout) -> DeltaSoftmaxLayoutComplexity:
    steps = len(delta_softmax_row_rotations(layout))
    pairs = int(layout.query_pair_count)
    return DeltaSoftmaxLayoutComplexity(
        row_reduce_rotations=pairs * steps,
        row_reduce_additions=pairs * steps,
        slim_pack_pt_ct_multiplications=pairs,
        slim_pack_additions=pairs - 1,
        slim_broadcast_pt_ct_multiplications=pairs,
        slim_broadcast_rotations=(pairs - 1) + pairs * steps,
        slim_broadcast_additions=pairs * steps,
    )
