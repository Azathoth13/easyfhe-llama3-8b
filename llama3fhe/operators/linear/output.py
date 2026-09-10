from __future__ import annotations

"""Feature-major output projection.

Attention owns the ``O_C -> PairFM -> feature-major`` layout conversion.
This module begins at the linear boundary: eight complex token-pair carriers
whose feature channels are in query-pair order, and ends at the sixteen real
canonical feature-major shards consumed by the residual path.
"""

import time

import numpy as np

from llama3fhe.backend import release_all, synchronize_device
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout

from .complex import split_complex_ciphers_twice
from .config import (
    LLAMA3_LINEAR_SCHEDULES,
    LinearOperatorConfig,
    structured_weight_packing_enabled,
)
from .operator import LinearOperator
from .structured_weight import structured_linear_weight


def pair_interleaved_packed_to_logical(
    head_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    head_dim: int,
) -> np.ndarray:
    """Map physical ``(pair, feature, lane)`` channels to model channels."""

    mapping = np.asarray(
        [
            head * int(head_dim) + feature
            for first, second in head_pairs
            for feature in range(int(head_dim))
            for head in (int(first), int(second))
        ],
        dtype=np.int64,
    )
    if mapping.size and not np.array_equal(
        np.sort(mapping), np.arange(mapping.size, dtype=np.int64)
    ):
        raise ValueError("head pairs must cover every logical head exactly once.")
    return mapping


def _validate_w_o_layouts(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> int:
    if int(feature_layout.slots) != int(attention_layout.slots):
        raise ValueError("feature-major and attention slot counts must match.")
    if int(feature_layout.seq_len) != int(attention_layout.seq_len):
        raise ValueError("feature-major and attention sequence lengths must match.")
    dimension = int(feature_layout.hidden_dim)
    if dimension != int(attention_layout.query_heads) * int(attention_layout.head_dim):
        raise ValueError("feature-major dimension must equal query_heads*head_dim.")
    if int(feature_layout.cipher_count) != int(attention_layout.query_pair_count):
        raise ValueError("W_O requires one feature-major shard per query pair.")
    if int(feature_layout.seq_len) != (
        int(feature_layout.cipher_count) * int(feature_layout.tokens_per_cipher)
    ):
        raise ValueError("W_O requires completely filled feature-major token shards.")
    return int(feature_layout.cipher_count)


def w_o_fhe(
    input_carriers: tuple[object, ...] | list[object],
    weight: np.ndarray,
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    operator_config: LinearOperatorConfig | None = None,
    dtype: np.dtype = np.float64,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    """Apply W_O and return real canonical feature-major shards.

    ``input_carriers[r] = X[2r] + i*X[2r+1]``.  The input feature channels
    are pair-major.  A ``0.5`` and the pair-major-to-logical permutation are
    folded into the plaintext weight, so the final level-free complex split
    produces unscaled real outputs in canonical logical channel order.
    """

    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.w_o
        if operator_config is None
        else operator_config
    )
    # The stage code below reads the policy as plain locals; binding them here
    # keeps one object at the boundary.
    baby_steps = int(operator_config.baby_steps)
    baby_anchor_step = int(operator_config.baby_anchor_step)
    max_plaintext_rows = int(operator_config.max_plaintext_rows)
    reuse_baby_rotations = bool(operator_config.reuse_baby_rotations)
    hoist_strategy = str(operator_config.hoist_strategy)

    carriers = tuple(input_carriers)
    shard_count = _validate_w_o_layouts(feature_layout, attention_layout)
    if len(carriers) != shard_count // 2:
        raise ValueError(f"feature-major W_O expects {shard_count // 2} carriers.")
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in carriers}
    if len(levels) != 1:
        raise ValueError("feature-major W_O inputs must enter at one common level.")
    if int(crypto_context.max_slots) != int(feature_layout.slots):
        raise ValueError("crypto context and feature-major slot counts must match.")

    dimension = int(feature_layout.hidden_dim)
    weight = np.asarray(weight)
    if weight.shape != (dimension, dimension):
        raise ValueError(f"W_O must have shape {(dimension, dimension)}, got {weight.shape}.")
    weight_transform_start = time.perf_counter()
    mapping = pair_interleaved_packed_to_logical(
        attention_layout.query_head_pairs(),
        head_dim=int(attention_layout.head_dim),
    )
    if structured_weight_packing_enabled():
        rows = np.arange(dimension, dtype=np.int64)
        physical_weight = structured_linear_weight(
            (weight,),
            dimension=dimension,
            real_source=np.zeros((dimension,), dtype=np.int64),
            real_row=rows,
            real_scale=0.5,
            column=mapping,
        )
    else:
        physical_weight = np.asarray(0.5 * weight[:, mapping], dtype=dtype)
    if profile is not None:
        profile["weight_transform_cpu"] = float(
            time.perf_counter() - weight_transform_start
        )

    projected: tuple[object, ...] = ()
    outputs: tuple[object, ...] = ()
    projection_profile: dict[str, float] = {}
    projection_operator = LinearOperator(
        crypto_context=crypto_context,
        token_lanes=int(feature_layout.tokens_per_cipher),
        config=LinearOperatorConfig(
            baby_steps=int(baby_steps),
            baby_anchor_step=int(baby_anchor_step),
            max_plaintext_rows=int(max_plaintext_rows),
            reuse_baby_rotations=bool(reuse_baby_rotations),
            hoist_strategy=str(hoist_strategy),
        ),
    )
    try:
        projected, projection_profile = projection_operator.project(
            carriers,
            physical_weight,
            dtype=np.dtype(dtype),
        )
        split_start = time.perf_counter()
        outputs = split_complex_ciphers_twice(
            projected, crypto_context=crypto_context
        )
        synchronize_device(crypto_context.device)
        if profile is not None:
            profile.update(
                {str(key): float(value) for key, value in projection_profile.items()}
            )
            profile["output_split"] = float(time.perf_counter() - split_start)
        return outputs
    except Exception:
        release_all(outputs)
        raise
    finally:
        projection_operator.close()
        release_all(projected)
