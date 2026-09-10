from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from llama3fhe.layouts.attention import AttentionPairLayout


@dataclass(frozen=True)
class DeltaAttentionResult:
    query_rope: np.ndarray
    key_rope: np.ndarray
    score_delta: np.ndarray
    probability_delta: np.ndarray
    output_c: np.ndarray
    scores: np.ndarray
    probabilities: np.ndarray
    output: np.ndarray


def rope_cos_sin(
    *,
    seq_len: int,
    head_dim: int,
    theta: float = 500000.0,
    start_pos: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    seq_len = int(seq_len)
    head_dim = int(head_dim)
    if seq_len <= 0 or head_dim <= 0 or head_dim % 2:
        raise ValueError(f"RoPE requires positive seq_len and even head_dim, got {seq_len}, {head_dim}.")
    inv_freq = 1.0 / (float(theta) ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(int(start_pos), int(start_pos) + seq_len, dtype=np.float32)
    angles = np.outer(positions, inv_freq)
    embedding = np.concatenate([angles, angles], axis=-1)
    return np.cos(embedding).astype(np.float32), np.sin(embedding).astype(np.float32)


def apply_rope_heads(values: np.ndarray, cos: np.ndarray, sin: np.ndarray, *, num_heads: int, head_dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    num_heads = int(num_heads)
    head_dim = int(head_dim)
    expected = (int(values.shape[0]), num_heads * head_dim)
    if values.shape != expected:
        raise ValueError(f"RoPE values must have shape {expected}, got {values.shape}.")
    cos = np.asarray(cos, dtype=np.float32)
    sin = np.asarray(sin, dtype=np.float32)
    if cos.shape != (values.shape[0], head_dim) or sin.shape != cos.shape:
        raise ValueError(f"RoPE cos/sin must have shape {(values.shape[0], head_dim)}, got {cos.shape}, {sin.shape}.")
    heads = values.reshape(values.shape[0], num_heads, head_dim)
    half = head_dim // 2
    first = heads[:, :, :half]
    second = heads[:, :, half:]
    out = np.empty_like(heads, dtype=np.float32)
    out[:, :, :half] = first * cos[:, None, :half] - second * sin[:, None, :half]
    out[:, :, half:] = second * cos[:, None, :half] + first * sin[:, None, :half]
    return out.reshape(values.shape).astype(np.float32, copy=False)


def qk_delta_numpy(query_c: np.ndarray, key_delta: np.ndarray, *, layout: AttentionPairLayout) -> np.ndarray:
    """Evaluate the direct ``C x Delta -> S_Delta`` formula on slot arrays."""

    q_base = layout._base_payload(query_c, pairs=layout.query_pair_count, name="Q_C")
    k_base = layout._base_payload(key_delta, pairs=layout.key_value_pair_count, name="K_Delta")
    q_view = q_base.reshape(layout.query_pair_count, int(layout.head_dim), int(layout.seq_len), 2)
    k_view = k_base.reshape(layout.key_value_pair_count, int(layout.head_dim), int(layout.seq_len), 2)
    score_view = np.zeros_like(q_view, dtype=np.float64)
    ratio = layout.gqa_ratio
    for kv_pair in range(layout.key_value_pair_count):
        k_pair = k_view[kv_pair]
        for offset in range(ratio):
            q_pair_index = kv_pair * ratio + offset
            q_pair = q_view[q_pair_index]
            accumulator = np.zeros_like(q_pair, dtype=np.float64)
            for shift in range(int(layout.head_dim)):
                q_shifted = np.roll(q_pair, shift=-shift, axis=0)
                k_shifted = np.roll(np.roll(k_pair, shift=-shift, axis=0), shift=shift, axis=1)
                accumulator += q_shifted * k_shifted
            score_view[q_pair_index] = accumulator
    return layout._repeat_payload(score_view.reshape(layout.query_pair_count, layout.payload_slots))


def softmax_delta_numpy(
    score_delta: np.ndarray,
    *,
    layout: AttentionPairLayout,
    scale: bool = True,
) -> np.ndarray:
    """Causal row Softmax directly in Delta layout; output remains ``P_Delta``."""

    score_base = layout._base_payload(score_delta, pairs=layout.query_pair_count, name="S_Delta")
    score_view = score_base.reshape(layout.query_pair_count, int(layout.seq_len), int(layout.seq_len), 2)
    values = np.asarray(score_view, dtype=np.float64)
    if scale:
        values = values / math.sqrt(float(layout.head_dim))

    deltas = np.arange(int(layout.seq_len), dtype=np.int64)[:, None]
    queries = np.arange(int(layout.seq_len), dtype=np.int64)[None, :]
    keys = (queries + deltas) % int(layout.seq_len)
    valid = keys <= queries
    masked = np.where(valid[None, :, :, None], values, -np.inf)
    row_max = np.max(masked, axis=1, keepdims=True)
    exponentials = np.where(valid[None, :, :, None], np.exp(masked - row_max), 0.0)
    denominators = np.sum(exponentials, axis=1, keepdims=True)
    probabilities = exponentials / denominators
    return layout._repeat_payload(probabilities.reshape(layout.query_pair_count, layout.payload_slots))


def pv_delta_numpy(probability_delta: np.ndarray, value_delta: np.ndarray, *, layout: AttentionPairLayout) -> np.ndarray:
    """Evaluate the direct ``P_Delta x V_Delta -> O_C`` formula on slot arrays."""

    p_base = layout._base_payload(probability_delta, pairs=layout.query_pair_count, name="P_Delta")
    v_base = layout._base_payload(value_delta, pairs=layout.key_value_pair_count, name="V_Delta")
    p_view = p_base.reshape(layout.query_pair_count, int(layout.head_dim), int(layout.seq_len), 2)
    v_view = v_base.reshape(layout.key_value_pair_count, int(layout.head_dim), int(layout.seq_len), 2)
    output_view = np.zeros_like(p_view, dtype=np.float64)
    ratio = layout.gqa_ratio
    for kv_pair in range(layout.key_value_pair_count):
        v_pair = v_view[kv_pair]
        for offset in range(ratio):
            p_pair_index = kv_pair * ratio + offset
            p_pair = p_view[p_pair_index]
            accumulator = np.zeros_like(p_pair, dtype=np.float64)
            for shift in range(int(layout.head_dim)):
                p_shifted = np.roll(p_pair, shift=-shift, axis=0)
                v_shifted = np.roll(v_pair, shift=-shift, axis=1)
                accumulator += p_shifted * v_shifted
            output_view[p_pair_index] = accumulator
    return layout._repeat_payload(output_view.reshape(layout.query_pair_count, layout.payload_slots))


def attention_gqa_numpy(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conventional causal GQA oracle in standard logical head order."""

    query = layout._logical_heads(query, heads=int(layout.query_heads), name="query")
    key = layout._logical_heads(key, heads=int(layout.key_value_heads), name="key")
    value = layout._logical_heads(value, heads=int(layout.key_value_heads), name="value")
    ratio = layout.gqa_ratio
    key_for_q = key[:, np.arange(int(layout.query_heads)) // ratio, :]
    value_for_q = value[:, np.arange(int(layout.query_heads)) // ratio, :]
    scores = np.einsum("qhd,khd->hqk", query, key_for_q, optimize=True).astype(np.float32)
    scores /= np.float32(math.sqrt(float(layout.head_dim)))
    queries = np.arange(int(layout.seq_len), dtype=np.int64)[:, None]
    keys = np.arange(int(layout.seq_len), dtype=np.int64)[None, :]
    valid = keys <= queries
    masked = np.where(valid[None, :, :], scores, -np.inf)
    row_max = np.max(masked, axis=-1, keepdims=True)
    exponentials = np.where(valid[None, :, :], np.exp(masked - row_max), 0.0)
    probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    output = np.einsum("hqk,khd->qhd", probabilities, value_for_q, optimize=True)
    return (
        scores.astype(np.float32, copy=False),
        probabilities.astype(np.float32, copy=False),
        output.reshape(int(layout.seq_len), -1).astype(np.float32, copy=False),
    )


def run_delta_attention_numpy(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    layout: AttentionPairLayout,
    theta: float = 500000.0,
    start_pos: int = 0,
) -> DeltaAttentionResult:
    cos, sin = rope_cos_sin(
        seq_len=int(layout.seq_len),
        head_dim=int(layout.head_dim),
        theta=float(theta),
        start_pos=int(start_pos),
    )
    query_rope = apply_rope_heads(
        query, cos, sin, num_heads=int(layout.query_heads), head_dim=int(layout.head_dim)
    )
    key_rope = apply_rope_heads(
        key, cos, sin, num_heads=int(layout.key_value_heads), head_dim=int(layout.head_dim)
    )
    payload = layout.pack_qkv(query_rope, key_rope, value)
    score_delta = qk_delta_numpy(payload.query_c, payload.key_delta, layout=layout)
    probability_delta = softmax_delta_numpy(score_delta, layout=layout)
    output_c = pv_delta_numpy(probability_delta, payload.value_delta, layout=layout)
    scores = layout.unpack_score_delta(score_delta) / np.float32(math.sqrt(float(layout.head_dim)))
    probabilities = layout.unpack_score_delta(probability_delta)
    output = layout.unpack_output_c(output_c)
    return DeltaAttentionResult(
        query_rope=query_rope,
        key_rope=key_rope,
        score_delta=score_delta,
        probability_delta=probability_delta,
        output_c=output_c,
        scores=scores,
        probabilities=probabilities,
        output=output,
    )
