from __future__ import annotations

"""Plaintext poly transformer layer used as the FHE diagnostic reference.

This is the mathematics the encrypted layer approximates: polynomial RMSNorm,
exact causal GQA (not Alg2 Softmax), and Chebyshev SiLU.  Diagnostic diffs
are therefore FHE vs this poly graph, not activation amplitude.
"""

from dataclasses import dataclass

import numpy as np

from ..layouts.attention import AttentionPairLayout
from .attention import apply_rope_heads, attention_gqa_numpy, rope_cos_sin
from .chebyshev import eval_chebyshev_series_direct
from .rmsnorm import rmsnorm_poly_numpy


@dataclass(frozen=True)
class PolyLayerResult:
    hidden: np.ndarray
    silu_gate_max_abs: float | None
    silu_oob_frac: float | None
    silu_fit_lo: float
    silu_fit_hi: float
    rmsn1_s_max: float | None = None
    rmsn1_oob_frac: float | None = None
    rmsn1_fit: tuple[float, float] | None = None
    rmsn2_s_max: float | None = None
    rmsn2_oob_frac: float | None = None
    rmsn2_fit: tuple[float, float] | None = None


def _poly_entry_interval(entry: dict) -> tuple[np.ndarray, float, float]:
    coeffs = np.asarray(entry["coefficients"], dtype=np.float64)
    lo, hi = (float(value) for value in entry["fit_interval"])
    return coeffs, lo, hi


def _rmsnorm_poly(values: np.ndarray, entry: dict) -> np.ndarray:
    coeffs, lo, hi = _poly_entry_interval(entry)
    return rmsnorm_poly_numpy(values, coeffs, (lo, hi), skip_clamp=True)


def _silu_poly(values: np.ndarray, entry: dict) -> np.ndarray:
    coeffs, lo, hi = _poly_entry_interval(entry)
    return eval_chebyshev_series_direct(values, coeffs, lo, hi)


def _variance_domain(values: np.ndarray, entry: dict) -> tuple[float | None, float | None, tuple[float, float]]:
    """Token-wise mean(x²) vs the rsqrt Chebyshev interval (FHE does not clip)."""

    s = np.mean(np.square(np.asarray(values, dtype=np.float32)), axis=-1)
    lo, hi = (float(v) for v in entry["fit_interval"])
    finite = np.isfinite(s)
    if not finite.any():
        return None, None, (lo, hi)
    s_f = s[finite]
    oob = float(np.mean((s < lo) | (s > hi)))
    return float(np.max(s_f)), oob, (lo, hi)


def poly_transformer_layer_numpy(
    hidden: np.ndarray,
    *,
    weights,
    approximations,
    attention_layout: AttentionPairLayout,
    rope_theta: float = 500000.0,
) -> PolyLayerResult:
    """One transformer layer with poly RMSNorm / SiLU and exact Softmax."""

    residual = np.asarray(hidden, dtype=np.float32)
    seq_len, hidden_dim = residual.shape
    expected_q = (hidden_dim, hidden_dim)
    if weights.q_proj.shape != expected_q:
        raise ValueError(
            f"q_proj must have shape {expected_q}, got {weights.q_proj.shape}."
        )
    kv_width = int(attention_layout.key_value_heads) * int(attention_layout.head_dim)
    if weights.k_proj.shape != (kv_width, hidden_dim):
        raise ValueError(
            "k_proj must have shape "
            f"{(kv_width, hidden_dim)}, got {weights.k_proj.shape}."
        )

    rmsn1_s_max, rmsn1_oob_frac, rmsn1_fit = _variance_domain(
        residual, approximations.input_norm
    )
    attn_in = _rmsnorm_poly(residual, approximations.input_norm)
    query = attn_in @ np.asarray(weights.q_proj, dtype=np.float32).T
    key = attn_in @ np.asarray(weights.k_proj, dtype=np.float32).T
    value = attn_in @ np.asarray(weights.v_proj, dtype=np.float32).T
    cos, sin = rope_cos_sin(
        seq_len=int(seq_len),
        head_dim=int(attention_layout.head_dim),
        theta=float(rope_theta),
    )
    query = apply_rope_heads(
        query,
        cos,
        sin,
        num_heads=int(attention_layout.query_heads),
        head_dim=int(attention_layout.head_dim),
    )
    key = apply_rope_heads(
        key,
        cos,
        sin,
        num_heads=int(attention_layout.key_value_heads),
        head_dim=int(attention_layout.head_dim),
    )
    _, _, attn_out = attention_gqa_numpy(
        query, key, value, layout=attention_layout
    )
    residual = residual + attn_out @ np.asarray(weights.o_proj, dtype=np.float32).T

    rmsn2_s_max, rmsn2_oob_frac, rmsn2_fit = _variance_domain(
        residual, approximations.post_attention_norm
    )
    mlp_in = _rmsnorm_poly(residual, approximations.post_attention_norm)
    gate = mlp_in @ np.asarray(weights.gate_proj, dtype=np.float32).T
    up = mlp_in @ np.asarray(weights.up_proj, dtype=np.float32).T
    silu_entry = approximations.silu
    _, lo, hi = _poly_entry_interval(silu_entry)
    finite_gate = np.isfinite(gate)
    if finite_gate.any():
        gate_abs = np.abs(gate[finite_gate])
        silu_gate_max_abs = float(np.max(gate_abs))
        silu_oob_frac = float(np.mean((gate < lo) | (gate > hi)))
    else:
        silu_gate_max_abs = None
        silu_oob_frac = None
    activated = _silu_poly(gate, silu_entry) * up
    residual = residual + activated @ np.asarray(weights.down_proj, dtype=np.float32).T
    return PolyLayerResult(
        hidden=np.asarray(residual, dtype=np.float32),
        silu_gate_max_abs=silu_gate_max_abs,
        silu_oob_frac=silu_oob_frac,
        silu_fit_lo=lo,
        silu_fit_hi=hi,
        rmsn1_s_max=rmsn1_s_max,
        rmsn1_oob_frac=rmsn1_oob_frac,
        rmsn1_fit=rmsn1_fit,
        rmsn2_s_max=rmsn2_s_max,
        rmsn2_oob_frac=rmsn2_oob_frac,
        rmsn2_fit=rmsn2_fit,
    )


__all__ = ["PolyLayerResult", "poly_transformer_layer_numpy"]
