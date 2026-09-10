"""Attention subsystem: QKV carriers in, W_O-ready carriers out.

The package owns RoPE/layout preparation, Delta QK, Delta-native Softmax,
Delta PV, and the PairFM output adapter. QKV and W_O remain linear operators
outside this boundary.
"""

from ...approx.attention import (
    DeltaAttentionResult,
    apply_rope_heads,
    attention_gqa_numpy,
    pv_delta_numpy,
    qk_delta_numpy,
    rope_cos_sin,
    run_delta_attention_numpy,
    softmax_delta_numpy,
)
from .config import AttentionOperatorConfig
from .input import (
    ComplexTokenAttentionComplexity,
    ComplexTokenAttentionResult,
    complex_token_attention_complexity,
    complex_token_attention_rotations,
    prepare_attention_inputs_fhe,
)

__all__ = [name for name in globals() if not name.startswith("_")]
