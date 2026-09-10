"""Shared feature-major diagonal-linear infrastructure."""

from .complex import (
    pair_real_ciphers,
    split_complex_cipher_twice,
    split_complex_ciphers_twice,
)
from .config import (
    LLAMA3_LINEAR_SCHEDULES,
    LinearOperatorConfig,
    LinearPlan,
    Llama3LinearSchedules,
    baby_reuse_across_pages_enabled,
    linear_encode_reuse_enabled,
    linear_rotations,
    structured_weight_packing_enabled,
)
from .operator import LinearOperator
from .structured_weight import StructuredLinearWeight, structured_linear_weight
from .weight import LinearWeight

__all__ = [
    "LinearOperator",
    "LinearOperatorConfig",
    "LinearPlan",
    "LinearWeight",
    "StructuredLinearWeight",
    "LLAMA3_LINEAR_SCHEDULES",
    "Llama3LinearSchedules",
    "linear_rotations",
    "pair_real_ciphers",
    "split_complex_cipher_twice",
    "split_complex_ciphers_twice",
    "structured_linear_weight",
    "structured_weight_packing_enabled",
    "linear_encode_reuse_enabled",
    "baby_reuse_across_pages_enabled",
]
