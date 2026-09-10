"""Elementwise approximations on ciphertext scalar streams.

One family, one module: exp, reciprocal, inverse square root, and SiLU,
each evaluated through the shared Chebyshev machinery in
:mod:`polynomial`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import numpy as np

from .polynomial import eval_chebyshev_series_cipher


def eval_exp_cipher(
    input_cipher,
    layer_exp_entry: Dict[str, Any],
    *,
    crypto_context,
    input_is_chebyshev_mapped: bool = False,
):
    """Evaluate a pre-fitted exponential Chebyshev series."""
    lo, hi = layer_exp_entry["fit_interval"]
    kwargs = {}
    if bool(input_is_chebyshev_mapped):
        kwargs["input_is_chebyshev_mapped"] = True
    return eval_chebyshev_series_cipher(
        input_cipher,
        np.asarray(layer_exp_entry["coefficients"], dtype=np.float64),
        lower_bound=float(lo),
        upper_bound=float(hi),
        crypto_context=crypto_context,
        **kwargs,
    )


def eval_invsqrt_cipher(
    input_cipher,
    poly_entry: Dict[str, Any],
    *,
    crypto_context,
    input_is_chebyshev_mapped: bool = False,
):
    """Evaluate a pre-fitted inverse-square-root Chebyshev series."""
    cheb = poly_entry.get("chebyshev_fhe")
    if cheb is None:
        raise ValueError("invsqrt entry missing chebyshev_fhe coefficients")
    lo, hi = cheb["fit_interval"]
    kwargs = {}
    if bool(input_is_chebyshev_mapped):
        kwargs["input_is_chebyshev_mapped"] = True
    return eval_chebyshev_series_cipher(
        input_cipher,
        np.asarray(cheb["coefficients"], dtype=np.float64),
        lower_bound=float(lo),
        upper_bound=float(hi),
        crypto_context=crypto_context,
        **kwargs,
    )


def eval_silu_chebyshev_cipher(
    input_cipher,
    coefficients: Sequence[float],
    *,
    lower_bound: float,
    upper_bound: float,
    crypto_context,
    input_is_chebyshev_mapped: bool = False,
):
    """Evaluate pre-fitted SiLU coefficients on one scalar-stream ciphertext."""
    return eval_chebyshev_series_cipher(
        input_cipher,
        np.asarray(coefficients, dtype=np.float64),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        crypto_context=crypto_context,
        input_is_chebyshev_mapped=bool(input_is_chebyshev_mapped),
    )
