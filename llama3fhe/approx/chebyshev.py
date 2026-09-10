"""Chebyshev and Paterson-Stockmeyer coefficient algebra.

Pure coefficient arithmetic on NumPy arrays: fitting a Chebyshev series,
choosing the Paterson-Stockmeyer degree split, the multiplicative depth that
split implies, and direct plaintext evaluation for reference checks.

Every consumer of encrypted polynomial evaluation needs the depth and the split
from here, which is why this is a shared approximation module rather than
something owned by one operator.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

import numpy as np

_PS_SPLIT_CACHE: dict[bytes, tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = {}


def eval_chebyshev_coefficients(
    func: Callable[[float], float],
    a: float,
    b: float,
    poly_degree: int,
) -> np.ndarray:
    """Compute Chebyshev series coefficients for *func* on [a, b].

    Evaluate coefficients in the convention used by the release artifacts.
    """
    if poly_degree <= 0:
        raise ValueError("poly_degree must be positive")
    coeff_total = poly_degree + 1
    b_minus_a = 0.5 * (b - a)
    b_plus_a = 0.5 * (b + a)
    pi_by_deg = math.pi / coeff_total
    function_points = [
        func(math.cos(pi_by_deg * (i + 0.5)) * b_minus_a + b_plus_a)
        for i in range(coeff_total)
    ]
    mult_factor = 2.0 / coeff_total
    coefficients = [0.0] * coeff_total
    for i in range(coeff_total):
        for j in range(coeff_total):
            coefficients[i] += function_points[j] * math.cos(pi_by_deg * i * (j + 0.5))
        coefficients[i] *= mult_factor
    return np.asarray(coefficients, dtype=np.float64)


def _populate_parameter_ps(upper_bound_degree: int) -> np.ndarray:
    mlist = np.zeros(upper_bound_degree, dtype=np.int32)
    ranges = [
        (1, 2, 1),
        (3, 11, 2),
        (12, 13, 3),
        (14, 17, 2),
        (18, 55, 3),
        (56, 59, 4),
        (60, 76, 3),
        (77, 239, 4),
        (240, 247, 5),
        (248, 284, 4),
        (285, 991, 5),
        (992, 1007, 6),
        (1008, 1083, 5),
        (1084, 2015, 6),
        (2016, 2031, 7),
        (2032, 2204, 6),
    ]
    for start, end, m in ranges:
        if upper_bound_degree < start:
            break
        actual_end = min(end, upper_bound_degree)
        mlist[start - 1 : actual_end] = m
    return mlist


def compute_degrees_ps(n: int) -> Tuple[int, int]:
    """Return the Paterson–Stockmeyer (k, m) for Chebyshev degree *n*."""
    if n <= 0:
        raise ValueError("degree must be positive")
    upper_bound_ps = 2204
    if n <= upper_bound_ps:
        mlist = _populate_parameter_ps(upper_bound_ps)
        m = int(mlist[n - 1])
        k = math.floor(n / ((1 << m) - 1)) + 1
        return k, m

    klist: List[int] = []
    mlist_out: List[int] = []
    multlist: List[int] = []
    sqrt_half_n = math.sqrt(n / 2)
    floor_log2_sqrt_half_n = math.floor(math.log2(sqrt_half_n)) if sqrt_half_n > 0 else 0
    for k in range(1, n + 1):
        max_m = math.ceil(math.log2(n / k) + 1) + 1
        for m in range(1, int(max_m) + 1):
            rhs = k * ((1 << m) - 1)
            if n - rhs < 0:
                floor_log2_k = math.floor(math.log2(k))
                if abs(floor_log2_k - floor_log2_sqrt_half_n) <= 1:
                    klist.append(k)
                    mlist_out.append(m)
                    multlist.append(k + 2 * m + (1 << (m - 1)) - 4)
    if not multlist:
        raise ValueError("No valid (k, m) pairs found")
    min_index = multlist.index(min(multlist))
    return klist[min_index], mlist_out[min_index]


def chebyshev_ps_mul_depth(coefficients: Sequence[float]) -> int:
    """CKKS multiplicative depth for Paterson–Stockmeyer Chebyshev eval.

    Return the Paterson-Stockmeyer multiplicative-depth estimate:
    ``depth = ceil(log2(k)) + m``.
    """
    n = chebyshev_degree(coefficients)
    if n <= 0:
        return 0
    k, m = compute_degrees_ps(n)
    return math.ceil(math.log2(k)) + m


def _map_to_chebyshev_domain(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Map *x* from [a, b] → [-1, 1]."""
    alpha = 2.0 / (b - a)
    beta = 2.0 * a / (b - a)
    y = x.astype(np.float64, copy=True)
    if not math.isclose(alpha, 1.0):
        y *= alpha
    if not math.isclose(beta, -1.0):
        y += -1.0 - beta
    return y


def eval_chebyshev_series_direct(
    x: np.ndarray,
    coefficients: Sequence[float],
    a: float,
    b: float,
) -> np.ndarray:
    """Evaluate a Chebyshev series on [a, b] using the T_n recurrence (float64).

    Reference evaluator; **not** depth-optimal for CKKS.
    """
    coeffs = np.asarray(coefficients, dtype=np.float64)
    y = _map_to_chebyshev_domain(np.asarray(x), a, b)
    if len(coeffs) == 0:
        return np.zeros_like(x, dtype=np.float32)
    if len(coeffs) == 1:
        return np.full_like(x, coeffs[0] / 2.0, dtype=np.float32)

    t0 = np.ones_like(y)
    t1 = y
    acc = np.full_like(y, coeffs[0] / 2.0)
    if len(coeffs) > 1:
        acc += coeffs[1] * t1
    t_prev, t_curr = t0, t1
    for i in range(2, len(coeffs)):
        t_next = 2.0 * y * t_curr - t_prev
        acc += coeffs[i] * t_next
        t_prev, t_curr = t_curr, t_next
    return acc.astype(np.float32, copy=False)


def pad_coeffs(coefficients: Sequence[float], size: int) -> np.ndarray:
    """Zero-pad or truncate a Chebyshev coefficient vector."""

    result = np.zeros(int(size), dtype=np.float64)
    source = np.asarray(coefficients, dtype=np.float64)
    result[: min(source.size, int(size))] = source[: int(size)]
    return result


def chebyshev_degree(coefficients: Sequence[float]) -> int:
    for index in range(len(coefficients) - 1, -1, -1):
        if coefficients[index] != 0:
            return index
    return 0


def long_division_chebyshev(
    dividend: np.ndarray, divisor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return quotient and remainder for Chebyshev-basis long division."""

    f = np.asarray(dividend, dtype=np.float64)
    g = np.asarray(divisor, dtype=np.float64)
    if math.isclose(f[-1], 0) or math.isclose(g[-1], 0):
        raise ValueError("leading Chebyshev coefficients must be non-zero.")
    n, k = len(f) - 1, len(g) - 1
    if n < k:
        return np.array([1.0], dtype=np.float64), np.copy(f)

    quotient = np.zeros(n - k + 1, dtype=np.float64)
    remainder = np.copy(f)
    while n > k:
        quotient[n - k] = 2.0 * remainder[-1] / g[-1]
        product = np.zeros(n + 1, dtype=np.float64)
        if k == n - k:
            product[0] = 2.0 * g[n - k]
            for index in range(1, 2 * k + 1):
                product[index] = g[abs(n - k - index)]
        elif k > n - k:
            product[0] = 2.0 * g[n - k]
            for index in range(1, k - (n - k) + 1):
                product[index] = g[abs(n - k - index)] + g[n - k + index]
            for index in range(k - (n - k) + 1, n + 1):
                product[index] = g[abs(index - n + k)]
        else:
            product[n - k] = g[0]
            for index in range(n - 2 * k, n + 1):
                if index != n - k:
                    product[index] = g[abs(index - n + k)]
        remainder -= product * remainder[-1] / g[-1]
        if len(remainder) > 1:
            n = chebyshev_degree(remainder)
            remainder = remainder[: n + 1]

    if n == k:
        quotient[0] = remainder[-1] / g[-1]
        remainder -= g * quotient[0]
        if len(remainder) > 1:
            n = chebyshev_degree(remainder)
            remainder = remainder[: n + 1]

    quotient[0] *= 2.0
    return quotient, remainder


def prepare_top_level_ps_split(
    coefficients: Sequence[float],
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(k, m, divqr_q, divcs_q, s2)`` for encrypted PS evaluation.

    The split is a pure function of the (immutable, run-constant) Chebyshev
    coefficients, yet it is invoked on every nonlinear ciphertext evaluation
    (SiLU twice per gate cipher, exp per softmax round, rsqrt per RMSNorm).
    Memoize on the coefficient bytes and hand back fresh array copies so the
    O(degree^2) long-division runs once per distinct polynomial per process.
    """

    coefficients = np.asarray(coefficients, dtype=np.float64)
    cache_key = np.ascontiguousarray(coefficients).tobytes()
    cached = _PS_SPLIT_CACHE.get(cache_key)
    if cached is not None:
        k, m, divqr_q, divcs_q, s2 = cached
        return k, m, divqr_q.copy(), divcs_q.copy(), s2.copy()

    degree = chebyshev_degree(coefficients)
    if degree <= 0:
        raise ValueError("Chebyshev series degree must be positive.")
    polynomial = np.copy(coefficients[: degree + 1])
    k, m = compute_degrees_ps(degree)
    split_degree = k * (1 << (m - 1)) - k
    polynomial = pad_coeffs(polynomial, 2 * split_degree + k + 1)
    polynomial[-1] = 1.0

    top_basis = np.zeros(split_degree + k + 1, dtype=np.float64)
    top_basis[-1] = 1.0
    divqr_q, divqr_r = long_division_chebyshev(polynomial, top_basis)

    adjusted = np.copy(divqr_r)
    if split_degree - chebyshev_degree(divqr_r) <= 0:
        adjusted[split_degree] -= 1.0
        adjusted = adjusted[: chebyshev_degree(adjusted) + 1]
    else:
        adjusted = pad_coeffs([], split_degree + 1)
        adjusted[-1] = -1.0

    divcs_q, divcs_r = long_division_chebyshev(adjusted, divqr_q)
    s2 = pad_coeffs(divcs_r, split_degree + 1)
    s2[-1] = 1.0
    _PS_SPLIT_CACHE[cache_key] = (k, m, divqr_q, divcs_q, s2)
    return k, m, divqr_q.copy(), divcs_q.copy(), s2.copy()


__all__ = [
    "eval_chebyshev_coefficients",
    "compute_degrees_ps",
    "chebyshev_ps_mul_depth",
    "eval_chebyshev_series_direct",
    "pad_coeffs",
    "chebyshev_degree",
    "long_division_chebyshev",
    "prepare_top_level_ps_split",
]
