"""Shared FHE polynomial evaluation on CKKS ciphertext scalar streams.

Uses Paterson–Stockmeyer evaluation for depth-optimal Chebyshev series on
ciphertext inputs, with EasyFHE fused multiply-relin-rescale-postop kernels
(``homo_mul_relin_rescale_postop``, ``grouped_scalar_weighted_acc``, etc.)
for single-GPU-kernel basis construction and linear combinations.
"""

from __future__ import annotations

import math

import numpy as np

from ...approx.chebyshev import (
    chebyshev_degree,
    long_division_chebyshev,
    pad_coeffs,
    prepare_top_level_ps_split,
)
from ...backend import release_if_supported as _release


class _ChebPSEvaluator:
    """Paterson–Stockmeyer Chebyshev evaluator on CKKS ciphertexts.

    Uses EasyFHE fused kernels (``homo_mul_relin_rescale_postop``,
    ``grouped_scalar_weighted_acc``, ``homo_mul_scalar_double``,
    ``homo_add_scalar_double``) to minimize GPU kernel launches during
    Chebyshev basis construction and linear combination.  This shaves
    ~30 % off the per-cipher evaluation time vs the chunked pt-mul path.
    """

    def __init__(self, crypto_context):
        self.fhe = crypto_context.fhe
        self.ctx = crypto_context.context

    # ---- scalar helpers (one kernel each) ----

    def _enc_double(self, value: float, ct):
        return self.fhe.encode_scalar(
            float(value),
            cur_limbs=int(ct.state.cur_limbs),
            scale_degree=1,
            scaling_factor=ct.state.scaling_factor,
            context=self.ctx,
        )

    def _enc_post_double(self, value: float, left, right):
        """Encode a scalar for the output state of fused mul/rescale."""

        cur_limbs = min(
            int(left.state.cur_limbs), int(right.state.cur_limbs)
        )
        output_limbs = cur_limbs - 1
        output_scale = (
            float(left.state.scaling_factor)
            * float(right.state.scaling_factor)
            / float(self.ctx.rescale_divisor_at(output_limbs))
        )
        return self.fhe.encode_scalar(
            float(value),
            cur_limbs=output_limbs,
            scale_degree=1,
            scaling_factor=output_scale,
            context=self.ctx,
        )

    def _mul_scalar(self, ct, scalar: float):
        if math.isclose(scalar, 1.0):
            return ct
        enc = self._enc_double(scalar, ct)
        out = self.fhe.homo_mul_scalar_rescale(ct, enc, self.ctx)
        if out is not ct:
            _release(ct)
        return out

    def _add_scalar(self, ct, scalar: float):
        if abs(scalar) < 1e-20:
            return ct
        enc = self._enc_double(scalar, ct)
        return self.fhe.homo_add_scalar(ct, enc, self.ctx)

    # ---- alignment helpers ----

    def _align_to(self, ct, target_state):
        if int(ct.state.cur_limbs) == int(target_state.cur_limbs):
            return ct
        if int(ct.state.cur_limbs) < int(target_state.cur_limbs):
            raise ValueError(
                "cannot align shallower ciphertext to deeper target; "
                f"source cur_limbs={ct.state.cur_limbs}, target cur_limbs={target_state.cur_limbs}"
            )
        aligned = self.fhe.align_to(ct, target_state, self.ctx)
        if aligned is not ct:
            _release(ct)
        return aligned

    def _add_aligned(self, left, right):
        if int(left.state.cur_limbs) > int(right.state.cur_limbs):
            left = self.fhe.align_to(left, right.state, self.ctx)
        elif int(right.state.cur_limbs) > int(left.state.cur_limbs):
            right = self.fhe.align_to(right, left.state, self.ctx)
        return self.fhe.homo_add(left, right, self.ctx)

    # ---- Chebyshev basis (fused — one kernel per order) ----

    def _chebyshev_basis(self, x, k: int) -> list:
        """Return ``[T_1, …, T_k]`` using ``homo_mul_relin_rescale_postop``."""
        T: list = [x]
        for order in range(2, k + 1):
            lhs = T[order // 2 - 1]
            rhs = T[(order + 1) // 2 - 1]
            if order & 1:  # odd: 2·lhs·rhs − T₁
                value = self.fhe.homo_mul_relin_rescale_postop(
                    lhs, rhs, self.ctx, apply_double=True, sub=T[0],
                )
            else:  # even: 2·lhs·rhs − 1 (i.e. + encoded(−1))
                value = self.fhe.homo_mul_relin_rescale_postop(
                    lhs,
                    rhs,
                    self.ctx,
                    apply_double=True,
                    scalar=self._enc_post_double(-1.0, lhs, rhs),
                )
            T.append(value)

        # Align all to same level for batch usage.
        target = T[-1].state
        for i in range(k):
            if int(T[i].state.cur_limbs) != int(target.cur_limbs):
                old = T[i]
                T[i] = self.fhe.align_to(old, target, self.ctx)
                if T[i] is not old:
                    _release(old)
        return T

    def _doubling_basis(self, Tk, m: int) -> list:
        """Return ``[T_k, T_{2k}, …, T_{k·2^{m-1}}]``."""
        T2 = [Tk]
        for _ in range(1, m):
            current = T2[-1]
            value = self.fhe.homo_mul_relin_rescale_postop(
                current,
                current,
                self.ctx,
                apply_double=True,
                scalar=self._enc_post_double(-1.0, current, current),
            )
            T2.append(value)
        return T2

    def _odd_multiple(self, T2, Tk) -> object:
        """Return ``T_{k·(2^m−1)}`` from ``[T_k, …, T_{k·2^{m-1}}]``."""
        value = Tk
        for doubled in T2[1:]:
            value = self.fhe.homo_mul_relin_rescale_postop(
                value, doubled, self.ctx, apply_double=True, sub=Tk,
            )
        return value

    # ---- Linear combination (batched fused kernel) ----

    def _linear_wsum(self, ciphers: list, weights: np.ndarray):
        raw_terms = []
        for ct, weight in zip(ciphers, weights, strict=False):
            w = float(weight)
            if abs(w) < 1e-20:
                continue
            enc = self._enc_double(w, ct)
            term = self.fhe.homo_mul_scalar(ct, enc, self.ctx)
            raw_terms.append(term)
        if not raw_terms:
            raise ValueError("linear combination produced no terms")
        acc = raw_terms[0]
        for term in raw_terms[1:]:
            term = self._align_to(term, acc.state)
            new_acc = self._add_aligned(acc, term)
            _release(acc)
            _release(term)
            acc = new_acc
        return self.fhe.rescale(acc, self.ctx)

    # ---- Mapping ----

    def map_to_chebyshev_domain(self, input_cipher, *, a: float, b: float):
        """Map ``[a, b]`` onto the Chebyshev domain ``[-1, 1]`` in place."""

        alpha = 2.0 / (b - a)
        beta = -1.0 - 2.0 * a / (b - a)
        y = input_cipher
        if not math.isclose(alpha, 1.0):
            y = self._mul_scalar(y, alpha)
        if not math.isclose(beta, 0.0):
            y = self._add_scalar(y, beta)
        return y

    # ---- Products and sums used by the PS recursion ----

    def _mul_rescale(self, left, right):
        return self.fhe.homo_mul_relin_rescale_postop(
            left, right, self.ctx
        )

    def _add(self, left, right):
        return self._add_aligned(left, right)

    def _sub(self, left, right):
        aligned_right = self.fhe.align_to(right, left.state, self.ctx)
        try:
            return self.fhe.homo_sub(left, aligned_right, self.ctx)
        finally:
            if aligned_right is not right:
                _release(aligned_right)


def _inner_qu_from_divqr_cipher(
    ev: _ChebPSEvaluator, divqr_q: np.ndarray, t_basis: list, k: int
):
    """Inner ``qu`` branch of the Paterson-Stockmeyer recursion."""

    deg_qcopy = chebyshev_degree(pad_coeffs(np.copy(divqr_q), k))
    lead = float(divqr_q[-1]) + 1.1
    scale = 2 ** math.floor(math.log2(lead))
    if deg_qcopy > 0:
        qu = ev._linear_wsum(t_basis[:deg_qcopy], divqr_q[1 : deg_qcopy + 1])
        scaled = ev._mul_scalar(t_basis[k - 1], float(scale))
        qu = ev._add(qu, scaled)
        _release(scaled)
    else:
        qu = ev._mul_scalar(t_basis[k - 1], float(scale))
    return ev._add_scalar(qu, float(divqr_q[0]) / 2.0)


def _top_qu_from_divqr_cipher(
    ev: _ChebPSEvaluator, divqr_q: np.ndarray, t_basis: list, k: int
):
    """Top-level ``qu`` branch of the Paterson-Stockmeyer split."""

    deg_qcopy = chebyshev_degree(divqr_q[:k])
    if deg_qcopy > 0:
        qu = ev._linear_wsum(t_basis[:deg_qcopy], divqr_q[1 : deg_qcopy + 1])
        tk_sum = ev._add(t_basis[k - 1], t_basis[k - 1])
        qu = ev._add(qu, tk_sum)
        _release(tk_sum)
    else:
        qu = t_basis[k - 1]
        for _ in range(1, int(divqr_q[-1])):
            new_qu = ev._add(qu, t_basis[k - 1])
            if qu is not t_basis[k - 1]:
                _release(qu)
            qu = new_qu
    return ev._add_scalar(qu, float(divqr_q[0]) / 2.0)


def _inner_eval_chebyshev_ps_cipher(
    ev: _ChebPSEvaluator,
    coefficients: np.ndarray,
    k: int,
    m: int,
    t_basis: list,
    t2_basis: list,
):
    """One recursive Paterson-Stockmeyer step on the ciphertext basis."""

    k2m2k = k * (1 << (m - 1)) - k
    tkm = np.zeros(int(k2m2k + k) + 1, dtype=np.float64)
    tkm[-1] = 1.0
    divqr_q, divqr_r = long_division_chebyshev(coefficients, tkm)

    r2 = np.copy(divqr_r)
    if int(k2m2k - chebyshev_degree(divqr_r)) <= 0:
        r2[k2m2k] -= 1.0
        r2 = r2[: chebyshev_degree(r2) + 1]
    else:
        r2 = pad_coeffs([], k2m2k + 1)
        r2[-1] = -1.0

    divcs_q, divcs_r = long_division_chebyshev(r2, divqr_q)
    s2 = pad_coeffs(divcs_r, k2m2k + 1)
    s2[-1] = 1.0

    dc = chebyshev_degree(divcs_q)
    cu = None
    if dc >= 1:
        if dc == 1:
            cu = (
                t_basis[0]
                if math.isclose(float(divcs_q[1]), 1.0)
                else ev._mul_scalar(t_basis[0], float(divcs_q[1]))
            )
        else:
            cu = ev._linear_wsum(t_basis[:dc], divcs_q[1 : dc + 1])
        cu = ev._add_scalar(cu, float(divcs_q[0]) / 2.0)
        cu = ev._align_to(cu, t2_basis[m - 1].state)

    if chebyshev_degree(divqr_q) > k:
        qu = _inner_eval_chebyshev_ps_cipher(
            ev, divqr_q, k, m - 1, t_basis, t2_basis
        )
    else:
        qu = _inner_qu_from_divqr_cipher(ev, divqr_q, t_basis, k)

    if chebyshev_degree(s2) > k:
        su = _inner_eval_chebyshev_ps_cipher(ev, s2, k, m - 1, t_basis, t2_basis)
    else:
        deg_scopy = chebyshev_degree(pad_coeffs(np.copy(s2), k))
        if deg_scopy > 0:
            su = ev._linear_wsum(t_basis[:deg_scopy], s2[1 : deg_scopy + 1])
            su = ev._add(su, t_basis[k - 1])
        else:
            su = t_basis[k - 1]
        su = ev._add_scalar(su, float(s2[0]) / 2.0)

    if cu is not None:
        result = ev._add(t2_basis[m - 1], cu)
    else:
        result = ev._add_scalar(t2_basis[m - 1], float(divcs_q[0]) / 2.0)

    su_aligned = ev._align_to(su, result.state)
    prod = ev._mul_rescale(result, qu)
    _release(result)
    _release(qu)
    out = ev._add(prod, su_aligned)
    _release(prod)
    if su_aligned is not su:
        _release(su_aligned)
    return out


def eval_chebyshev_series_cipher(
    input_cipher,
    coefficients: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    crypto_context,
    input_is_chebyshev_mapped: bool = False,
) -> object:
    """Evaluate pre-fitted Chebyshev coefficients on one ciphertext.

    Paterson-Stockmeyer evaluation, so the depth is ``ceil(log2(k)) + m``
    (~7 CKKS levels at degree 68) rather than the degree itself.
    """

    coeffs = np.asarray(coefficients, dtype=np.float64)
    n = chebyshev_degree(coeffs)
    if n < 1:
        raise ValueError(f"degree must be >= 1, got {n}.")

    a = float(lower_bound)
    b = float(upper_bound)
    k, m, divqr_q, divcs_q, s2 = prepare_top_level_ps_split(coeffs)

    ev = _ChebPSEvaluator(crypto_context)
    y = (
        input_cipher
        if bool(input_is_chebyshev_mapped)
        else ev.map_to_chebyshev_domain(input_cipher, a=a, b=b)
    )
    t_basis = ev._chebyshev_basis(y, k)
    t2_basis = ev._doubling_basis(t_basis[-1], m)
    t2km1 = ev._odd_multiple(t2_basis, t2_basis[0])

    dc = chebyshev_degree(divcs_q)
    cu = None
    if dc >= 1:
        if dc == 1:
            cu = (
                t_basis[0]
                if math.isclose(float(divcs_q[1]), 1.0)
                else ev._mul_scalar(t_basis[0], float(divcs_q[1]))
            )
        else:
            cu = ev._linear_wsum(t_basis[:dc], divcs_q[1 : dc + 1])
        cu = ev._add_scalar(cu, float(divcs_q[0]) / 2.0)

    if chebyshev_degree(divqr_q) > k:
        qu = _inner_eval_chebyshev_ps_cipher(
            ev, divqr_q, k, m - 1, t_basis, t2_basis
        )
    else:
        qu = _top_qu_from_divqr_cipher(ev, divqr_q, t_basis, k)

    if chebyshev_degree(s2) > k:
        su = _inner_eval_chebyshev_ps_cipher(ev, s2, k, m - 1, t_basis, t2_basis)
    else:
        deg_scopy = chebyshev_degree(np.copy(s2[:k]))
        if deg_scopy > 0:
            su = ev._linear_wsum(t_basis[:deg_scopy], s2[1 : deg_scopy + 1])
            su = ev._add(su, t_basis[k - 1])
        else:
            su = t_basis[k - 1]
        su = ev._add_scalar(su, float(s2[0]) / 2.0)

    if cu is not None:
        cu_aligned = ev._align_to(cu, t2_basis[m - 1].state)
        result = ev._add(t2_basis[m - 1], cu_aligned)
        if cu_aligned is not cu:
            _release(cu_aligned)
    else:
        result = ev._add_scalar(t2_basis[m - 1], float(divcs_q[0]) / 2.0)

    su_aligned = ev._align_to(su, result.state)
    prod = ev._mul_rescale(result, qu)
    _release(result)
    _release(qu)
    result = ev._add(prod, su_aligned)
    _release(prod)
    _release(su_aligned)
    result = ev._sub(result, t2km1)
    return result
