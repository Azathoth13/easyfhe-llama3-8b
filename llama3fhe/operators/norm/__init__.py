"""Encrypted normalization: RMSNorm and the fused residual refresh.

The initializer stays deliberately light, so public callers import the
submodule they need. The plaintext side of RMSNorm — its reference evaluation
and its fitted Chebyshev coefficients — lives in :mod:`llama3fhe.approx`.
"""

__all__: list[str] = []
