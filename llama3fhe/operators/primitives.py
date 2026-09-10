"""Owned-lifetime ciphertext arithmetic shared by the operator packages.

Every helper here follows one convention: it *consumes* the ciphertexts its
name says it consumes — the temporaries it allocates are released on both the
success and the failure path, so callers never see a leaked handle. Before
this module existed, each operator package carried its own byte-identical
copies of these helpers.

Nothing here is attention-, linear-, or norm-specific; kernels that need
layout-aware helpers keep them next to the kernel.
"""

from __future__ import annotations

import numpy as np
from easyfhe import fhe

from ..backend import release_if_supported


def mul_cipher_rescale(left, right, *, crypto_context):
    """``left * right`` with relinearization, rescaled; the product temporary
    is released."""

    product = crypto_context.fhe.homo_mul_relin(
        left, right, crypto_context.context
    )
    try:
        return crypto_context.fhe.rescale(product, crypto_context.context)
    finally:
        release_if_supported(product)


def square_rescale(cipher, *, crypto_context):
    """``cipher ** 2`` with relinearization, rescaled."""

    return mul_cipher_rescale(cipher, cipher, crypto_context=crypto_context)


def mul_plain_rescale(cipher, plain, *, crypto_context):
    """``cipher * plain``, rescaled; the product temporary is released."""

    product = crypto_context.fhe.homo_mul_pt(
        cipher, plain, crypto_context.context
    )
    try:
        return crypto_context.fhe.rescale(product, crypto_context.context)
    finally:
        release_if_supported(product)


def add_owned(left, right, *, crypto_context):
    """Add two owned temporaries; both are consumed unconditionally."""

    try:
        return crypto_context.fhe.homo_add(left, right, crypto_context.context)
    finally:
        release_if_supported(left)
        release_if_supported(right)


def accumulate_owned(current, term, *, crypto_context):
    """Fold ``term`` into an accumulator that may still be ``None``.

    On failure only ``term`` is released: ``current`` is still referenced by
    the caller's accumulator and is released by its outer cleanup.
    """

    if current is None:
        return term
    try:
        combined = crypto_context.fhe.homo_add(
            current, term, crypto_context.context
        )
    except Exception:
        release_if_supported(term)
        raise
    release_if_supported(current)
    release_if_supported(term)
    return combined


def bundle_plain(bundle, name: str, cipher, *, crypto_context, cache: bool = False):
    """Encode ``bundle[name]`` as a plaintext at ``cipher``'s limb state."""

    cur_limbs = int(cipher.state.cur_limbs)
    return bundle.plaintext(
        str(name),
        state=fhe.CipherState(
            cur_limbs=cur_limbs,
            scale_degree=1,
            scaling_factor=crypto_context.context.scale_at(cur_limbs),
        ),
        slots=int(crypto_context.max_slots),
        context=crypto_context.context,
        cache=bool(cache),
    )


def drop_to_limbs(
    ciphers, *, target_limbs: int, crypto_context
) -> tuple[object, ...]:
    """Return owned views of ``ciphers`` at ``target_limbs``, scale preserved.

    A modulus-only step: it sheds limbs without consuming multiplicative
    depth, which is what makes it cheap enough to schedule deliberately. The
    inputs must be non-extended degree-1 ciphertexts sharing one limb count --
    the shape every stream boundary in this system already has -- and the
    result is always owned by the caller, even where ``align_to`` could return
    its argument unchanged.
    """

    ciphers = tuple(ciphers)
    target_limbs = int(target_limbs)
    if target_limbs <= 1:
        raise ValueError(
            f"modulus drop must leave more than one limb, got {target_limbs}."
        )
    if any(
        int(cipher.state.scale_degree) != 1 or bool(cipher.is_ext)
        for cipher in ciphers
    ):
        raise ValueError(
            "modulus-only scheduling requires non-extended degree-1 "
            "ciphertexts."
        )
    current = {int(cipher.state.cur_limbs) for cipher in ciphers}
    if len(current) > 1:
        raise ValueError(
            f"modulus drop expects one common limb count, got {sorted(current)}."
        )
    if current and next(iter(current)) < target_limbs:
        raise ValueError(
            f"too few limbs for the drop: {next(iter(current))} < {target_limbs}."
        )

    outputs: list[object] = []
    try:
        for cipher in ciphers:
            target_state = cipher.state.replace(
                cur_limbs=target_limbs,
                scale_degree=1,
                scaling_factor=None,
            )
            aligned = crypto_context.fhe.align_to(
                cipher, target_state, crypto_context.context
            )
            outputs.append(
                cipher.deep_copy() if aligned is cipher else aligned
            )
        return tuple(outputs)
    except Exception:
        for cipher in outputs:
            release_if_supported(cipher)
        raise


def drop_levels(ciphers, levels: int, *, crypto_context) -> tuple[object, ...]:
    """``drop_to_limbs`` expressed relative to the stream's current level."""

    ciphers = tuple(ciphers)
    levels = int(levels)
    if levels < 0:
        raise ValueError(f"levels must be non-negative, got {levels}.")
    if not ciphers:
        return ()
    current = {int(cipher.state.cur_limbs) for cipher in ciphers}
    if len(current) != 1:
        raise ValueError(
            f"modulus drop expects one common limb count, got {sorted(current)}."
        )
    return drop_to_limbs(
        ciphers,
        target_limbs=next(iter(current)) - levels,
        crypto_context=crypto_context,
    )


def encode_mask(mask: np.ndarray, *, name: str, level: int, crypto_context, dtype=None):
    """Encode a mask vector as a non-extended plaintext at ``level``."""

    values = np.asarray(mask)
    return crypto_context.plaintext(
        values,
        name=str(name),
        level=int(level),
        slots=int(crypto_context.max_slots),
        dtype=values.dtype if dtype is None else dtype,
        is_ext=False,
    )


__all__ = [
    "accumulate_owned",
    "add_owned",
    "bundle_plain",
    "drop_levels",
    "drop_to_limbs",
    "encode_mask",
    "mul_cipher_rescale",
    "mul_plain_rescale",
    "square_rescale",
]
