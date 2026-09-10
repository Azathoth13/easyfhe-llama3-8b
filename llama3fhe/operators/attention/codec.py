from __future__ import annotations

"""Small ciphertext codecs shared by attention layout transforms."""

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout


def repeat_payload_cipher(cipher, *, layout: AttentionPairLayout, crypto_context):
    """Repeat one active pair payload across otherwise unused slot copies."""

    result = cipher
    step = int(layout.payload_slots)
    while step < int(layout.slots):
        rotated = crypto_context.fhe.homo_rotate(
            result, -step, crypto_context.context
        )
        combined = crypto_context.fhe.homo_add(
            result, rotated, crypto_context.context
        )
        release_if_supported(rotated)
        if result is not cipher:
            release_if_supported(result)
        result = combined
        step *= 2
    if result is cipher:
        return cipher
    release_if_supported(cipher)
    return result


def decrypt_complex_rows(
    ciphers: tuple[object, ...], *, crypto_context
) -> np.ndarray:
    """Verification-only decode preserving real and imaginary slot values."""

    rows: list[np.ndarray] = []
    for cipher in ciphers:
        imaginary_cipher = fhe.homo_mul_i(
            cipher, crypto_context.context, negative=True
        )
        try:
            real = np.asarray(crypto_context.decrypt(cipher), dtype=np.float64)
            imaginary = np.asarray(
                crypto_context.decrypt(imaginary_cipher), dtype=np.float64
            )
        finally:
            release_if_supported(imaginary_cipher)
        rows.append(real.astype(np.complex128) + 1j * imaginary)
    return np.stack(rows)


__all__ = ["decrypt_complex_rows", "repeat_payload_cipher"]
