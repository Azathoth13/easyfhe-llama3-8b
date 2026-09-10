from __future__ import annotations

"""Level-free Re/Im multiplexing around linear projections."""

from easyfhe import fhe

from llama3fhe.backend import release_if_supported


def pair_real_ciphers(
    ciphers: tuple[object, ...] | list[object], *, crypto_context
) -> tuple[object, ...]:
    """Return adjacent real streams as ``x[2r] + i*x[2r+1]``."""

    ciphers = tuple(ciphers)
    if len(ciphers) % 2:
        raise ValueError("complex pairing requires an even ciphertext count.")
    outputs: list[object] = []
    try:
        for index in range(0, len(ciphers), 2):
            imaginary = fhe.homo_mul_i(
                ciphers[index + 1], crypto_context.context
            )
            try:
                outputs.append(
                    crypto_context.fhe.homo_add(
                        ciphers[index], imaginary, crypto_context.context
                    )
                )
            finally:
                release_if_supported(imaginary)
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for output in outputs:
            release_if_supported(output)


def split_complex_cipher_twice(
    cipher: object, *, crypto_context
) -> tuple[object, object]:
    """Return ``(2*Re(z), 2*Im(z))`` without consuming a level."""

    conjugated = real_twice = delta = imag_twice = None
    try:
        conjugated = crypto_context.fhe.homo_rotate(
            cipher,
            int(crypto_context.context.M) - 1,
            crypto_context.context,
        )
        real_twice = crypto_context.fhe.homo_add(
            cipher, conjugated, crypto_context.context
        )
        delta = crypto_context.fhe.homo_sub(
            conjugated, cipher, crypto_context.context
        )
        imag_twice = fhe.homo_mul_i(delta, crypto_context.context)
        result = (real_twice, imag_twice)
        real_twice = imag_twice = None
        return result
    finally:
        release_if_supported(conjugated)
        release_if_supported(real_twice)
        release_if_supported(delta)
        release_if_supported(imag_twice)


def split_complex_ciphers_twice(
    ciphers: tuple[object, ...] | list[object], *, crypto_context
) -> tuple[object, ...]:
    """Split streams into interleaved ``2*Re, 2*Im`` outputs."""

    outputs: list[object] = []
    try:
        for cipher in tuple(ciphers):
            outputs.extend(
                split_complex_cipher_twice(
                    cipher, crypto_context=crypto_context
                )
            )
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for output in outputs:
            release_if_supported(output)
