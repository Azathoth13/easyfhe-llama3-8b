from __future__ import annotations

"""Shared BSGS transform for public slot-permutation masks."""

import time
from math import ceil

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_all, release_if_supported, synchronize_device


def transform_rotation_set(
    *, seq_len: int, stride: int, baby_steps: int
) -> set[int]:
    rows = 2 * int(seq_len) - 1
    giant_count = int(ceil(rows / int(baby_steps)))
    rotations = {-(int(seq_len) - 1) * int(stride)}
    rotations.update(baby * int(stride) for baby in range(1, int(baby_steps)))
    if giant_count > 1:
        rotations.add(int(baby_steps) * int(stride))
    return {int(rotation) for rotation in rotations if int(rotation)}


def _bsgs_plaintext_rows(
    masks: np.ndarray, *, stride: int, baby_steps: int
) -> tuple[np.ndarray, tuple[int, ...], int, int]:
    masks = np.asarray(masks)
    giant_count = int(ceil(masks.shape[0] / int(baby_steps)))
    padded = np.zeros(
        (giant_count * int(baby_steps), masks.shape[1]), dtype=masks.dtype
    )
    for relative in range(masks.shape[0]):
        giant = relative // int(baby_steps)
        padded[relative] = np.roll(
            masks[relative], giant * int(baby_steps) * int(stride)
        )
    baby_offsets = tuple(
        baby * int(stride) for baby in range(int(baby_steps))
    )
    base_rotation = -(masks.shape[0] // 2) * int(stride)
    return padded, baby_offsets, giant_count, base_rotation


def encode_transform_rows(
    rows: np.ndarray, *, name: str, level: int, crypto_context
):
    rows = np.asarray(rows)
    bundle = crypto_context.constant_bundle(
        {"rows": rows},
        cache_key=("transform_rows", str(name)),
    )
    cur_limbs = int(crypto_context.context.L) - int(level)
    return bundle.plaintext(
        "rows",
        state=fhe.CipherState(
            cur_limbs=cur_limbs,
            scale_degree=1,
            scaling_factor=crypto_context.context.scale_at(cur_limbs),
        ),
        slots=int(crypto_context.max_slots),
        context=crypto_context.context,
        is_ext=True,
        cache=True,
    )


def _apply_level_group(
    ciphers: tuple[object, ...],
    *,
    masks: np.ndarray,
    stride: int,
    baby_steps: int,
    crypto_context,
    name: str,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    if not ciphers:
        return ()
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    if len(levels) != 1:
        raise ValueError("one transform group must contain one ciphertext level.")
    rows, baby_offsets, giant_count, base_rotation = _bsgs_plaintext_rows(
        masks, stride=int(stride), baby_steps=int(baby_steps)
    )
    started = time.perf_counter()
    plaintexts = None
    outputs: list[object] = []
    try:
        plaintexts = encode_transform_rows(
            rows,
            name=name,
            level=levels.pop(),
            crypto_context=crypto_context,
        )
        synchronize_device(crypto_context.device)
        if profile is not None:
            profile["pack_encode"] = (
                profile.get("pack_encode", 0.0) + time.perf_counter() - started
            )
        started = time.perf_counter()
        for cipher in ciphers:
            base = crypto_context.fhe.homo_rotate(
                cipher, base_rotation, crypto_context.context
            )
            try:
                accumulated = crypto_context.fhe.hoisted_mac_sum(
                    base,
                    baby_offsets,
                    plaintexts,
                    int(baby_steps) * int(stride),
                    giant_count,
                    crypto_context.context,
                    strategy="ext_normal",
                )
            finally:
                release_if_supported(base)
            try:
                outputs.append(
                    crypto_context.fhe.rescale(
                        accumulated, crypto_context.context
                    )
                )
            finally:
                release_if_supported(accumulated)
        synchronize_device(crypto_context.device)
        if profile is not None:
            profile["online_rescale"] = (
                profile.get("online_rescale", 0.0)
                + time.perf_counter()
                - started
            )
        return tuple(outputs)
    except Exception:
        release_all(outputs)
        raise
    finally:
        release_if_supported(plaintexts)


def apply_transform_by_level(
    ciphers: tuple[object, ...],
    *,
    masks: np.ndarray,
    stride: int,
    baby_steps: int,
    crypto_context,
    name: str,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    """Apply one mask transform, encoding the masks once per input level.

    Ciphertexts at the same level share one plaintext encode, so the number of
    distinct levels in ``ciphers`` is what this costs; it is reported as
    ``profile["level_groups"]`` rather than returned, because every caller
    wants only the transformed stream.
    """

    by_level: dict[int, list[tuple[int, object]]] = {}
    for index, cipher in enumerate(ciphers):
        level = int(crypto_context.level_for_cipher(cipher))
        by_level.setdefault(level, []).append((index, cipher))
    result: list[object | None] = [None] * len(ciphers)
    try:
        for level, entries in by_level.items():
            outputs = _apply_level_group(
                tuple(cipher for _, cipher in entries),
                masks=masks,
                stride=int(stride),
                baby_steps=int(baby_steps),
                crypto_context=crypto_context,
                name=f"{name}.level{level}",
                profile=profile,
            )
            for (index, _), output in zip(entries, outputs, strict=True):
                result[index] = output
        if any(output is None for output in result):
            raise RuntimeError("transform produced an incomplete output set.")
        if profile is not None:
            profile["level_groups"] = float(len(by_level))
        return tuple(result)  # type: ignore[return-value]
    except Exception:
        release_all(output for output in result if output is not None)
        raise


__all__ = [
    "apply_transform_by_level",
    "encode_transform_rows",
    "transform_rotation_set",
]
