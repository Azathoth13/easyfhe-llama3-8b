"""Kernel-specific helpers shared by the Δ QK and PV kernels.

Small owned-lifetime building blocks — masked rotations, rescale-and-release
wrappers, mask bundles — used by both the QK and the PV kernel. Everything
here follows the package convention: a helper that allocates a ciphertext
releases it on the failure path, and consumes what its name says it consumes.
"""

from __future__ import annotations

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout

from ..primitives import (
    add_owned as _add_owned,
)
from ..primitives import (
    mul_plain_rescale as _mask_rescale,
)


def _level_one_scale_state(cipher, cur_limbs: int):
    """Build an aligned degree-one state at ``cur_limbs``."""

    return cipher.state.replace(
        cur_limbs=int(cur_limbs),
        scale_degree=1,
        scaling_factor=None,
    )


def _raw_bundle(vectors: dict[str, np.ndarray], *, crypto_context):
    arrays = {
        str(name): np.asarray(values, dtype=np.float64).reshape(-1)
        for name, values in vectors.items()
    }
    return crypto_context.constant_bundle(
        arrays,
        cache_key=("attention.kernel_masks", tuple(sorted(arrays))),
    )


def _mask_vectors(layout: AttentionPairLayout, *, kind: str) -> dict[str, np.ndarray]:
    """Build the 2*(S-1) shift masks the K or V torus shift selects between.

    A pure function of the layout, rebuilt on every QK and PV call: 254 vectors
    of full slot width, 63.5 MiB and 21 ms (k) / 5 ms (v) at production
    geometry, so roughly 0.8 s across a 32-layer model -- 0.16%. Memoizing it
    the way _choose_sparse_diagonal_plan_cached and prepare_top_level_ps_split
    memoize their pure functions would mean holding 127 MiB of host arrays and
    trusting every future caller not to mutate them; measured against the
    return, that trade was declined deliberately.
    """

    seq_len = int(layout.seq_len)
    vectors: dict[str, np.ndarray] = {}
    for shift in range(1, seq_len):
        if kind == "k":
            first = np.zeros((int(layout.head_dim), seq_len, 2), dtype=np.float64)
            second = np.zeros_like(first)
            first[:, shift:, :] = 1.0
            second[:, :shift, :] = 1.0
        elif kind == "v":
            first = np.zeros((int(layout.head_dim), seq_len, 2), dtype=np.float64)
            second = np.zeros_like(first)
            first[:, : seq_len - shift, :] = 1.0
            second[:, seq_len - shift :, :] = 1.0
        else:
            raise ValueError(f"unknown mask kind={kind!r}.")
        vectors[f"{kind}.{shift}.first"] = np.tile(first.reshape(-1), layout.payload_repetitions)
        vectors[f"{kind}.{shift}.second"] = np.tile(second.reshape(-1), layout.payload_repetitions)
    return vectors


def _align_owned(cipher, target_state, *, crypto_context):
    aligned = fhe.align_to(cipher, target_state, crypto_context.context)
    return cipher.deep_copy() if aligned is cipher else aligned


def _fast_rotations(cipher, offsets: tuple[int, ...], *, crypto_context) -> tuple[object, ...]:
    batch = crypto_context.fhe.fast_rotate(cipher, offsets, crypto_context.context)
    return tuple(crypto_context.fhe.unpack_cipher_batch(batch))


def _two_masked_rotations(
    first_rotated,
    second_rotated,
    *,
    first_plain,
    second_plain,
    crypto_context,
):
    first = _mask_rescale(first_rotated, first_plain, crypto_context=crypto_context)
    second = _mask_rescale(second_rotated, second_plain, crypto_context=crypto_context)
    return _add_owned(first, second, crypto_context=crypto_context)


def _two_rotation_masked_shift(
    cipher,
    *,
    first_rotation: int,
    second_rotation: int,
    first_plain,
    second_plain,
    crypto_context,
):
    first_rotated = crypto_context.fhe.homo_rotate(cipher, int(first_rotation), crypto_context.context)
    second_rotated = crypto_context.fhe.homo_rotate(cipher, int(second_rotation), crypto_context.context)
    try:
        first = _mask_rescale(first_rotated, first_plain, crypto_context=crypto_context)
        second = _mask_rescale(second_rotated, second_plain, crypto_context=crypto_context)
    finally:
        release_if_supported(first_rotated)
        release_if_supported(second_rotated)
    return _add_owned(first, second, crypto_context=crypto_context)


def _validate_rotation_strategy(mode: str, chunk_size: int) -> tuple[str, int]:
    mode = str(mode).lower()
    if mode not in {"loop", "fast"}:
        raise ValueError(f"unsupported rotation mode={mode!r}; expected 'loop' or 'fast'.")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"rotation_chunk_size must be positive, got {chunk_size}.")
    # QK/PV request two K/V rotations per shift.  EasyFHE's current broadcast
    # kernel supports the resulting 32-entry batch but rejects 64 entries.
    if mode == "fast" and chunk_size > 16:
        raise ValueError(
            f"rotation_chunk_size={chunk_size} exceeds the current safe maximum of 16."
        )
    return mode, chunk_size


def _finalize_triplet_accumulators(
    accumulators: list[object | None],
    *,
    crypto_context,
    extract_real: bool,
) -> tuple[object, ...]:
    """Relinearize/rescale independent lazy accumulators.

    Inputs are consumed on success and on failure.
    """

    if any(cipher is None for cipher in accumulators):
        raise RuntimeError("triplet finalization received an empty accumulator")
    owned = tuple(accumulators)  # type: ignore[arg-type]
    try:
        values: list[object] = []
        try:
            for accumulator in owned:
                item_relin = crypto_context.fhe.homo_relinearize(
                    accumulator, crypto_context.context
                )
                try:
                    item_scaled = crypto_context.fhe.rescale(
                        item_relin, crypto_context.context
                    )
                finally:
                    release_if_supported(item_relin)
                if bool(extract_real):
                    item_conjugated = crypto_context.fhe.homo_rotate(
                        item_scaled,
                        int(crypto_context.context.M) - 1,
                        crypto_context.context,
                    )
                    try:
                        item_output = crypto_context.fhe.homo_add(
                            item_scaled,
                            item_conjugated,
                            crypto_context.context,
                        )
                    finally:
                        release_if_supported(item_conjugated)
                        release_if_supported(item_scaled)
                else:
                    item_output = item_scaled
                values.append(item_output)
            outputs = tuple(values)
            values = []
            return outputs
        finally:
            for value in values:
                release_if_supported(value)
    finally:
        for index, accumulator in enumerate(accumulators):
            release_if_supported(accumulator)
            accumulators[index] = None
