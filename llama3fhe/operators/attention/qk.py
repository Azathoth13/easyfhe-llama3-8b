"""The Δ-native QK kernel: scores from Q_C and K_Δ carriers.

One public entry, ``qk_delta_fhe``, with an eager path and the production
lazy-relinearization path (accumulate degree-2 products, relinearize once).
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
    bundle_plain as _bundle_plain,
)
from ..primitives import (
    mul_cipher_rescale as _mul_rescale,
)
from ..primitives import (
    mul_plain_rescale as _mask_rescale,
)
from .config import AttentionOperatorConfig
from .kernel_helpers import (
    _align_owned,
    _fast_rotations,
    _finalize_triplet_accumulators,
    _level_one_scale_state,
    _mask_vectors,
    _raw_bundle,
    _two_masked_rotations,
    _two_rotation_masked_shift,
    _validate_rotation_strategy,
)


def _validate_qk_inputs(
    query_ciphers: tuple[object, ...],
    key_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
) -> None:
    if len(query_ciphers) != layout.query_pair_count:
        raise ValueError(f"QK expects {layout.query_pair_count} Q ciphers, got {len(query_ciphers)}.")
    if len(key_delta_ciphers) != layout.key_value_pair_count:
        raise ValueError(f"QK expects {layout.key_value_pair_count} K ciphers, got {len(key_delta_ciphers)}.")


def qk_delta_fhe(
    query_ciphers: tuple[object, ...],
    key_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    operator_config: AttentionOperatorConfig | None = None,
) -> tuple[object, ...]:
    """Evaluate direct Delta QK.

    The default path keeps each CT-CT product as a three-component triplet,
    accumulates all feature terms for one score ciphertext, and performs one
    final relinearization/rescale.  ``lazy_relinearization=False`` retains the
    original bring-up kernel for correctness/performance comparisons.  The
    optional complex feature fold pairs dimensions ``d`` and ``d+d_h/2`` in
    the real/imaginary components.  It halves the triplet products and main
    rotations while extracting the same real ``S_Delta`` output in place.
    """

    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    lazy_relinearization = bool(operator_config.qk_lazy_relinearization)
    rotation_mode = str(operator_config.rotation_mode)
    rotation_chunk_size = int(operator_config.rotation_chunk_size)
    output_level = operator_config.qk_output_level
    complex_feature_folding = bool(operator_config.qk_complex_feature_folding)
    key_features_pre_folded = bool(operator_config.qk_key_features_pre_folded)
    query_features_pre_scaled = bool(
        operator_config.qk_query_features_pre_scaled
    )
    query_features_pre_folded = bool(
        operator_config.qk_query_features_pre_folded
    )

    if bool(key_features_pre_folded) and not bool(complex_feature_folding):
        raise ValueError("pre-folded K requires complex QK feature folding.")
    if bool(query_features_pre_scaled) and not bool(complex_feature_folding):
        raise ValueError("pre-scaled Q requires complex QK feature folding.")
    if bool(query_features_pre_folded) and not bool(complex_feature_folding):
        raise ValueError("pre-folded Q requires complex QK feature folding.")
    if bool(complex_feature_folding) and int(layout.head_dim) % 2:
        raise ValueError("complex QK feature folding requires even head_dim.")
    if not bool(lazy_relinearization):
        if bool(complex_feature_folding):
            raise ValueError(
                "complex QK feature folding requires lazy_relinearization=True."
            )
        if output_level is not None:
            raise ValueError("QK output_level requires lazy_relinearization=True.")
        return _qk_delta_fhe_eager(
            query_ciphers,
            key_delta_ciphers,
            layout=layout,
            crypto_context=crypto_context,
        )
    return _qk_delta_fhe_lazy(
        query_ciphers,
        key_delta_ciphers,
        layout=layout,
        crypto_context=crypto_context,
        rotation_mode=rotation_mode,
        rotation_chunk_size=rotation_chunk_size,
        output_level=output_level,
        complex_feature_folding=bool(complex_feature_folding),
        key_features_pre_folded=bool(key_features_pre_folded),
        query_features_pre_scaled=bool(query_features_pre_scaled),
        query_features_pre_folded=bool(query_features_pre_folded),
    )


def _qk_delta_fhe_eager(
    query_ciphers: tuple[object, ...],
    key_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    _validate_qk_inputs(query_ciphers, key_delta_ciphers, layout=layout)
    masks = _raw_bundle(_mask_vectors(layout, kind="k"), crypto_context=crypto_context)
    accumulators: list[object | None] = [None] * layout.query_pair_count
    seq_len = int(layout.seq_len)
    try:
        for shift in range(seq_len):
            first_plain = second_plain = None
            if shift:
                first_plain = _bundle_plain(masks, f"k.{shift}.first", key_delta_ciphers[0], crypto_context=crypto_context)
                second_plain = _bundle_plain(masks, f"k.{shift}.second", key_delta_ciphers[0], crypto_context=crypto_context)
            try:
                for kv_pair, key_cipher in enumerate(key_delta_ciphers):
                    if shift:
                        key_shifted = _two_rotation_masked_shift(
                            key_cipher,
                            first_rotation=2 * shift * (seq_len - 1),
                            second_rotation=2 * (shift * (seq_len - 1) + seq_len),
                            first_plain=first_plain,
                            second_plain=second_plain,
                            crypto_context=crypto_context,
                        )
                    else:
                        key_shifted = key_cipher
                    try:
                        for offset in range(layout.gqa_ratio):
                            query_index = kv_pair * layout.gqa_ratio + offset
                            query_cipher = query_ciphers[query_index]
                            query_shifted = (
                                query_cipher
                                if shift == 0
                                else crypto_context.fhe.homo_rotate(
                                    query_cipher, 2 * seq_len * shift, crypto_context.context
                                )
                            )
                            try:
                                term = _mul_rescale(query_shifted, key_shifted, crypto_context=crypto_context)
                            finally:
                                if query_shifted is not query_cipher:
                                    release_if_supported(query_shifted)
                            current = accumulators[query_index]
                            accumulators[query_index] = (
                                term if current is None else _add_owned(current, term, crypto_context=crypto_context)
                            )
                    finally:
                        if key_shifted is not key_cipher:
                            release_if_supported(key_shifted)
            finally:
                if first_plain is not None:
                    release_if_supported(first_plain)
                if second_plain is not None:
                    release_if_supported(second_plain)
        if any(cipher is None for cipher in accumulators):
            raise RuntimeError("QK produced an empty score accumulator.")
        outputs = tuple(accumulators)  # type: ignore[arg-type]
        accumulators = [None] * layout.query_pair_count
        return outputs
    finally:
        for cipher in accumulators:
            if cipher is not None:
                release_if_supported(cipher)


def _qk_delta_fhe_lazy(
    query_ciphers: tuple[object, ...],
    key_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    rotation_mode: str,
    rotation_chunk_size: int,
    output_level: int | None,
    complex_feature_folding: bool,
    key_features_pre_folded: bool,
    query_features_pre_scaled: bool,
    query_features_pre_folded: bool,
) -> tuple[object, ...]:
    _validate_qk_inputs(query_ciphers, key_delta_ciphers, layout=layout)
    rotation_mode, rotation_chunk_size = _validate_rotation_strategy(
        rotation_mode, rotation_chunk_size
    )
    seq_len = int(layout.seq_len)
    feature_folding = bool(complex_feature_folding)
    key_pre_folded = bool(key_features_pre_folded)
    query_pre_scaled = bool(query_features_pre_scaled)
    query_pre_folded = bool(query_features_pre_folded)
    # Every nonzero K torus shift consumes one plaintext level.  Unless the
    # transpose already supplied 0.5*Q, folded Q is scaled from one level above
    # target. Both K halves use the normal one-level-higher shift source, so
    # either form is depth-neutral at the QK output.
    query_prep_levels = 1 if feature_folding and not query_pre_scaled else 0
    target_limbs = min(
        min(
            int(cipher.state.cur_limbs) - query_prep_levels
            for cipher in query_ciphers
        ),
        min(
            int(cipher.state.cur_limbs) - 1
            for cipher in key_delta_ciphers
        ),
    )
    if output_level is not None:
        output_level = int(output_level)
        if output_level <= 0:
            raise ValueError(f"QK output_level must be positive, got {output_level}.")
        requested_operand_limbs = int(crypto_context.context.L) - (output_level - 1)
        target_limbs = min(target_limbs, requested_operand_limbs)
    if target_limbs <= 1:
        raise ValueError("lazy Delta QK requires enough limbs for a final rescale.")
    prepared_queries: tuple[object, ...] = ()
    prepared_keys: tuple[object, ...] = ()
    prepared_shift_keys: tuple[object, ...] = ()
    folded_initial_keys: tuple[object, ...] = ()
    masks = _raw_bundle(_mask_vectors(layout, kind="k"), crypto_context=crypto_context)
    accumulators: list[object | None] = [None] * layout.query_pair_count
    outputs: list[object] = []
    success = False
    try:
        if feature_folding:
            half = int(layout.head_dim) // 2
            if query_pre_scaled:
                prepared_queries = tuple(
                    _align_owned(
                        cipher,
                        _level_one_scale_state(cipher, target_limbs),
                        crypto_context=crypto_context,
                    )
                    for cipher in query_ciphers
                )
            else:
                query_sources = tuple(
                    _align_owned(
                        cipher,
                        _level_one_scale_state(cipher, target_limbs + 1),
                        crypto_context=crypto_context,
                    )
                    for cipher in query_ciphers
                )
                query_scale_bundle = _raw_bundle(
                    {
                        "fold.query_half": np.full(
                            (int(layout.slots),), 0.5, dtype=np.float64
                        )
                    },
                    crypto_context=crypto_context,
                )
                query_half_plain = _bundle_plain(
                    query_scale_bundle,
                    "fold.query_half",
                    query_sources[0],
                    crypto_context=crypto_context,
                )
                try:
                    prepared_queries = tuple(
                        _mask_rescale(
                            source,
                            query_half_plain,
                            crypto_context=crypto_context,
                        )
                        for source in query_sources
                    )
                finally:
                    release_if_supported(query_half_plain)
                    for source in query_sources:
                        release_if_supported(source)
            if not query_pre_folded:
                folded_queries: list[object] = []
                try:
                    for query in prepared_queries:
                        rotated = crypto_context.fhe.homo_rotate(
                            query,
                            2 * int(layout.seq_len) * half,
                            crypto_context.context,
                        )
                        imaginary = fhe.homo_mul_i(
                            rotated, crypto_context.context
                        )
                        try:
                            folded_queries.append(
                                crypto_context.fhe.homo_add(
                                    query, imaginary, crypto_context.context
                                )
                            )
                        finally:
                            release_if_supported(rotated)
                            release_if_supported(imaginary)
                    for query in prepared_queries:
                        release_if_supported(query)
                    prepared_queries = tuple(folded_queries)
                    folded_queries = []
                finally:
                    for query in folded_queries:
                        release_if_supported(query)
        else:
            prepared_queries = tuple(
                _align_owned(
                    cipher,
                    _level_one_scale_state(cipher, target_limbs),
                    crypto_context=crypto_context,
                )
                for cipher in query_ciphers
            )
        prepared_keys = tuple(
            _align_owned(
                cipher,
                _level_one_scale_state(cipher, target_limbs),
                crypto_context=crypto_context,
            )
            for cipher in key_delta_ciphers
        )
        prepared_shift_keys = tuple(
            _align_owned(
                cipher,
                _level_one_scale_state(cipher, target_limbs + 1),
                crypto_context=crypto_context,
            )
            for cipher in key_delta_ciphers
        )

        initial_keys = prepared_keys
        if feature_folding and not key_pre_folded:
            first_plain = _bundle_plain(
                masks,
                f"k.{half}.first",
                prepared_shift_keys[0],
                crypto_context=crypto_context,
            )
            second_plain = _bundle_plain(
                masks,
                f"k.{half}.second",
                prepared_shift_keys[0],
                crypto_context=crypto_context,
            )
            folded_keys: list[object] = []
            try:
                for key, shift_source in zip(prepared_keys, prepared_shift_keys):
                    second_half = _two_rotation_masked_shift(
                        shift_source,
                        first_rotation=2 * half * (seq_len - 1),
                        second_rotation=2 * (half * (seq_len - 1) + seq_len),
                        first_plain=first_plain,
                        second_plain=second_plain,
                        crypto_context=crypto_context,
                    )
                    negative_imaginary = fhe.homo_mul_i(
                        second_half,
                        crypto_context.context,
                        negative=True,
                    )
                    try:
                        folded_keys.append(
                            crypto_context.fhe.homo_add(
                                key, negative_imaginary, crypto_context.context
                            )
                        )
                    finally:
                        release_if_supported(second_half)
                        release_if_supported(negative_imaginary)
            finally:
                release_if_supported(first_plain)
                release_if_supported(second_plain)
            folded_initial_keys = tuple(folded_keys)
            initial_keys = folded_initial_keys

        # The unrotated diagonal is handled separately.  It also initializes
        # every triplet accumulator before batched nonzero rotations begin.
        for kv_pair, prepared_key in enumerate(initial_keys):
            for offset in range(layout.gqa_ratio):
                query_index = kv_pair * layout.gqa_ratio + offset
                term = fhe.homo_mul_no_relin(
                    prepared_queries[query_index],
                    prepared_key,
                    crypto_context.context,
                )
                accumulators[query_index] = term
        for key in folded_initial_keys:
            release_if_supported(key)
        folded_initial_keys = ()

        chunk_size = (
            1
            if rotation_mode == "loop"
            else min(
                rotation_chunk_size,
                16 if key_pre_folded or not feature_folding else 8,
            )
        )
        shift_count = int(layout.head_dim) // 2 if feature_folding else seq_len
        for chunk_start in range(1, shift_count, chunk_size):
            shifts = tuple(range(chunk_start, min(shift_count, chunk_start + chunk_size)))
            query_rotations: list[tuple[object, ...]] = []
            key_rotations: list[tuple[object, ...]] = []
            try:
                for prepared_query in prepared_queries:
                    offsets = tuple(2 * seq_len * shift for shift in shifts)
                    query_rotations.append(
                        (
                            crypto_context.fhe.homo_rotate(
                                prepared_query, offsets[0], crypto_context.context
                            ),
                        )
                        if rotation_mode == "loop"
                        else _fast_rotations(
                            prepared_query, offsets, crypto_context=crypto_context
                        )
                    )
                for key_cipher in prepared_shift_keys:
                    effective_shifts = (
                        tuple(
                            effective
                            for shift in shifts
                            for effective in (shift, shift + half)
                        )
                        if feature_folding and not key_pre_folded
                        else shifts
                    )
                    offsets = tuple(
                        rotation
                        for shift in effective_shifts
                        for rotation in (
                            2 * shift * (seq_len - 1),
                            2 * (shift * (seq_len - 1) + seq_len),
                        )
                    )
                    key_rotations.append(
                        tuple(
                            crypto_context.fhe.homo_rotate(
                                key_cipher, rotation, crypto_context.context
                            )
                            for rotation in offsets
                        )
                        if rotation_mode == "loop"
                        else _fast_rotations(key_cipher, offsets, crypto_context=crypto_context)
                    )

                for local_index, shift in enumerate(shifts):
                    effective_shifts = (
                        (shift, shift + half) if feature_folding else (shift,)
                    )
                    if key_pre_folded:
                        effective_shifts = (shift,)
                    plains = tuple(
                        (
                            _bundle_plain(
                                masks,
                                f"k.{effective}.first",
                                prepared_shift_keys[0],
                                crypto_context=crypto_context,
                            ),
                            _bundle_plain(
                                masks,
                                f"k.{effective}.second",
                                prepared_shift_keys[0],
                                crypto_context=crypto_context,
                            ),
                        )
                        for effective in effective_shifts
                    )
                    try:
                        for kv_pair in range(layout.key_value_pair_count):
                            key_base = (
                                4 * local_index
                                if feature_folding and not key_pre_folded
                                else 2 * local_index
                            )
                            first_shifted = _two_masked_rotations(
                                key_rotations[kv_pair][key_base],
                                key_rotations[kv_pair][key_base + 1],
                                first_plain=plains[0][0],
                                second_plain=plains[0][1],
                                crypto_context=crypto_context,
                            )
                            if feature_folding and not key_pre_folded:
                                second_shifted = _two_masked_rotations(
                                    key_rotations[kv_pair][key_base + 2],
                                    key_rotations[kv_pair][key_base + 3],
                                    first_plain=plains[1][0],
                                    second_plain=plains[1][1],
                                    crypto_context=crypto_context,
                                )
                                negative_imaginary = fhe.homo_mul_i(
                                    second_shifted,
                                    crypto_context.context,
                                    negative=True,
                                )
                                try:
                                    key_shifted = crypto_context.fhe.homo_add(
                                        first_shifted,
                                        negative_imaginary,
                                        crypto_context.context,
                                    )
                                finally:
                                    release_if_supported(first_shifted)
                                    release_if_supported(second_shifted)
                                    release_if_supported(negative_imaginary)
                            else:
                                key_shifted = first_shifted
                            try:
                                for offset in range(layout.gqa_ratio):
                                    query_index = kv_pair * layout.gqa_ratio + offset
                                    term = fhe.homo_mul_no_relin(
                                        query_rotations[query_index][local_index],
                                        key_shifted,
                                        crypto_context.context,
                                    )
                                    current = accumulators[query_index]
                                    if current is None:
                                        raise RuntimeError("QK triplet accumulator was not initialized.")
                                    accumulators[query_index] = _add_owned(
                                        current, term, crypto_context=crypto_context
                                    )
                            finally:
                                release_if_supported(key_shifted)
                    finally:
                        for first_plain, second_plain in plains:
                            release_if_supported(first_plain)
                            release_if_supported(second_plain)
            finally:
                for rotations in query_rotations + key_rotations:
                    for rotated in rotations:
                        release_if_supported(rotated)

        outputs.extend(
            _finalize_triplet_accumulators(
                accumulators,
                crypto_context=crypto_context,
                extract_real=bool(feature_folding),
            )
        )
        success = True
        return tuple(outputs)
    finally:
        for cipher in folded_initial_keys:
            release_if_supported(cipher)
        for cipher in prepared_queries + prepared_keys + prepared_shift_keys:
            release_if_supported(cipher)
        for accumulator in accumulators:
            if accumulator is not None:
                release_if_supported(accumulator)
        if not success:
            for output in outputs:
                release_if_supported(output)


__all__ = [
    "qk_delta_fhe",
]
