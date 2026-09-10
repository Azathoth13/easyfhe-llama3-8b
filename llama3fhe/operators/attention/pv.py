"""The Δ-native PV kernel: attention output from probabilities and V_Δ.

One public entry, ``pv_delta_fhe``, plus the two fold helpers that pack the
Δ halves into complex components so PV runs on half the shifts. The lazy
paths accumulate degree-2 triplets and relinearize once per accumulator.
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


def _validate_pv_inputs(
    probability_ciphers: tuple[object, ...],
    value_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
) -> None:
    if len(probability_ciphers) != layout.query_pair_count:
        raise ValueError(f"PV expects {layout.query_pair_count} P ciphers, got {len(probability_ciphers)}.")
    if len(value_delta_ciphers) != layout.key_value_pair_count:
        raise ValueError(f"PV expects {layout.key_value_pair_count} V ciphers, got {len(value_delta_ciphers)}.")


def fold_probability_delta_halves_fhe(
    probability_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    target_limbs: int | None = None,
) -> tuple[object, ...]:
    """Build ``0.5*(P[d] + i*P[d+S/2])`` complex PV carriers.

    This is the legal complex boundary after the last elementwise Softmax
    nonlinearity.  The returned ciphertexts are one level below the supplied
    ordinary-real ``P_Delta`` ciphertexts.
    """

    if len(probability_ciphers) != int(layout.query_pair_count):
        raise ValueError(
            f"P fold expects {layout.query_pair_count} ciphers, got "
            f"{len(probability_ciphers)}."
        )
    seq_len = int(layout.seq_len)
    if seq_len % 2:
        raise ValueError("complex PV delta folding requires even seq_len.")
    if not probability_ciphers:
        return ()
    available = min(int(cipher.state.cur_limbs) - 1 for cipher in probability_ciphers)
    target_limbs = available if target_limbs is None else int(target_limbs)
    if target_limbs <= 1 or target_limbs > available:
        raise ValueError(
            f"invalid P-fold target_limbs={target_limbs}; available={available}."
        )

    sources: tuple[object, ...] = ()
    scaled: tuple[object, ...] = ()
    source_parts: list[object] = []
    scaled_parts: list[object] = []
    outputs: list[object] = []
    success = False
    try:
        for cipher in probability_ciphers:
            source_parts.append(
                _align_owned(
                    cipher,
                    _level_one_scale_state(cipher, target_limbs + 1),
                    crypto_context=crypto_context,
                )
            )
        sources = tuple(source_parts)
        source_parts = []
        scale_bundle = _raw_bundle(
            {
                "fold.probability_half": np.full(
                    (int(layout.slots),), 0.5, dtype=np.float64
                )
            },
            crypto_context=crypto_context,
        )
        half_plain = _bundle_plain(
            scale_bundle,
            "fold.probability_half",
            sources[0],
            crypto_context=crypto_context,
        )
        try:
            for source in sources:
                scaled_parts.append(
                    _mask_rescale(source, half_plain, crypto_context=crypto_context)
                )
            scaled = tuple(scaled_parts)
            scaled_parts = []
        finally:
            release_if_supported(half_plain)
            for source in sources:
                release_if_supported(source)
            sources = ()

        half_rotation = 2 * seq_len * (seq_len // 2)
        for probability in scaled:
            rotated = crypto_context.fhe.homo_rotate(
                probability,
                half_rotation,
                crypto_context.context,
            )
            imaginary = fhe.homo_mul_i(rotated, crypto_context.context)
            try:
                outputs.append(
                    crypto_context.fhe.homo_add(
                        probability, imaginary, crypto_context.context
                    )
                )
            finally:
                release_if_supported(rotated)
                release_if_supported(imaginary)
        success = True
        return tuple(outputs)
    finally:
        for cipher in sources + scaled + tuple(source_parts) + tuple(scaled_parts):
            release_if_supported(cipher)
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)


def fold_value_delta_halves_fhe(
    value_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    target_limbs: int | None = None,
    masks=None,
) -> tuple[object, ...]:
    """Build ``V_Delta[d,q] - i*V_Delta[d,q+S/2]`` PV carriers."""

    if len(value_delta_ciphers) != int(layout.key_value_pair_count):
        raise ValueError(
            f"V fold expects {layout.key_value_pair_count} ciphers, got "
            f"{len(value_delta_ciphers)}."
        )
    seq_len = int(layout.seq_len)
    if seq_len % 2:
        raise ValueError("complex PV delta folding requires even seq_len.")
    if not value_delta_ciphers:
        return ()
    available = min(int(cipher.state.cur_limbs) - 1 for cipher in value_delta_ciphers)
    target_limbs = available if target_limbs is None else int(target_limbs)
    if target_limbs <= 1 or target_limbs > available:
        raise ValueError(
            f"invalid V-fold target_limbs={target_limbs}; available={available}."
        )

    mask_bundle = (
        _raw_bundle(_mask_vectors(layout, kind="v"), crypto_context=crypto_context)
        if masks is None
        else masks
    )
    values: tuple[object, ...] = ()
    shift_sources: tuple[object, ...] = ()
    value_parts: list[object] = []
    shift_parts: list[object] = []
    outputs: list[object] = []
    success = False
    try:
        for cipher in value_delta_ciphers:
            value_parts.append(
                _align_owned(
                    cipher,
                    _level_one_scale_state(cipher, target_limbs),
                    crypto_context=crypto_context,
                )
            )
        values = tuple(value_parts)
        value_parts = []
        for cipher in value_delta_ciphers:
            shift_parts.append(
                _align_owned(
                    cipher,
                    _level_one_scale_state(cipher, target_limbs + 1),
                    crypto_context=crypto_context,
                )
            )
        shift_sources = tuple(shift_parts)
        shift_parts = []
        half = seq_len // 2
        first_plain = second_plain = None
        try:
            first_plain = _bundle_plain(
                mask_bundle,
                f"v.{half}.first",
                shift_sources[0],
                crypto_context=crypto_context,
            )
            second_plain = _bundle_plain(
                mask_bundle,
                f"v.{half}.second",
                shift_sources[0],
                crypto_context=crypto_context,
            )
            for value, shift_source in zip(values, shift_sources, strict=True):
                paired_half = _two_rotation_masked_shift(
                    shift_source,
                    first_rotation=2 * half,
                    second_rotation=-2 * (seq_len - half),
                    first_plain=first_plain,
                    second_plain=second_plain,
                    crypto_context=crypto_context,
                )
                negative_imaginary = fhe.homo_mul_i(
                    paired_half,
                    crypto_context.context,
                    negative=True,
                )
                try:
                    outputs.append(
                        crypto_context.fhe.homo_add(
                            value, negative_imaginary, crypto_context.context
                        )
                    )
                finally:
                    release_if_supported(paired_half)
                    release_if_supported(negative_imaginary)
        finally:
            if first_plain is not None:
                release_if_supported(first_plain)
            if second_plain is not None:
                release_if_supported(second_plain)
        success = True
        return tuple(outputs)
    finally:
        for cipher in (
            values
            + shift_sources
            + tuple(value_parts)
            + tuple(shift_parts)
        ):
            release_if_supported(cipher)
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)


def pv_delta_fhe(
    probability_ciphers: tuple[object, ...],
    value_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    operator_config: AttentionOperatorConfig | None = None,
    inputs_pre_folded: bool = False,
) -> tuple[object, ...]:
    """Evaluate direct Delta PV with optional lazy triplet accumulation.

    ``inputs_pre_folded`` says the caller already folded both the probability
    and the value halves (the prefold-after-softmax schedule); it is a runtime
    fact about the ciphertexts, not a policy, so it stays a parameter.
    """

    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    lazy_relinearization = bool(operator_config.pv_lazy_relinearization)
    rotation_mode = str(operator_config.rotation_mode)
    rotation_chunk_size = int(operator_config.rotation_chunk_size)
    complex_delta_folding = bool(operator_config.pv_complex_delta_folding)
    probability_pre_folded = value_pre_folded = bool(inputs_pre_folded)

    if not bool(lazy_relinearization):
        if (
            bool(complex_delta_folding)
            or bool(probability_pre_folded)
            or bool(value_pre_folded)
        ):
            raise ValueError(
                "complex PV delta folding requires lazy_relinearization=True."
            )
    if (bool(probability_pre_folded) or bool(value_pre_folded)) and not bool(
        complex_delta_folding
    ):
        raise ValueError("pre-folded P/V requires complex PV delta folding.")
    if not bool(lazy_relinearization):
        return _pv_delta_fhe_eager(
            probability_ciphers,
            value_delta_ciphers,
            layout=layout,
            crypto_context=crypto_context,
        )
    if bool(complex_delta_folding):
        return _pv_delta_fhe_lazy_complex_fold(
            probability_ciphers,
            value_delta_ciphers,
            layout=layout,
            crypto_context=crypto_context,
            rotation_mode=rotation_mode,
            rotation_chunk_size=rotation_chunk_size,
            probability_pre_folded=bool(probability_pre_folded),
            value_pre_folded=bool(value_pre_folded),
            )
    return _pv_delta_fhe_lazy(
        probability_ciphers,
        value_delta_ciphers,
        layout=layout,
        crypto_context=crypto_context,
        rotation_mode=rotation_mode,
        rotation_chunk_size=rotation_chunk_size,
    )


def _pv_delta_fhe_eager(
    probability_ciphers: tuple[object, ...],
    value_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    _validate_pv_inputs(probability_ciphers, value_delta_ciphers, layout=layout)
    masks = _raw_bundle(_mask_vectors(layout, kind="v"), crypto_context=crypto_context)
    accumulators: list[object | None] = [None] * layout.query_pair_count
    seq_len = int(layout.seq_len)
    try:
        for shift in range(seq_len):
            first_plain = second_plain = None
            if shift:
                first_plain = _bundle_plain(masks, f"v.{shift}.first", value_delta_ciphers[0], crypto_context=crypto_context)
                second_plain = _bundle_plain(masks, f"v.{shift}.second", value_delta_ciphers[0], crypto_context=crypto_context)
            try:
                for kv_pair, value_cipher in enumerate(value_delta_ciphers):
                    if shift:
                        value_shifted = _two_rotation_masked_shift(
                            value_cipher,
                            first_rotation=2 * shift,
                            second_rotation=-2 * (seq_len - shift),
                            first_plain=first_plain,
                            second_plain=second_plain,
                            crypto_context=crypto_context,
                        )
                    else:
                        value_shifted = value_cipher
                    try:
                        for offset in range(layout.gqa_ratio):
                            probability_index = kv_pair * layout.gqa_ratio + offset
                            probability_cipher = probability_ciphers[probability_index]
                            probability_shifted = (
                                probability_cipher
                                if shift == 0
                                else crypto_context.fhe.homo_rotate(
                                    probability_cipher, 2 * seq_len * shift, crypto_context.context
                                )
                            )
                            try:
                                term = _mul_rescale(
                                    probability_shifted, value_shifted, crypto_context=crypto_context
                                )
                            finally:
                                if probability_shifted is not probability_cipher:
                                    release_if_supported(probability_shifted)
                            current = accumulators[probability_index]
                            accumulators[probability_index] = (
                                term if current is None else _add_owned(current, term, crypto_context=crypto_context)
                            )
                    finally:
                        if value_shifted is not value_cipher:
                            release_if_supported(value_shifted)
            finally:
                if first_plain is not None:
                    release_if_supported(first_plain)
                if second_plain is not None:
                    release_if_supported(second_plain)
        if any(cipher is None for cipher in accumulators):
            raise RuntimeError("PV produced an empty output accumulator.")
        outputs = tuple(accumulators)  # type: ignore[arg-type]
        accumulators = [None] * layout.query_pair_count
        return outputs
    finally:
        for cipher in accumulators:
            if cipher is not None:
                release_if_supported(cipher)


def _pv_delta_fhe_lazy_complex_fold(
    probability_ciphers: tuple[object, ...],
    value_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    rotation_mode: str,
    rotation_chunk_size: int,
    probability_pre_folded: bool,
    value_pre_folded: bool,
) -> tuple[object, ...]:
    """Evaluate PV by folding delta halves into CKKS real/imaginary parts.

    For ``h = seq_len / 2``, construct the two reusable streams

    ``0.5*(P[d,q] + i P[d+h,q])`` and
    ``V[d,q] - i V[d,q+h]``.

    The real part of their product is the sum of the two wanted diagonal
    products.  Thus only ``h`` lazy triplet products are accumulated per query
    pair, followed by one relinearization/rescale and one conjugate-add.
    """

    _validate_pv_inputs(probability_ciphers, value_delta_ciphers, layout=layout)
    rotation_mode, rotation_chunk_size = _validate_rotation_strategy(
        rotation_mode, rotation_chunk_size
    )
    seq_len = int(layout.seq_len)
    if seq_len % 2:
        raise ValueError("complex PV delta folding requires even seq_len.")
    half = seq_len // 2
    probability_prep_levels = 0 if bool(probability_pre_folded) else 1
    value_prep_levels = 0 if bool(value_pre_folded) else 1
    target_limbs = min(
        min(
            int(cipher.state.cur_limbs) - probability_prep_levels
            for cipher in probability_ciphers
        ),
        min(
            int(cipher.state.cur_limbs) - value_prep_levels
            for cipher in value_delta_ciphers
        ),
    )
    if target_limbs <= 1:
        raise ValueError("folded lazy Delta PV requires enough limbs for a final rescale.")

    folded_probabilities: tuple[object, ...] = ()
    folded_values: tuple[object, ...] = ()
    masks = _raw_bundle(
        _mask_vectors(layout, kind="v"), crypto_context=crypto_context
    )
    accumulators: list[object | None] = [None] * layout.query_pair_count
    outputs: list[object] = []
    success = False
    try:
        if bool(probability_pre_folded):
            folded_probabilities = tuple(
                _align_owned(
                    cipher,
                    _level_one_scale_state(cipher, target_limbs),
                    crypto_context=crypto_context,
                )
                for cipher in probability_ciphers
            )
        else:
            folded_probabilities = fold_probability_delta_halves_fhe(
                probability_ciphers,
                layout=layout,
                crypto_context=crypto_context,
                target_limbs=target_limbs,
            )

        if bool(value_pre_folded):
            folded_values = tuple(
                _align_owned(
                    cipher,
                    _level_one_scale_state(cipher, target_limbs),
                    crypto_context=crypto_context,
                )
                for cipher in value_delta_ciphers
            )
        else:
            folded_values = fold_value_delta_halves_fhe(
                value_delta_ciphers,
                layout=layout,
                crypto_context=crypto_context,
                target_limbs=target_limbs,
                masks=masks,
            )

        for kv_pair, folded_value in enumerate(folded_values):
            for offset in range(layout.gqa_ratio):
                probability_index = kv_pair * layout.gqa_ratio + offset
                accumulators[probability_index] = fhe.homo_mul_no_relin(
                    folded_probabilities[probability_index],
                    folded_value,
                    crypto_context.context,
                )

        chunk_size = 1 if rotation_mode == "loop" else rotation_chunk_size
        for chunk_start in range(1, half, chunk_size):
            shifts = tuple(range(chunk_start, min(half, chunk_start + chunk_size)))
            probability_rotations: list[tuple[object, ...]] = []
            value_rotations: list[tuple[object, ...]] = []
            try:
                for probability in folded_probabilities:
                    offsets = tuple(2 * seq_len * shift for shift in shifts)
                    probability_rotations.append(
                        (
                            crypto_context.fhe.homo_rotate(
                                probability, offsets[0], crypto_context.context
                            ),
                        )
                        if rotation_mode == "loop"
                        else _fast_rotations(
                            probability, offsets, crypto_context=crypto_context
                        )
                    )
                for value in folded_values:
                    value_stride = 2
                    offsets = tuple(
                        rotation
                        for shift in shifts
                        for rotation in (
                            value_stride * shift,
                            -value_stride * (seq_len - shift),
                        )
                    )
                    value_rotations.append(
                        tuple(
                            crypto_context.fhe.homo_rotate(
                                value, rotation, crypto_context.context
                            )
                            for rotation in offsets
                        )
                        if rotation_mode == "loop"
                        else _fast_rotations(value, offsets, crypto_context=crypto_context)
                    )

                for local_index, shift in enumerate(shifts):
                    first_plain = _bundle_plain(
                        masks,
                        f"v.{shift}.first",
                        folded_values[0],
                        crypto_context=crypto_context,
                    )
                    second_plain = _bundle_plain(
                        masks,
                        f"v.{shift}.second",
                        folded_values[0],
                        crypto_context=crypto_context,
                    )
                    try:
                        for kv_pair in range(layout.key_value_pair_count):
                            value_shifted = _two_masked_rotations(
                                value_rotations[kv_pair][2 * local_index],
                                value_rotations[kv_pair][2 * local_index + 1],
                                first_plain=first_plain,
                                second_plain=second_plain,
                                crypto_context=crypto_context,
                            )
                            try:
                                for offset in range(layout.gqa_ratio):
                                    probability_index = (
                                        kv_pair * layout.gqa_ratio + offset
                                    )
                                    term = fhe.homo_mul_no_relin(
                                        probability_rotations[probability_index][
                                            local_index
                                        ],
                                        value_shifted,
                                        crypto_context.context,
                                    )
                                    current = accumulators[probability_index]
                                    if current is None:
                                        raise RuntimeError(
                                            "PV triplet accumulator was not initialized."
                                        )
                                    accumulators[probability_index] = _add_owned(
                                        current, term, crypto_context=crypto_context
                                    )
                            finally:
                                release_if_supported(value_shifted)
                    finally:
                        release_if_supported(first_plain)
                        release_if_supported(second_plain)
            finally:
                for rotations in probability_rotations + value_rotations:
                    for rotated in rotations:
                        release_if_supported(rotated)

        outputs.extend(
            _finalize_triplet_accumulators(
                accumulators,
                crypto_context=crypto_context,
                extract_real=True,
            )
        )
        success = True
        return tuple(outputs)
    finally:
        for cipher in folded_probabilities + folded_values:
            release_if_supported(cipher)
        for accumulator in accumulators:
            if accumulator is not None:
                release_if_supported(accumulator)
        if not success:
            for output in outputs:
                release_if_supported(output)


def _pv_delta_fhe_lazy(
    probability_ciphers: tuple[object, ...],
    value_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    rotation_mode: str,
    rotation_chunk_size: int,
) -> tuple[object, ...]:
    _validate_pv_inputs(probability_ciphers, value_delta_ciphers, layout=layout)
    rotation_mode, rotation_chunk_size = _validate_rotation_strategy(
        rotation_mode, rotation_chunk_size
    )
    seq_len = int(layout.seq_len)
    target_limbs = min(
        min(int(cipher.state.cur_limbs) for cipher in probability_ciphers),
        min(int(cipher.state.cur_limbs) - 1 for cipher in value_delta_ciphers),
    )
    if target_limbs <= 1:
        raise ValueError("lazy Delta PV requires enough limbs for a final rescale.")
    prepared_probabilities: tuple[object, ...] = ()
    prepared_values: tuple[object, ...] = ()
    prepared_shift_values: tuple[object, ...] = ()
    masks = _raw_bundle(_mask_vectors(layout, kind="v"), crypto_context=crypto_context)
    accumulators: list[object | None] = [None] * layout.query_pair_count
    outputs: list[object] = []
    success = False
    try:
        prepared_probabilities = tuple(
            _align_owned(
                cipher,
                _level_one_scale_state(cipher, target_limbs),
                crypto_context=crypto_context,
            )
            for cipher in probability_ciphers
        )
        prepared_values = tuple(
            _align_owned(
                cipher,
                _level_one_scale_state(cipher, target_limbs),
                crypto_context=crypto_context,
            )
            for cipher in value_delta_ciphers
        )
        # Nonzero V shifts consume one mask/rescale level.  Drop the source to
        # exactly one limb above the CT-CT target before hoisted rotations;
        # rotating the original early-layer V at production depth otherwise
        # materializes enormous 32-way batches and repeats modulus alignment.
        prepared_shift_values = tuple(
            _align_owned(
                cipher,
                _level_one_scale_state(cipher, target_limbs + 1),
                crypto_context=crypto_context,
            )
            for cipher in value_delta_ciphers
        )

        for kv_pair, prepared_value in enumerate(prepared_values):
            for offset in range(layout.gqa_ratio):
                probability_index = kv_pair * layout.gqa_ratio + offset
                term = fhe.homo_mul_no_relin(
                    prepared_probabilities[probability_index],
                    prepared_value,
                    crypto_context.context,
                )
                accumulators[probability_index] = term

        chunk_size = 1 if rotation_mode == "loop" else rotation_chunk_size
        for chunk_start in range(1, seq_len, chunk_size):
            shifts = tuple(range(chunk_start, min(seq_len, chunk_start + chunk_size)))
            probability_rotations: list[tuple[object, ...]] = []
            value_rotations: list[tuple[object, ...]] = []
            try:
                for prepared_probability in prepared_probabilities:
                    offsets = tuple(2 * seq_len * shift for shift in shifts)
                    probability_rotations.append(
                        (
                            crypto_context.fhe.homo_rotate(
                                prepared_probability, offsets[0], crypto_context.context
                            ),
                        )
                        if rotation_mode == "loop"
                        else _fast_rotations(
                            prepared_probability, offsets, crypto_context=crypto_context
                        )
                    )
                for value_cipher in prepared_shift_values:
                    offsets = tuple(
                        rotation
                        for shift in shifts
                        for rotation in (2 * shift, -2 * (seq_len - shift))
                    )
                    value_rotations.append(
                        tuple(
                            crypto_context.fhe.homo_rotate(
                                value_cipher, rotation, crypto_context.context
                            )
                            for rotation in offsets
                        )
                        if rotation_mode == "loop"
                        else _fast_rotations(value_cipher, offsets, crypto_context=crypto_context)
                    )

                for local_index, shift in enumerate(shifts):
                    first_plain = _bundle_plain(
                        masks,
                        f"v.{shift}.first",
                        prepared_shift_values[0],
                        crypto_context=crypto_context,
                    )
                    second_plain = _bundle_plain(
                        masks,
                        f"v.{shift}.second",
                        prepared_shift_values[0],
                        crypto_context=crypto_context,
                    )
                    try:
                        for kv_pair in range(layout.key_value_pair_count):
                            value_shifted = _two_masked_rotations(
                                value_rotations[kv_pair][2 * local_index],
                                value_rotations[kv_pair][2 * local_index + 1],
                                first_plain=first_plain,
                                second_plain=second_plain,
                                crypto_context=crypto_context,
                            )
                            try:
                                for offset in range(layout.gqa_ratio):
                                    probability_index = kv_pair * layout.gqa_ratio + offset
                                    term = fhe.homo_mul_no_relin(
                                        probability_rotations[probability_index][local_index],
                                        value_shifted,
                                        crypto_context.context,
                                    )
                                    current = accumulators[probability_index]
                                    if current is None:
                                        raise RuntimeError("PV triplet accumulator was not initialized.")
                                    accumulators[probability_index] = _add_owned(
                                        current, term, crypto_context=crypto_context
                                    )
                            finally:
                                release_if_supported(value_shifted)
                    finally:
                        release_if_supported(first_plain)
                        release_if_supported(second_plain)
            finally:
                for rotations in probability_rotations + value_rotations:
                    for rotated in rotations:
                        release_if_supported(rotated)

        outputs.extend(
            _finalize_triplet_accumulators(
                accumulators,
                crypto_context=crypto_context,
                extract_real=False,
            )
        )
        success = True
        return tuple(outputs)
    finally:
        for cipher in prepared_probabilities + prepared_values + prepared_shift_values:
            release_if_supported(cipher)
        for accumulator in accumulators:
            if accumulator is not None:
                release_if_supported(accumulator)
        if not success:
            for output in outputs:
                release_if_supported(output)


__all__ = [
    "fold_probability_delta_halves_fhe",
    "fold_value_delta_halves_fhe",
    "pv_delta_fhe",
]
