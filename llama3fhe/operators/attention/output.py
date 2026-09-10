from __future__ import annotations

"""PairFM-factored ``O_C -> feature-major`` conversion.

This module keeps the existing attention and public feature-major contracts
unchanged.  ``PairFM`` exists only as an internal, block-friendly layout::

    O_C[c, d, q, lane]
      -> PairFM[c, i=2*d+lane, q]
      -> blocked[c, token_shard, i, local_token]
      -> FM[token_shard, global_feature=(c, i), local_token]

Adjacent query-pair ciphertexts are carried in the real/imaginary components
through the two common slot permutations.  They are split only before the
final ciphertext-owner transpose.
"""

import os
import time
from dataclasses import asdict, dataclass
from math import ceil, gcd

import numpy as np

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import synchronize_device as sync_device
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout

from ..linear import pair_real_ciphers, split_complex_ciphers_twice
from ..primitives import (
    accumulate_owned as _add_owned,
)
from ..primitives import (
    drop_to_limbs,
)
from ..primitives import (
    encode_mask as _encode_mask,
)
from ..transform import apply_transform_by_level, transform_rotation_set
from .config import AttentionOperatorConfig


@dataclass(frozen=True)
class PairFMOutputComplexity:
    query_pair_count: int
    token_shards: int
    token_lanes: int
    pair_feature_width: int
    complex_source_count: int
    pairfm_useful_rows: int
    pairfm_padded_rows: int
    pairfm_pt_ct: int
    pairfm_rotations: int
    blocked_useful_rows: int
    blocked_padded_rows: int
    blocked_pt_ct: int
    blocked_rotations: int
    owner_encoded_rows: int
    owner_pt_ct: int
    owner_rotations: int
    conversion_encoded_rows: int
    conversion_pt_ct: int
    conversion_rotations: int
    conversion_conjugations: int
    conversion_imults: int
    conversion_depth: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


@dataclass
class PairFMConversionResult:
    real_ciphers: tuple[object, ...]
    carrier_ciphers: tuple[object, ...]
    crypto_context: object
    feature_layout: FeatureMajorPrefillLayout
    attention_layout: AttentionPairLayout
    complexity: PairFMOutputComplexity
    wall_seconds: float
    stage_seconds: dict[str, float]
    values: np.ndarray | None = None
    max_abs_diff: float | None = None

    def release(self) -> None:
        for cipher in self.real_ciphers + self.carrier_ciphers:
            release_if_supported(cipher)


def _geometry(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> tuple[int, int, int, int]:
    """Validate the selected production geometry.

    PairFM deliberately relies on the equality between query-pair ownership
    and feature-major token-shard ownership.  Keep that requirement next to
    the transform rather than importing it from the unrelated W_O operator.
    """

    pairs = int(attention_layout.query_pair_count)
    shards = int(feature_layout.cipher_count)
    lanes = int(feature_layout.tokens_per_cipher)
    pair_feature = 2 * int(attention_layout.head_dim)
    dimension = int(feature_layout.hidden_dim)
    if int(feature_layout.slots) != int(attention_layout.slots):
        raise ValueError("feature-major and attention slot counts must match.")
    if dimension != int(attention_layout.query_heads) * int(attention_layout.head_dim):
        raise ValueError("feature-major dimension must equal query_heads*head_dim.")
    if int(feature_layout.seq_len) != int(attention_layout.seq_len):
        raise ValueError("feature-major and attention sequence lengths must match.")
    if pairs != shards:
        raise ValueError("PairFM requires query pairs == token shards.")
    if int(attention_layout.seq_len) != shards * lanes:
        raise ValueError("PairFM requires a full token lane in every shard.")
    if dimension != pairs * pair_feature:
        raise ValueError("query-pair feature blocks must cover the hidden width.")
    if pair_feature % lanes:
        raise ValueError("pair feature width must be divisible by token lanes.")
    if int(attention_layout.payload_repetitions) != 1:
        raise ValueError("PairFM requires one full O_C payload per ciphertext.")
    return pairs, lanes, pair_feature, pair_feature // lanes


def _round_rows(rows: int, modulus: int) -> int:
    rows = int(rows)
    modulus = int(modulus)
    if rows <= 0 or modulus <= 0:
        raise ValueError("row counts and baby steps must be positive.")
    return int(ceil(rows / modulus)) * modulus


def _rotation_count(*, rows: int, baby_steps: int, ciphers: int) -> int:
    giant_count = int(ceil(int(rows) / int(baby_steps)))
    # One centered base rotation, B-1 baby rotations, and G-1 giant
    # rotations per ciphertext.  Both selected permutations have a nonzero
    # base offset at every supported nontrivial geometry.
    per_cipher = 1 + max(0, int(baby_steps) - 1) + max(0, giant_count - 1)
    return int(ciphers) * per_cipher


def _blocked_diagonal_geometry(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> tuple[int, int, int]:
    """Return ``(half_rows, stride, rows)`` for PairFM -> blocked.

    For ``i`` inside one query-pair feature block and token shard ``s``,

    ``source-destination = U*((C-1)*i - (F-1)*s)``.

    Dividing by the gcd gives a centered, possibly sparse integer row range.
    Empty rows are deliberately retained: this makes the existing BSGS
    helper applicable without changing its scale or ownership semantics.
    """

    pairs, lanes, pair_feature, _ = _geometry(feature_layout, attention_layout)
    divisor = gcd(pairs - 1, pair_feature - 1)
    stride = lanes * divisor
    half_rows = (pairs - 1) * (pair_feature - 1) // divisor
    return int(half_rows), int(stride), 2 * int(half_rows) + 1


def _pairfm_masks(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    """Masks for ``[d,q,lane] -> [d,lane,q]``."""

    pairs, _, pair_feature, _ = _geometry(feature_layout, attention_layout)
    seq_len = int(attention_layout.seq_len)
    head_dim = int(attention_layout.head_dim)
    masks = np.zeros((2 * seq_len - 1, int(feature_layout.slots)), dtype=dtype)
    for feature in range(head_dim):
        for lane in range(2):
            queries = np.arange(seq_len, dtype=np.int64)
            offsets = queries - (seq_len - 1) * lane
            destinations = (2 * feature + lane) * seq_len + queries
            masks[offsets + seq_len - 1, destinations] = 1.0
    # Every ciphertext contains exactly one query-pair payload; ``pairs`` is
    # used above only to force the common geometry validation.
    del pairs, pair_feature
    return masks, 1


def _blocked_masks(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    """Masks for ``[i, shard, u] -> [shard, i, u]``."""

    pairs, lanes, pair_feature, _ = _geometry(feature_layout, attention_layout)
    half_rows, stride, rows = _blocked_diagonal_geometry(
        feature_layout, attention_layout
    )
    divisor = gcd(pairs - 1, pair_feature - 1)
    masks = np.zeros((rows, int(feature_layout.slots)), dtype=dtype)
    local = np.arange(lanes, dtype=np.int64)
    for feature in range(pair_feature):
        for shard in range(pairs):
            normalized = (
                (pairs - 1) * feature - (pair_feature - 1) * shard
            ) // divisor
            destinations = (shard * pair_feature + feature) * lanes + local
            masks[normalized + half_rows, destinations] = 1.0
    return masks, stride


def pairfm_output_complexity(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    first_baby_steps: int = 16,
    block_baby_steps: int = 32,
) -> PairFMOutputComplexity:
    pairs, lanes, pair_feature, _ = _geometry(feature_layout, attention_layout)
    if pairs % 2:
        raise ValueError("PairFM complex source pairing requires an even pair count.")
    complex_sources = pairs // 2
    first_useful = 2 * int(attention_layout.seq_len) - 1
    first_padded = _round_rows(first_useful, int(first_baby_steps))
    first_rotations = _rotation_count(
        rows=first_useful,
        baby_steps=int(first_baby_steps),
        ciphers=complex_sources,
    )
    _, _, blocked_useful = _blocked_diagonal_geometry(
        feature_layout, attention_layout
    )
    blocked_padded = _round_rows(blocked_useful, int(block_baby_steps))
    owner_rows = pairs
    owner_pt_ct = pairs * pairs
    owner_rotations = pairs * max(0, pairs - 1)
    return PairFMOutputComplexity(
        query_pair_count=pairs,
        token_shards=pairs,
        token_lanes=lanes,
        pair_feature_width=pair_feature,
        complex_source_count=complex_sources,
        pairfm_useful_rows=first_useful,
        pairfm_padded_rows=first_padded,
        pairfm_pt_ct=first_padded * complex_sources,
        pairfm_rotations=first_rotations,
        blocked_useful_rows=blocked_useful,
        blocked_padded_rows=blocked_padded,
        blocked_pt_ct=blocked_padded * complex_sources,
        blocked_rotations=_rotation_count(
            rows=blocked_useful,
            baby_steps=int(block_baby_steps),
            ciphers=complex_sources,
        ),
        owner_encoded_rows=owner_rows,
        owner_pt_ct=owner_pt_ct,
        owner_rotations=owner_rotations,
        conversion_encoded_rows=first_padded + blocked_padded + owner_rows,
        conversion_pt_ct=(first_padded + blocked_padded) * complex_sources
        + owner_pt_ct,
        conversion_rotations=first_rotations
        + _rotation_count(
            rows=blocked_useful,
            baby_steps=int(block_baby_steps),
            ciphers=complex_sources,
        )
        + owner_rotations,
        conversion_conjugations=complex_sources,
        conversion_imults=3 * complex_sources,
        conversion_depth=3,
    )


def pairfm_output_rotations(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    first_baby_steps: int = 16,
    block_baby_steps: int = 32,
    log_n: int = 16,
) -> tuple[int, ...]:
    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    half_rows, blocked_stride, _ = _blocked_diagonal_geometry(
        feature_layout, attention_layout
    )
    rotations: set[int] = set()
    rotations.update(
        transform_rotation_set(
            seq_len=int(attention_layout.seq_len),
            stride=1,
            baby_steps=int(first_baby_steps),
        )
    )
    rotations.update(
        transform_rotation_set(
            seq_len=half_rows + 1,
            stride=blocked_stride,
            baby_steps=int(block_baby_steps),
        )
    )
    block = int(feature_layout.slots) // pairs
    rotations.update(
        (target - source) * block
        for source in range(pairs)
        for target in range(pairs)
        if target != source
    )
    rotations.add((1 << (int(log_n) + 1)) - 1)
    return tuple(sorted(int(value) for value in rotations if int(value)))


def output_c_to_pairfm_numpy(
    output_c: np.ndarray,
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> np.ndarray:
    """Return real PairFM ciphertext payloads, ``[pair, i, q]``."""

    pairs, _, pair_feature, _ = _geometry(feature_layout, attention_layout)
    output_c = np.asarray(output_c)
    expected = (pairs, int(attention_layout.slots))
    if output_c.shape != expected:
        raise ValueError(f"O_C must have shape {expected}, got {output_c.shape}.")
    return (
        output_c.reshape(
            pairs,
            int(attention_layout.head_dim),
            int(attention_layout.seq_len),
            2,
        )
        .transpose(0, 1, 3, 2)
        .reshape(pairs, pair_feature, int(attention_layout.seq_len))
        .reshape(pairs, int(attention_layout.slots))
    )


def output_c_to_feature_major_pairfm_numpy(
    output_c: np.ndarray,
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> np.ndarray:
    """Reference PairFM factorization ending in the public real FM layout."""

    pairs, lanes, pair_feature, _ = _geometry(feature_layout, attention_layout)
    pairfm = output_c_to_pairfm_numpy(
        output_c,
        feature_layout=feature_layout,
        attention_layout=attention_layout,
    )
    pairfm = pairfm.reshape(pairs, pair_feature, pairs, lanes)
    blocked = pairfm.transpose(0, 2, 1, 3)
    return blocked.transpose(1, 0, 2, 3).reshape(
        pairs, int(feature_layout.slots)
    )


def _owner_transpose_fhe(
    logical_sources: tuple[object, ...],
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    dtype: np.dtype,
    name: str,
    profile: dict[str, float],
) -> tuple[object, ...]:
    """Transpose external ``query_pair <-> token_shard`` ownership."""

    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    if len(logical_sources) != pairs:
        raise ValueError(f"owner transpose expects {pairs} real sources.")
    levels = {
        int(crypto_context.level_for_cipher(cipher)) for cipher in logical_sources
    }
    if len(levels) != 1:
        raise ValueError("owner transpose inputs must enter at one common level.")
    level = levels.pop()
    block = int(feature_layout.slots) // pairs
    accumulators: list[object | None] = [None] * pairs
    outputs: list[object] = []
    try:
        for source, cipher in enumerate(logical_sources):
            start = time.perf_counter()
            mask = np.zeros((int(feature_layout.slots),), dtype=dtype)
            # The complex split produced twice the wanted real value.
            mask[source * block : (source + 1) * block] = 0.5
            plain = None
            try:
                plain = _encode_mask(
                    mask,
                    name=f"{name}.source{source}",
                    level=level,
                    crypto_context=crypto_context,
                )
                sync_device(crypto_context.device)
                profile["pack_encode"] += time.perf_counter() - start
                start = time.perf_counter()
                for target in range(pairs):
                    offset = (target - source) * block
                    rotated = (
                        cipher
                        if offset == 0
                        else crypto_context.fhe.homo_rotate(
                            cipher, offset, crypto_context.context
                        )
                    )
                    try:
                        product = crypto_context.fhe.homo_mul_pt(
                            rotated, plain, crypto_context.context
                        )
                        accumulators[target] = _add_owned(
                            accumulators[target],
                            product,
                            crypto_context=crypto_context,
                        )
                    finally:
                        if rotated is not cipher:
                            release_if_supported(rotated)
                sync_device(crypto_context.device)
                profile["online"] += time.perf_counter() - start
            finally:
                release_if_supported(plain)

        start = time.perf_counter()
        for index, accumulator in enumerate(accumulators):
            if accumulator is None:
                raise RuntimeError(f"empty owner-transpose accumulator {index}.")
            output = crypto_context.fhe.rescale(
                accumulator, crypto_context.context
            )
            release_if_supported(accumulator)
            accumulators[index] = None
            outputs.append(output)
        sync_device(crypto_context.device)
        profile["rescale"] += time.perf_counter() - start
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for accumulator in accumulators:
            release_if_supported(accumulator)
        for output in outputs:
            release_if_supported(output)


def prepare_w_o_inputs_fhe(
    output_ciphers: tuple[object, ...] | list[object],
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    model_config: Llama3CKKSConfig | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    bootstrap_operator=None,
    verify: bool = False,
    expected_output_c: np.ndarray | None = None,
    function_prefix: str = "model.layers.0.self_attn.output_c_to_feature_major_pairfm",
) -> PairFMConversionResult:
    """Convert O_C inputs to feature-major W_O carriers (depth three)."""

    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    first_baby_steps = int(operator_config.pairfm_first_baby_steps)
    block_baby_steps = int(operator_config.pairfm_block_baby_steps)
    refresh_paired_inputs = bool(operator_config.refresh_output_before_pairfm)
    refresh_paired_inputs_if_below = operator_config.pairfm_refresh_min_input_limbs
    post_refresh_input_limbs = operator_config.pairfm_post_refresh_input_limbs
    checkpoint_precision_bits = int(
        operator_config.checkpoint_bootstrap_precision_bits
    )

    output_ciphers = tuple(output_ciphers)
    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    if len(output_ciphers) != pairs:
        raise ValueError(f"PairFM conversion expects {pairs} O_C ciphertexts.")
    input_levels = {
        int(crypto_context.level_for_cipher(cipher)) for cipher in output_ciphers
    }
    if len(input_levels) != 1:
        raise ValueError("all O_C ciphertexts must enter at one common level.")
    if int(crypto_context.max_slots) != int(feature_layout.slots):
        raise ValueError("crypto context and feature layout slot counts must match.")
    if post_refresh_input_limbs is not None and int(
        post_refresh_input_limbs
    ) <= 1:
        raise ValueError("post_refresh_input_limbs must exceed one.")
    model_config = Llama3CKKSConfig() if model_config is None else model_config
    config_slots = 1 << (int(model_config.simulator.logN) - 1)
    if config_slots != int(feature_layout.slots):
        raise ValueError("config logN and feature layout slot counts must match.")
    dtype = np.dtype(model_config.layout_transform.dtype)
    complexity = pairfm_output_complexity(
        feature_layout,
        attention_layout,
        first_baby_steps=int(first_baby_steps),
        block_baby_steps=int(block_baby_steps),
    )
    pairfm_masks, pairfm_stride = _pairfm_masks(
        feature_layout, attention_layout, dtype=dtype
    )
    blocked_masks, blocked_stride = _blocked_masks(
        feature_layout, attention_layout, dtype=dtype
    )
    wall_start = time.perf_counter()
    paired: tuple[object, ...] = ()
    pairfm: tuple[object, ...] = ()
    blocked: tuple[object, ...] = ()
    logical_sources: tuple[object, ...] = ()
    real_outputs: tuple[object, ...] = ()
    carrier_outputs: tuple[object, ...] = ()
    success = False
    stages = {
        "input_pairing": 0.0,
        "input_pair_bootstrap": 0.0,
        "input_pair_post_refresh_modulus_drop": 0.0,
        "pairfm_pack_encode": 0.0,
        "pairfm_online_rescale": 0.0,
        "block_pack_encode": 0.0,
        "block_online_rescale": 0.0,
        "source_split": 0.0,
        "scatter_pack_encode": 0.0,
        "scatter_online": 0.0,
        "scatter_rescale": 0.0,
        "output_pairing": 0.0,
    }
    try:
        start = time.perf_counter()
        paired = pair_real_ciphers(output_ciphers, crypto_context=crypto_context)
        sync_device(crypto_context.device)
        stages["input_pairing"] = time.perf_counter() - start

        should_refresh = bool(refresh_paired_inputs)
        if refresh_paired_inputs_if_below is not None:
            minimum = int(refresh_paired_inputs_if_below)
            if minimum <= 1:
                raise ValueError(
                    "refresh_paired_inputs_if_below must exceed one."
                )
            should_refresh = should_refresh or min(
                int(cipher.state.cur_limbs) for cipher in paired
            ) < minimum
        if os.environ.get("LLAMA_FHE_PAIRFM_LIMB_TRACE", "") == "1":
            print(
                f"[pairfm-limb-trace] {function_prefix}: paired limbs="
                f"{[int(c.state.cur_limbs) for c in paired]} "
                f"threshold={refresh_paired_inputs_if_below} "
                f"refresh={should_refresh}",
                flush=True,
            )
        if should_refresh:
            if bootstrap_operator is None:
                raise ValueError(
                    "refresh_paired_inputs requires a prepared bootstrap operator."
                )
            start = time.perf_counter()
            refreshed: list[object] = []
            try:
                for index, cipher in enumerate(paired):
                    refreshed.append(
                        bootstrap_operator.refresh(
                            cipher,
                            iterations=int(
                                operator_config
                                .pairfm_refresh_bootstrap_iterations
                            ),
                            precision_bits=int(checkpoint_precision_bits),
                        )
                    )
            except Exception:
                for cipher in refreshed:
                    release_if_supported(cipher)
                raise
            for cipher in paired:
                release_if_supported(cipher)
            paired = tuple(refreshed)
            sync_device(crypto_context.device)
            stages["input_pair_bootstrap"] = time.perf_counter() - start

            if post_refresh_input_limbs is not None:
                current_limbs = int(paired[0].state.cur_limbs)
                if current_limbs != int(post_refresh_input_limbs):
                    start = time.perf_counter()
                    dropped = drop_to_limbs(
                        paired,
                        target_limbs=int(post_refresh_input_limbs),
                        crypto_context=crypto_context,
                    )
                    sync_device(crypto_context.device)
                    for cipher in paired:
                        release_if_supported(cipher)
                    paired = dropped
                    stages["input_pair_post_refresh_modulus_drop"] = (
                        time.perf_counter() - start
                    )

        profile: dict[str, float] = {}
        pairfm = apply_transform_by_level(
            paired,
            masks=pairfm_masks,
            stride=pairfm_stride,
            baby_steps=int(first_baby_steps),
            crypto_context=crypto_context,
            name=f"{function_prefix}.pairfm",
            profile=profile,
        )
        for cipher in paired:
            release_if_supported(cipher)
        paired = ()
        stages["pairfm_pack_encode"] = float(
            profile.get("pack_encode", 0.0)
        )
        stages["pairfm_online_rescale"] = float(
            profile.get("online_rescale", 0.0)
        )

        profile = {}
        blocked = apply_transform_by_level(
            pairfm,
            masks=blocked_masks,
            stride=blocked_stride,
            baby_steps=int(block_baby_steps),
            crypto_context=crypto_context,
            name=f"{function_prefix}.blocked",
            profile=profile,
        )
        for cipher in pairfm:
            release_if_supported(cipher)
        pairfm = ()
        stages["block_pack_encode"] = float(profile.get("pack_encode", 0.0))
        stages["block_online_rescale"] = float(
            profile.get("online_rescale", 0.0)
        )

        start = time.perf_counter()
        logical_sources = split_complex_ciphers_twice(
            blocked, crypto_context=crypto_context
        )
        sync_device(crypto_context.device)
        stages["source_split"] = time.perf_counter() - start
        for cipher in blocked:
            release_if_supported(cipher)
        blocked = ()

        profile = {"pack_encode": 0.0, "online": 0.0, "rescale": 0.0}
        real_outputs = _owner_transpose_fhe(
            logical_sources,
            feature_layout=feature_layout,
            attention_layout=attention_layout,
            crypto_context=crypto_context,
            dtype=dtype,
            name=f"{function_prefix}.owner",
            profile=profile,
        )
        for cipher in logical_sources:
            release_if_supported(cipher)
        logical_sources = ()
        stages["scatter_pack_encode"] = profile["pack_encode"]
        stages["scatter_online"] = profile["online"]
        stages["scatter_rescale"] = profile["rescale"]

        start = time.perf_counter()
        carrier_outputs = pair_real_ciphers(
            real_outputs, crypto_context=crypto_context
        )
        sync_device(crypto_context.device)
        stages["output_pairing"] = time.perf_counter() - start

        values = None
        max_abs_diff = None
        if verify:
            start = time.perf_counter()
            packed = np.stack(
                [np.asarray(crypto_context.decrypt(cipher)) for cipher in real_outputs]
            )
            values = feature_layout.unpack(packed)
            if expected_output_c is not None:
                wanted_packed = output_c_to_feature_major_pairfm_numpy(
                    expected_output_c,
                    feature_layout=feature_layout,
                    attention_layout=attention_layout,
                )
                wanted = feature_layout.unpack(wanted_packed)
                max_abs_diff = float(np.max(np.abs(values - wanted)))
            stages["verification"] = time.perf_counter() - start

        success = True
        return PairFMConversionResult(
            real_ciphers=real_outputs,
            carrier_ciphers=carrier_outputs,
            crypto_context=crypto_context,
            feature_layout=feature_layout,
            attention_layout=attention_layout,
            complexity=complexity,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={str(key): float(value) for key, value in stages.items()},
            values=values,
            max_abs_diff=max_abs_diff,
        )
    finally:
        for cipher in paired + pairfm + blocked + logical_sources:
            release_if_supported(cipher)
        if not success:
            for cipher in real_outputs + carrier_outputs:
                release_if_supported(cipher)
