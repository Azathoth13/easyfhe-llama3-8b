from __future__ import annotations

import os
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import synchronize_device as sync_device
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.linear import LinearCarrierLayout

from ...approx.attention import apply_rope_heads, rope_cos_sin
from ..linear import split_complex_ciphers_twice
from ..primitives import (
    accumulate_owned as _add_owned,
)
from ..primitives import (
    encode_mask as _encode_mask,
)
from ..transform import apply_transform_by_level, encode_transform_rows
from .codec import decrypt_complex_rows, repeat_payload_cipher
from .config import AttentionOperatorConfig
from .sparse_bsgs_scheduler import (
    SparseMaskRoute,
    SparseRouteBatch,
    SparseRouteGroup,
    build_sparse_route_schedule,
    materialize_sparse_batch_masks,
    sparse_route_arrays,
)
from .sparse_bsgs_scheduler import (
    canonical_rotation as _canonical_rotation,
)


@dataclass(frozen=True)
class SparseDiagonalPlan:
    """A sparse BSGS decomposition of arbitrary cyclic matrix diagonals."""

    modulus: int
    diagonal_offsets: tuple[int, ...]
    baby_offsets: tuple[int, ...]
    giant_offsets: tuple[int, ...]

    @property
    def plaintext_rows(self) -> int:
        return len(self.diagonal_offsets)

    @property
    def rotations_per_cipher(self) -> int:
        return max(0, len(self.baby_offsets) - (0 in self.baby_offsets)) + max(
            0, len(self.giant_offsets) - (0 in self.giant_offsets)
        )


@dataclass(frozen=True)
class ComplexTokenAttentionComplexity:
    input_cipher_count: int
    query_output_cipher_count: int
    key_output_cipher_count: int
    value_output_cipher_count: int
    token_lanes: int
    lane_phase_plaintext_rows: int
    lane_phase_pt_ct_multiplications: int
    lane_phase_rotations: int
    block_gather_plaintext_rows: int
    block_gather_pt_ct_multiplications: int
    block_gather_rotations: int
    rectangular_plaintext_rows: int
    rectangular_pt_ct_multiplications: int
    rectangular_rotations: int
    rectangular_bsgs_modulus: int
    feature_expansion_rotations: int
    feature_expansion_conjugations: int
    key_shear_plaintext_rows: int
    key_shear_pt_ct_multiplications: int
    key_shear_rotations: int
    value_delta_plaintext_rows: int
    value_delta_pt_ct_multiplications: int
    value_delta_rotations: int
    value_delta_bsgs_modulus: int
    value_split_conjugations: int
    payload_repeat_rotations: int
    encoded_static_plaintext_rows: int
    pt_ct_multiplications: int
    rotations: int
    conjugations: int
    query_multiplicative_depth: int
    key_multiplicative_depth: int
    value_multiplicative_depth: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


@dataclass
class ComplexTokenAttentionResult:
    """Selected QK-ready boundary produced from private linear carriers."""

    query_ciphers: tuple[object, ...]
    key_ciphers: tuple[object, ...]
    value_ciphers: tuple[object, ...]
    wall_seconds: float
    stage_seconds: dict[str, float]
    scheduler_stats: dict[str, float]
    max_abs_diff: dict[str, float] | None = None

    def release(self) -> None:
        for cipher in self.query_ciphers + self.key_ciphers + self.value_ciphers:
            release_if_supported(cipher)


def _layout_parameters(
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
) -> dict[str, int]:
    if int(carrier_layout.seq_len) != int(attention_layout.seq_len):
        raise ValueError("carrier and attention layouts must have the same sequence length.")
    if int(carrier_layout.slots) != int(attention_layout.slots):
        raise ValueError("carrier and attention layouts must have the same slot count.")
    token_lanes = int(carrier_layout.token_lanes)
    seq_len = int(attention_layout.seq_len)
    head_dim = int(attention_layout.head_dim)
    if seq_len % token_lanes:
        raise ValueError(
            f"the direct adapter requires seq_len divisible by token_lanes, got {seq_len}/{token_lanes}."
        )
    if head_dim % 2:
        raise ValueError("folded Q/K carriers require an even head dimension.")
    query_pairs = int(attention_layout.query_pair_count)
    kv_pairs = int(attention_layout.key_value_pair_count)
    if kv_pairs % 2:
        raise ValueError("V/V carriers require an even number of KV head pairs.")
    source_count = seq_len // token_lanes
    if source_count != int(carrier_layout.cipher_count):
        raise ValueError("the direct adapter requires fully occupied carrier shards.")
    query_fold_width = query_pairs * head_dim
    key_fold_width = kv_pairs * head_dim
    value_carrier_count = kv_pairs // 2
    carrier_width = query_fold_width + key_fold_width + kv_pairs * head_dim
    if carrier_width > int(carrier_layout.dimension):
        raise ValueError(
            f"heterogeneous carrier width {carrier_width} exceeds carrier dimension "
            f"{carrier_layout.dimension}."
        )
    repetitions = int(attention_layout.payload_repetitions)
    if repetitions & (repetitions - 1):
        raise ValueError("payload repetitions must be a power of two for ciphertext repetition.")
    return {
        "slots": int(carrier_layout.slots),
        "token_lanes": token_lanes,
        "source_count": source_count,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "half": head_dim // 2,
        "query_pairs": query_pairs,
        "kv_pairs": kv_pairs,
        "value_carriers": value_carrier_count,
        "query_fold_width": query_fold_width,
        "key_fold_width": key_fold_width,
        "carrier_width": carrier_width,
        "qk_block": token_lanes * head_dim,
        "value_block": 2 * token_lanes * head_dim,
        "repeat_steps": max(0, repetitions.bit_length() - 1),
    }


def _canonical_offsets(values, *, slots: int) -> tuple[int, ...]:
    return tuple(sorted({_canonical_rotation(int(value), int(slots)) for value in values}))


@lru_cache(maxsize=64)
def _choose_sparse_diagonal_plan_cached(
    diagonal_offsets: tuple[int, ...], slots: int, max_baby_offsets: int
) -> SparseDiagonalPlan:
    offsets = _canonical_offsets(diagonal_offsets, slots=int(slots))
    if not offsets:
        raise ValueError("a sparse diagonal transform requires at least one diagonal.")
    max_baby_offsets = int(max_baby_offsets)
    if max_baby_offsets <= 0:
        raise ValueError("max_baby_offsets must be positive.")

    # Searching up to 4096 finds the production optima (1104 for Q/K and 238
    # for V) without making small layout-contract tests expensive.
    search_stop = max(2, min(int(slots) // 2, 4096))
    best: tuple[int, int, int, int, tuple[int, ...], tuple[int, ...]] | None = None
    for modulus in range(1, search_stop + 1):
        babies = tuple(sorted({int(offset) % modulus for offset in offsets}))
        if len(babies) > max_baby_offsets:
            continue
        giants = tuple(
            sorted(
                {
                    _canonical_rotation(int(offset) - (int(offset) % modulus), int(slots))
                    for offset in offsets
                }
            )
        )
        rotations = len(babies) - int(0 in babies) + len(giants) - int(0 in giants)
        candidate = (rotations, len(babies), len(giants), modulus, babies, giants)
        if best is None or candidate[:4] < best[:4]:
            best = candidate
    if best is None:
        raise ValueError(
            f"no sparse decomposition fits max_baby_offsets={max_baby_offsets}."
        )
    _, _, _, modulus, babies, giants = best
    return SparseDiagonalPlan(
        modulus=int(modulus),
        diagonal_offsets=offsets,
        baby_offsets=tuple(int(value) for value in babies),
        giant_offsets=tuple(int(value) for value in giants),
    )


def choose_sparse_diagonal_plan(
    diagonal_offsets: tuple[int, ...] | list[int],
    *,
    slots: int,
    max_baby_offsets: int = 32,
) -> SparseDiagonalPlan:
    return _choose_sparse_diagonal_plan_cached(
        tuple(int(value) for value in diagonal_offsets),
        int(slots),
        int(max_baby_offsets),
    )


def _value_delta_routes(params: dict[str, int]) -> dict[int, np.ndarray]:
    slots = params["slots"]
    token_lanes = params["token_lanes"]
    source_count = params["source_count"]
    seq_len = params["seq_len"]
    head_dim = params["head_dim"]
    value_block = params["value_block"]
    destinations: dict[int, list[int]] = {}
    for source in range(source_count):
        for local_token in range(token_lanes):
            token = source * token_lanes + local_token
            for feature in range(head_dim):
                query = (token - feature) % seq_len
                for lane in (0, 1):
                    source_slot = (
                        source * value_block
                        + 2 * token_lanes * feature
                        + 2 * local_token
                        + lane
                    )
                    dst = 2 * (feature * seq_len + query) + lane
                    offset = _canonical_rotation(source_slot - dst, slots)
                    destinations.setdefault(offset, []).append(dst)
    return {
        int(offset): np.asarray(indices, dtype=np.int64)
        for offset, indices in destinations.items()
    }


def _qk_axis_masks(params: dict[str, int], *, stage: int) -> tuple[np.ndarray, int]:
    """Two 31-diagonal swaps for ``[source 16][feature 4][feature 16]``."""

    source_count = params["source_count"]
    token_lanes = params["token_lanes"]
    half = params["half"]
    if source_count != 16 or half != 64:
        raise NotImplementedError(
            "the selected two-stage Q/K axis swap currently requires source=16 and half=64."
        )
    high_radix = 4
    low_radix = 16
    suffix = 2 * token_lanes
    masks = np.zeros((31, params["slots"]), dtype=np.float64)
    if int(stage) == 1:
        stride = 3 * low_radix * suffix
        for source in range(source_count):
            for high in range(high_radix):
                relative = source - 5 * high
                row = relative + 15
                for low in range(low_radix):
                    for tail in range(suffix):
                        destination = (
                            ((high * source_count + source) * low_radix + low)
                            * suffix
                            + tail
                        )
                        masks[row, destination] = 1.0
    elif int(stage) == 2:
        stride = 15 * suffix
        for high in range(high_radix):
            for source in range(source_count):
                for low in range(low_radix):
                    relative = source - low
                    row = relative + 15
                    for tail in range(suffix):
                        destination = (
                            ((high * low_radix + low) * source_count + source)
                            * suffix
                            + tail
                        )
                        masks[row, destination] = 1.0
    else:
        raise ValueError(f"Q/K axis stage must be 1 or 2, got {stage}.")
    return masks, int(stride)


def _qk_fused_gather_axis1_masks(
    params: dict[str, int],
    *,
    source: int,
) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    """Masks for target gather fused with the first Q/K axis swap.

    The input of source ``s`` is the output of :func:`_apply_lane_phase`:

    ``[target][feature_high 4][feature_low 16][token 8][lane 2]``.

    One plaintext multiplication routes a fixed ``(target, high)`` rectangle
    directly into the intermediate

    ``[feature_high 4][source 16][feature_low 16][token 8][lane 2]``.

    There are exactly ``(Q-pairs + K-pairs) * 4`` useful rows per source,
    i.e. 80 rows for production Llama.  Padding is intentionally not part of
    this stage: unlike ordinary regular BSGS, these are heterogeneous target
    accumulators and every useful row already has an independent destination.
    """

    source = int(source)
    if params["source_count"] != 16 or params["half"] != 64:
        raise NotImplementedError(
            "the fused Q/K gather-axis stage requires source=16 and half=64."
        )
    if not 0 <= source < params["source_count"]:
        raise IndexError(f"source {source} is outside the source-shard axis.")
    targets = params["query_pairs"] + params["kv_pairs"]
    suffix = 2 * params["token_lanes"]
    target_width = params["head_dim"] * params["token_lanes"]
    # The destination mask depends on ``source`` and ``high``, but not on the
    # target.  Encode four unique rows and broadcast each over the 20 target
    # rotations; this is the key distinction between 64 reusable rows and
    # 1,280 PT-CT uses in production.
    masks = np.zeros((4, params["slots"]), dtype=np.float64)
    metadata: list[tuple[int, int, int]] = []
    for high in range(4):
        for low in range(16):
            destination = (
                ((high * params["source_count"] + source) * 16 + low)
                * suffix
            )
            masks[high, destination : destination + suffix] = 1.0
    for target in range(targets):
        for high in range(4):
            source_start = target * target_width + high * 16 * suffix
            destination_start = (
                (high * params["source_count"] + source) * 16 * suffix
            )
            offset = _canonical_rotation(
                source_start - destination_start, params["slots"]
            )
            metadata.append((target, high, offset))
    return masks, tuple(metadata)


def _qk_fused_gather_axis1_rotation_count(params: dict[str, int]) -> int:
    """Exact logical rotation count of the correctness-first scheduler."""

    total = 0
    for source in range(params["source_count"]):
        _, metadata = _qk_fused_gather_axis1_masks(params, source=source)
        offsets = {int(offset) for _, _, offset in metadata}
        total += len(offsets) - int(0 in offsets)
    return int(total)


def _key_shear_routes(params: dict[str, int], *, upper: bool) -> dict[int, np.ndarray]:
    """Folded K lower shear or conjugate-derived upper shear."""

    slots = params["slots"]
    seq_len = params["seq_len"]
    half = params["half"]
    destinations: dict[int, list[int]] = {}
    for feature in range(half):
        physical_feature = feature + (half if upper else 0)
        for query in range(seq_len):
            source_token = (query + physical_feature) % seq_len
            for lane in (0, 1):
                source_slot = 2 * (feature * seq_len + source_token) + lane
                dst = 2 * (physical_feature * seq_len + query) + lane
                offset = _canonical_rotation(source_slot - dst, slots)
                destinations.setdefault(offset, []).append(dst)
    return {
        int(offset): np.asarray(indices, dtype=np.int64)
        for offset, indices in destinations.items()
    }


def _block_gather_routes(params: dict[str, int], source: int):
    qk_block = params["qk_block"]
    value_block = params["value_block"]
    q_base = 0
    k_base = params["token_lanes"] * params["query_fold_width"]
    v_base = params["token_lanes"] * (
        params["query_fold_width"] + params["key_fold_width"]
    )
    routes: list[tuple[str, int, int, str]] = []
    for pair in range(params["query_pairs"]):
        routes.append(
            ("q", pair, q_base + pair * qk_block - source * qk_block, "qk")
        )
    for pair in range(params["kv_pairs"]):
        routes.append(
            ("k", pair, k_base + pair * qk_block - source * qk_block, "qk")
        )
    for carrier in range(params["value_carriers"]):
        routes.append(
            (
                "v",
                carrier,
                v_base + carrier * value_block - source * value_block,
                "v",
            )
        )
    return tuple(routes)


def complex_token_attention_complexity(
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    *,
    sparse_max_baby_offsets: int = 32,
    shear_baby_steps: int = 16,
) -> ComplexTokenAttentionComplexity:
    if int(shear_baby_steps) != 16:
        raise ValueError(
            "the selected hybrid adapter currently requires shear_baby_steps=16."
        )
    params = _layout_parameters(carrier_layout, attention_layout)
    # The selected production factorization is specialized to the Llama
    # source/feature axes.  Small arbitrary layouts retain the CPU oracle but
    # do not claim this production static audit.
    if params["source_count"] != 16 or params["half"] != 64:
        raise NotImplementedError(
            "the staged production adapter audit requires 16 source shards and head_dim=128."
        )
    lane_rows = params["source_count"] * (2 * params["token_lanes"] - 1)
    lane_rotations = params["source_count"] * (
        2 * params["token_lanes"] - 2
    )
    qk_transforms = params["query_pairs"] + params["kv_pairs"]
    fused_axis1_unique_rows = params["source_count"] * 4
    fused_axis1_ptct = params["source_count"] * qk_transforms * 4
    fused_axis1_rotations = _qk_fused_gather_axis1_rotation_count(params)
    axis_rows = 32  # 31 useful rows padded to one 16x2 BSGS group.
    axis_rotations_per_cipher = 1 + 15 + 1
    rectangular_ptct = fused_axis1_ptct + axis_rows * qk_transforms
    rectangular_rotations = (
        fused_axis1_rotations + axis_rotations_per_cipher * qk_transforms
    )

    lower_routes = _key_shear_routes(params, upper=False)
    upper_routes = _key_shear_routes(params, upper=True)
    lower_plan = choose_sparse_diagonal_plan(
        tuple(lower_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    upper_plan = choose_sparse_diagonal_plan(
        tuple(upper_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    shear_ptct = 256 * params["kv_pairs"]
    shear_rotations = (
        lower_plan.rotations_per_cipher + upper_plan.rotations_per_cipher
    ) * params["kv_pairs"]

    value_routes = _value_delta_routes(params)
    value_gather_rows = params["source_count"]
    value_gather_ptct = (
        params["source_count"] * params["value_carriers"]
    )
    value_gather_rotations = sum(
        len(
            {
                _canonical_rotation(route[2], params["slots"])
                for route in _block_gather_routes(params, source)
                if route[0] == "v"
            }
        )
        - int(
            0
            in {
                _canonical_rotation(route[2], params["slots"])
                for route in _block_gather_routes(params, source)
                if route[0] == "v"
            }
        )
        for source in range(params["source_count"])
    )
    value_plan = choose_sparse_diagonal_plan(
        tuple(value_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    value_final_ciphers = params["value_carriers"]
    value_ptct = value_plan.plaintext_rows * value_final_ciphers
    value_rotations = value_plan.rotations_per_cipher * value_final_ciphers
    feature_count = params["query_pairs"]
    repeat_rotations = params["repeat_steps"] * (
        params["query_pairs"] + 2 * params["kv_pairs"]
    )
    encoded_rows = (
        lane_rows
        + fused_axis1_unique_rows
        + axis_rows
        + value_gather_rows
        + value_plan.plaintext_rows
        + 256
    )
    ptct = (
        lane_rows
        + rectangular_ptct
        + value_gather_ptct
        + value_ptct
        + shear_ptct
    )
    rotations = (
        lane_rotations
        + rectangular_rotations
        + feature_count
        + shear_rotations
        + value_gather_rotations
        + value_rotations
        + repeat_rotations
    )
    conjugations = (
        params["query_pairs"] + params["kv_pairs"] + params["value_carriers"]
    )
    return ComplexTokenAttentionComplexity(
        input_cipher_count=params["source_count"],
        query_output_cipher_count=params["query_pairs"],
        key_output_cipher_count=params["kv_pairs"],
        value_output_cipher_count=params["kv_pairs"],
        token_lanes=params["token_lanes"],
        lane_phase_plaintext_rows=lane_rows,
        lane_phase_pt_ct_multiplications=lane_rows,
        lane_phase_rotations=lane_rotations,
        block_gather_plaintext_rows=value_gather_rows,
        block_gather_pt_ct_multiplications=value_gather_ptct,
        block_gather_rotations=value_gather_rotations,
        rectangular_plaintext_rows=fused_axis1_unique_rows + axis_rows,
        rectangular_pt_ct_multiplications=rectangular_ptct,
        rectangular_rotations=rectangular_rotations,
        rectangular_bsgs_modulus=16,
        feature_expansion_rotations=feature_count,
        feature_expansion_conjugations=params["query_pairs"],
        key_shear_plaintext_rows=256,
        key_shear_pt_ct_multiplications=shear_ptct,
        key_shear_rotations=shear_rotations,
        value_delta_plaintext_rows=value_plan.plaintext_rows,
        value_delta_pt_ct_multiplications=value_ptct,
        value_delta_rotations=value_rotations,
        value_delta_bsgs_modulus=value_plan.modulus,
        value_split_conjugations=params["value_carriers"],
        payload_repeat_rotations=repeat_rotations,
        encoded_static_plaintext_rows=encoded_rows,
        pt_ct_multiplications=ptct,
        rotations=rotations,
        conjugations=conjugations,
        query_multiplicative_depth=3,
        key_multiplicative_depth=4,
        value_multiplicative_depth=3,
    )


def complex_token_attention_rotations(
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    *,
    sparse_max_baby_offsets: int = 32,
    shear_baby_steps: int = 16,
    log_n: int = 16,
) -> tuple[int, ...]:
    if int(shear_baby_steps) != 16:
        raise ValueError(
            "the selected hybrid adapter currently requires shear_baby_steps=16."
        )
    params = _layout_parameters(carrier_layout, attention_layout)
    if params["source_count"] != 16 or params["half"] != 64:
        raise NotImplementedError(
            "production staged adapter rotations require 16 source shards and head_dim=128."
        )
    value_routes = _value_delta_routes(params)
    value_plan = choose_sparse_diagonal_plan(
        tuple(value_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    lower_plan = choose_sparse_diagonal_plan(
        tuple(_key_shear_routes(params, upper=False)),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    upper_plan = choose_sparse_diagonal_plan(
        tuple(_key_shear_routes(params, upper=True)),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    rotations = set(
        range(-(params["token_lanes"] - 1), params["token_lanes"])
    )
    for source in range(params["source_count"]):
        _, metadata = _qk_fused_gather_axis1_masks(params, source=source)
        rotations.update(int(offset) for _, _, offset in metadata)
        rotations.update(
            _canonical_rotation(route[2], params["slots"])
            for route in _block_gather_routes(params, source)
            if route[0] == "v"
        )
    rotations.update(value_plan.baby_offsets)
    rotations.update(value_plan.giant_offsets)
    rotations.update(lower_plan.baby_offsets)
    rotations.update(lower_plan.giant_offsets)
    rotations.update(upper_plan.baby_offsets)
    rotations.update(upper_plan.giant_offsets)
    stride = _qk_axis_masks(params, stage=2)[1]
    rotations.add(-15 * int(stride))
    rotations.update(index * int(stride) for index in range(1, 16))
    rotations.add(16 * int(stride))
    rotations.add(-int(attention_layout.payload_slots) // 2)
    rotations.add((1 << (int(log_n) + 1)) - 1)
    step = int(attention_layout.payload_slots)
    while step < int(attention_layout.slots):
        rotations.add(-step)
        step *= 2
    return tuple(sorted(int(value) for value in rotations if int(value)))


def _lane_phase_masks(
    *,
    source: int,
    params: dict[str, int],
    cos: np.ndarray,
    sin: np.ndarray,
) -> dict[int, np.ndarray]:
    masks = {
        offset: np.zeros((params["slots"],), dtype=np.complex128)
        for offset in range(-(params["token_lanes"] - 1), params["token_lanes"])
    }
    q_end = params["query_fold_width"]
    k_end = q_end + params["key_fold_width"]
    for channel in range(params["carrier_width"]):
        pair_feature = channel // 2
        head_lane = channel % 2
        for local_token in range(params["token_lanes"]):
            token = source * params["token_lanes"] + local_token
            source_slot = channel * params["token_lanes"] + local_token
            dst = 2 * params["token_lanes"] * pair_feature + 2 * local_token + head_lane
            offset = source_slot - dst
            coefficient: complex = 1.0
            if channel < q_end:
                feature = pair_feature % params["half"]
                coefficient = complex(cos[token, feature], sin[token, feature])
            elif channel < k_end:
                local_pair_feature = (channel - q_end) // 2
                feature = local_pair_feature % params["half"]
                coefficient = complex(cos[token, feature], -sin[token, feature])
            masks[offset][dst] = coefficient
    return masks


def _apply_lane_phase(
    ciphers: tuple[object, ...],
    *,
    params: dict[str, int],
    cos: np.ndarray,
    sin: np.ndarray,
    crypto_context,
    function_prefix: str,
) -> tuple[object, ...]:
    outputs: list[object] = []
    try:
        for source, cipher in enumerate(ciphers):
            level = int(crypto_context.level_for_cipher(cipher))
            accumulator = None
            masks = _lane_phase_masks(
                source=source, params=params, cos=cos, sin=sin
            )
            try:
                for offset, mask in sorted(masks.items()):
                    plain = rotated = product = None
                    try:
                        plain = _encode_mask(
                            mask,
                            name=f"{function_prefix}.source{source}.diag{offset}",
                            level=level,
                            crypto_context=crypto_context,
                            dtype=np.complex128,
                        )
                        rotated = (
                            cipher
                            if offset == 0
                            else crypto_context.fhe.homo_rotate(
                                cipher, offset, crypto_context.context
                            )
                        )
                        product = crypto_context.fhe.homo_mul_pt(
                            rotated, plain, crypto_context.context
                        )
                        accumulator = _add_owned(
                            accumulator, product, crypto_context=crypto_context
                        )
                        product = None
                    finally:
                        release_if_supported(plain)
                        if rotated is not cipher:
                            release_if_supported(rotated)
                        release_if_supported(product)
                if accumulator is None:
                    raise RuntimeError("lane/phase transform produced no terms.")
                output = crypto_context.fhe.rescale(
                    accumulator, crypto_context.context
                )
                release_if_supported(accumulator)
                accumulator = None
                outputs.append(output)
            finally:
                release_if_supported(accumulator)
        return tuple(outputs)
    except Exception:
        for output in outputs:
            release_if_supported(output)
        raise


def _apply_qk_fused_gather_axis1(
    ciphers: tuple[object, ...],
    *,
    params: dict[str, int],
    crypto_context,
    function_prefix: str,
) -> tuple[object, ...]:
    """Gather every Q/K target while performing axis-swap stage one.

    This is deliberately a correctness-first scheduler.  It batches the 80
    source-specific rotations and plaintext products, then accumulates the
    four feature-high contributions for each target.  All target accumulators
    are rescaled once, so gather and the first axis swap consume one level.
    """

    if len(ciphers) != params["source_count"]:
        raise ValueError(
            f"expected {params['source_count']} source ciphers, got {len(ciphers)}."
        )
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    if len(levels) != 1:
        raise ValueError("fused Q/K gather-axis inputs must have one common level.")
    level = levels.pop()
    target_count = params["query_pairs"] + params["kv_pairs"]
    accumulators: list[object | None] = [None] * target_count
    outputs: list[object] = []
    try:
        # Four rotations are needed per target.  EasyFHE's current broadcast
        # kernel accepts at most 32 rotations in one call, so process eight
        # targets (4*8 rotations) per hoisted batch at production shape.
        target_batch_size = 8
        for source, cipher in enumerate(ciphers):
            masks, metadata = _qk_fused_gather_axis1_masks(
                params, source=source
            )
            plaintexts = None
            try:
                plaintexts = encode_transform_rows(
                    masks,
                    name=f"{function_prefix}.source{source}",
                    level=level,
                    crypto_context=crypto_context,
                )
                for target_start in range(0, target_count, target_batch_size):
                    target_stop = min(
                        target_count, target_start + target_batch_size
                    )
                    batch_metadata = metadata[4 * target_start : 4 * target_stop]
                    offsets = tuple(int(item[2]) for item in batch_metadata)
                    rotations = None
                    try:
                        rotations = crypto_context.fhe.fast_rotate(
                            cipher,
                            offsets,
                            crypto_context.context,
                            output_ext=True,
                        )
                        rotation_items = crypto_context.fhe.unpack_cipher_batch(
                            rotations
                        )
                        for local_target, target in enumerate(
                            range(target_start, target_stop)
                        ):
                            rotated_target = product = None
                            try:
                                # Materialize this target's four rotations as
                                # one contiguous batch before multiplying.
                                # A strided cipher_like view across targets is
                                # not a valid backend batch on full-head runs.
                                rotated_target = crypto_context.fhe.pack_cipher_batch(
                                    rotation_items[4 * local_target + high]
                                    for high in range(4)
                                )
                                product = crypto_context.fhe.grouped_pairwise_mac(
                                    rotated_target,
                                    plaintexts,
                                    1,
                                    crypto_context.context,
                                )
                                contribution = (
                                    crypto_context.fhe.moddown_from_ext(
                                        product, crypto_context.context
                                    )
                                    if product.is_ext
                                    else product.deep_copy()
                                )
                                accumulators[target] = _add_owned(
                                    accumulators[target],
                                    contribution,
                                    crypto_context=crypto_context,
                                )
                            finally:
                                release_if_supported(product)
                                release_if_supported(rotated_target)
                    finally:
                        release_if_supported(rotations)
            finally:
                release_if_supported(plaintexts)

        for index, accumulator in enumerate(accumulators):
            if accumulator is None:
                raise RuntimeError(f"fused Q/K output {index} is empty.")
            try:
                output = crypto_context.fhe.rescale(
                    accumulator, crypto_context.context
                )
            finally:
                release_if_supported(accumulator)
                accumulators[index] = None
            outputs.append(output)
        return tuple(outputs)
    except Exception:
        for accumulator in accumulators:
            release_if_supported(accumulator)
        for output in outputs:
            release_if_supported(output)
        raise


def _apply_value_block_gather(
    ciphers: tuple[object, ...],
    *,
    params: dict[str, int],
    crypto_context,
    function_prefix: str,
) -> tuple[object, ...]:
    """Gather only the two complex V/V carriers from lane-phase outputs.

    Q/K use the fused gather-axis stage above.  Keeping this tiny V-only
    gather makes the subsequent direct compact-to-Delta transform one level,
    preserving the selected V depth while using 16 encoded masks and 32
    PT-CT products in production.
    """

    if len(ciphers) != params["source_count"]:
        raise ValueError(
            f"expected {params['source_count']} source ciphers, got {len(ciphers)}."
        )
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    if len(levels) != 1:
        raise ValueError("V gather inputs must have one common level.")
    level = levels.pop()
    accumulators: list[object | None] = [None] * params["value_carriers"]
    outputs: list[object] = []
    try:
        for source, cipher in enumerate(ciphers):
            mask = np.zeros((params["slots"],), dtype=np.float64)
            start = source * params["value_block"]
            mask[start : start + params["value_block"]] = 1.0
            plain = _encode_mask(
                mask,
                name=f"{function_prefix}.source{source}",
                level=level,
                crypto_context=crypto_context,
                dtype=np.float64,
            )
            cache: dict[int, object] = {}
            try:
                for kind, index, raw_offset, _ in _block_gather_routes(
                    params, source
                ):
                    if kind != "v":
                        continue
                    offset = _canonical_rotation(raw_offset, params["slots"])
                    rotated = cache.get(offset)
                    if rotated is None:
                        rotated = (
                            cipher
                            if offset == 0
                            else crypto_context.fhe.homo_rotate(
                                cipher, offset, crypto_context.context
                            )
                        )
                        cache[offset] = rotated
                    product = crypto_context.fhe.homo_mul_pt(
                        rotated, plain, crypto_context.context
                    )
                    accumulators[index] = _add_owned(
                        accumulators[index],
                        product,
                        crypto_context=crypto_context,
                    )
            finally:
                for offset, rotated in cache.items():
                    if offset and rotated is not cipher:
                        release_if_supported(rotated)
                release_if_supported(plain)

        for index, accumulator in enumerate(accumulators):
            if accumulator is None:
                raise RuntimeError(f"V gather output {index} is empty.")
            output = crypto_context.fhe.rescale(
                accumulator, crypto_context.context
            )
            release_if_supported(accumulator)
            accumulators[index] = None
            outputs.append(output)
        return tuple(outputs)
    except Exception:
        for accumulator in accumulators:
            release_if_supported(accumulator)
        for output in outputs:
            release_if_supported(output)
        raise


def _apply_sparse_routes(
    ciphers: tuple[object, ...],
    *,
    routes: dict[int, np.ndarray | SparseMaskRoute],
    plan: SparseDiagonalPlan,
    slots: int,
    crypto_context,
    function_prefix: str,
    padded_plaintext_rows: int | None = None,
    rescale_outputs: bool = True,
) -> tuple[object, ...]:
    if not ciphers:
        return ()
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    if len(levels) != 1:
        raise ValueError("one sparse diagonal transform requires one input level.")
    level = levels.pop()
    by_giant: dict[int, list[tuple[int, int]]] = {}
    for offset in plan.diagonal_offsets:
        baby = int(offset) % int(plan.modulus)
        giant = _canonical_rotation(int(offset) - baby, int(slots))
        by_giant.setdefault(giant, []).append((int(offset), baby))

    rotated_inputs: list[dict[int, object]] = []
    totals: list[object | None] = [None] * len(ciphers)
    outputs: list[object] = []
    try:
        for cipher in ciphers:
            cache: dict[int, object] = {}
            # Register the cache before creating rotations so the outer
            # cleanup also owns a partially constructed cache.
            rotated_inputs.append(cache)
            for baby in plan.baby_offsets:
                cache[baby] = (
                    cipher
                    if baby == 0
                    else crypto_context.fhe.homo_rotate(
                        cipher, baby, crypto_context.context
                    )
                )

        for giant, entries in sorted(by_giant.items()):
            plains: list[tuple[int, object]] = []
            try:
                for offset, baby in entries:
                    destinations, coefficients = sparse_route_arrays(
                        routes[offset]
                    )
                    mask = np.zeros(
                        (int(slots),),
                        dtype=np.result_type(np.float64, coefficients.dtype),
                    )
                    mask[destinations] = coefficients
                    if giant:
                        mask = np.roll(mask, int(giant))
                    plains.append(
                        (
                            baby,
                            _encode_mask(
                                mask,
                                name=f"{function_prefix}.diag{offset}.g{giant}",
                                level=level,
                                crypto_context=crypto_context,
                                dtype=np.complex128 if np.iscomplexobj(mask) else np.float64,
                            ),
                        )
                    )
                for cipher_index, cache in enumerate(rotated_inputs):
                    group = None
                    for baby, plain in plains:
                        product = crypto_context.fhe.homo_mul_pt(
                            cache[baby], plain, crypto_context.context
                        )
                        group = _add_owned(
                            group, product, crypto_context=crypto_context
                        )
                    if group is None:
                        raise RuntimeError("sparse BSGS giant group was empty.")
                    if giant:
                        try:
                            contribution = crypto_context.fhe.homo_rotate(
                                group, giant, crypto_context.context
                            )
                        finally:
                            release_if_supported(group)
                    else:
                        contribution = group
                    totals[cipher_index] = _add_owned(
                        totals[cipher_index],
                        contribution,
                        crypto_context=crypto_context,
                    )
            finally:
                for _, plain in plains:
                    release_if_supported(plain)

        padded_rows = (
            plan.plaintext_rows
            if padded_plaintext_rows is None
            else int(padded_plaintext_rows)
        )
        if padded_rows < plan.plaintext_rows:
            raise ValueError(
                f"padded_plaintext_rows={padded_rows} is below the "
                f"{plan.plaintext_rows} nonzero routes."
            )
        for pad_index in range(padded_rows - plan.plaintext_rows):
            zero_plain = _encode_mask(
                np.zeros((int(slots),), dtype=np.float64),
                name=f"{function_prefix}.pad{pad_index}",
                level=level,
                crypto_context=crypto_context,
                dtype=np.float64,
            )
            try:
                for cipher_index, cipher in enumerate(ciphers):
                    zero_product = crypto_context.fhe.homo_mul_pt(
                        cipher, zero_plain, crypto_context.context
                    )
                    totals[cipher_index] = _add_owned(
                        totals[cipher_index],
                        zero_product,
                        crypto_context=crypto_context,
                    )
            finally:
                release_if_supported(zero_plain)

        for index, total in enumerate(totals):
            if total is None:
                raise RuntimeError(f"sparse diagonal output {index} is empty.")
            if bool(rescale_outputs):
                output = crypto_context.fhe.rescale(
                    total, crypto_context.context
                )
                release_if_supported(total)
            else:
                # Transfer ownership of the pre-rescale PT-CT accumulator to
                # the caller.  Native PairFM V uses this to combine all 16
                # source shards before paying one rescale per target carrier.
                output = total
            totals[index] = None
            outputs.append(output)
        return tuple(outputs)
    except Exception:
        for total in totals:
            release_if_supported(total)
        for output in outputs:
            release_if_supported(output)
        raise
    finally:
        for cipher, cache in zip(ciphers, rotated_inputs):
            for baby, rotated in cache.items():
                if baby and rotated is not cipher:
                    release_if_supported(rotated)


def _cipher_batch_slice(batch, start: int, stop: int):
    """Return a borrowed contiguous batch view without a host round trip."""

    start = int(start)
    stop = int(stop)
    if not 0 <= start < stop <= int(batch.batch_size):
        raise IndexError(
            f"invalid cipher batch slice [{start}:{stop}] for size {batch.batch_size}."
        )
    return batch.cipher_like(
        [component[start:stop] for component in batch.cv],
        batch_size=stop - start,
    )


def _sparse_batch_masks(
    batch: SparseRouteBatch,
    *,
    routes: dict[int, np.ndarray | SparseMaskRoute],
    slots: int,
) -> tuple[np.ndarray, tuple[tuple[SparseRouteGroup, int, int], ...]]:
    """Materialize at most 32 sparse rows and their encoded-batch slices."""

    return materialize_sparse_batch_masks(
        batch, routes=routes, slots=int(slots)
    )


def _fast_baby_rotation_batch(
    cipher,
    *,
    baby_offsets: tuple[int, ...],
    crypto_context,
):
    if not baby_offsets:
        raise ValueError("baby_offsets must not be empty")
    if len(baby_offsets) > 32:
        raise ValueError("the attention sparse scheduler caps baby batches at 32")
    if len(baby_offsets) == 1 and int(baby_offsets[0]) == 0:
        return crypto_context.fhe.pack_cipher_batch((cipher.deep_copy(),))
    return crypto_context.fhe.fast_rotate(
        cipher,
        baby_offsets,
        crypto_context.context,
        output_ext=True,
    )


def _select_baby_rotation_batch(
    rotation_batch,
    rotation_items: tuple[object, ...],
    *,
    baby_offsets: tuple[int, ...],
    baby_index: dict[int, int],
    crypto_context,
):
    """Select one giant group's babies, borrowing a view when contiguous."""

    indices = tuple(baby_index[int(offset)] for offset in baby_offsets)
    start = min(indices)
    if indices == tuple(range(start, start + len(indices))):
        return _cipher_batch_slice(
            rotation_batch, start, start + len(indices)
        )
    return crypto_context.fhe.pack_cipher_batch(
        rotation_items[index] for index in indices
    )


def _apply_sparse_routes_batched(
    ciphers: tuple[object, ...],
    *,
    routes: dict[int, np.ndarray | SparseMaskRoute],
    plan: SparseDiagonalPlan,
    slots: int,
    crypto_context,
    function_prefix: str,
    padded_plaintext_rows: int | None = None,
    batch_cap: int = 32,
    profile: dict[str, float] | None = None,
    rescale_outputs: bool = True,
) -> tuple[object, ...]:
    """GPU-batched sparse BSGS with one reusable baby basis per input.

    This is a physical scheduling optimization only.  Every nonzero route and
    requested padding row still contributes exactly one logical PT-CT
    multiplication; every nonzero baby and giant offset is unchanged.  The
    differences from :func:`_apply_sparse_routes` are:

    * up to 32 plaintext rows are transferred and encoded together;
    * all baby rotations of one input are produced by one hoisted fast batch;
    * one grouped GPU MAC replaces the Python multiply/add loop of a giant
      group, while preserving that group's route count.
    """

    if not ciphers:
        return ()
    required = (
        "fast_rotate",
        "grouped_pairwise_mac",
        "moddown_from_ext",
        "pack_cipher_batch",
        "unpack_cipher_batch",
    )
    missing = [name for name in required if not hasattr(crypto_context.fhe, name)]
    if missing:
        raise NotImplementedError(
            "batched sparse BSGS requires EasyFHE APIs: " + ", ".join(missing)
        )
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    if len(levels) != 1:
        raise ValueError("one sparse diagonal transform requires one input level.")
    level = levels.pop()
    schedule = build_sparse_route_schedule(
        diagonal_offsets=plan.diagonal_offsets,
        baby_offsets=plan.baby_offsets,
        modulus=int(plan.modulus),
        slots=int(slots),
        batch_cap=int(batch_cap),
        padded_plaintext_rows=padded_plaintext_rows,
    )
    if schedule.padding_rows and 0 not in schedule.baby_offsets:
        raise ValueError("zero padding requires a zero-offset baby rotation")

    baby_batches: list[object] = []
    baby_items: list[tuple[object, ...]] = []
    totals: list[object | None] = [None] * len(ciphers)
    outputs: list[object] = []
    started = time.perf_counter()
    try:
        for cipher in ciphers:
            rotations = _fast_baby_rotation_batch(
                cipher,
                baby_offsets=schedule.baby_offsets,
                crypto_context=crypto_context,
            )
            baby_batches.append(rotations)
            baby_items.append(
                tuple(crypto_context.fhe.unpack_cipher_batch(rotations))
            )

        baby_index = {
            int(offset): index
            for index, offset in enumerate(schedule.baby_offsets)
        }
        for batch_index, batch in enumerate(schedule.batches):
            masks, metadata = _sparse_batch_masks(
                batch, routes=routes, slots=int(slots)
            )
            plaintexts = None
            try:
                plaintexts = encode_transform_rows(
                    masks,
                    name=f"{function_prefix}.batch{batch_index}",
                    level=level,
                    crypto_context=crypto_context,
                )
                for cipher_index, rotations in enumerate(baby_items):
                    for group, start, stop in metadata:
                        selected = plain_group = product = contribution = None
                        try:
                            selected = _select_baby_rotation_batch(
                                baby_batches[cipher_index],
                                rotations,
                                baby_offsets=group.baby_offsets,
                                baby_index=baby_index,
                                crypto_context=crypto_context,
                            )
                            plain_group = _cipher_batch_slice(
                                plaintexts, start, stop
                            )
                            product = crypto_context.fhe.grouped_pairwise_mac(
                                selected,
                                plain_group,
                                1,
                                crypto_context.context,
                            )
                            contribution = (
                                crypto_context.fhe.moddown_from_ext(
                                    product, crypto_context.context
                                )
                                if product.is_ext
                                else product.deep_copy()
                            )
                            if int(group.giant_offset):
                                rotated = crypto_context.fhe.homo_rotate(
                                    contribution,
                                    int(group.giant_offset),
                                    crypto_context.context,
                                )
                                release_if_supported(contribution)
                                contribution = rotated
                            totals[cipher_index] = _add_owned(
                                totals[cipher_index],
                                contribution,
                                crypto_context=crypto_context,
                            )
                            contribution = None
                        finally:
                            release_if_supported(contribution)
                            release_if_supported(product)
                            release_if_supported(plain_group)
                            release_if_supported(selected)

                    if int(batch.padding_rows):
                        start = batch.route_rows
                        stop = batch.encoded_rows
                        selected = plain_padding = product = contribution = None
                        try:
                            selected = _select_baby_rotation_batch(
                                baby_batches[cipher_index],
                                rotations,
                                baby_offsets=(0,) * int(batch.padding_rows),
                                baby_index=baby_index,
                                crypto_context=crypto_context,
                            )
                            plain_padding = _cipher_batch_slice(
                                plaintexts, start, stop
                            )
                            product = crypto_context.fhe.grouped_pairwise_mac(
                                selected,
                                plain_padding,
                                1,
                                crypto_context.context,
                            )
                            contribution = (
                                crypto_context.fhe.moddown_from_ext(
                                    product, crypto_context.context
                                )
                                if product.is_ext
                                else product.deep_copy()
                            )
                            totals[cipher_index] = _add_owned(
                                totals[cipher_index],
                                contribution,
                                crypto_context=crypto_context,
                            )
                            contribution = None
                        finally:
                            release_if_supported(contribution)
                            release_if_supported(product)
                            release_if_supported(plain_padding)
                            release_if_supported(selected)
            finally:
                release_if_supported(plaintexts)

        for index, total in enumerate(totals):
            if total is None:
                raise RuntimeError(f"sparse diagonal output {index} is empty.")
            if bool(rescale_outputs):
                output = crypto_context.fhe.rescale(
                    total, crypto_context.context
                )
                release_if_supported(total)
            else:
                output = total
            totals[index] = None
            outputs.append(output)
        sync_device(crypto_context.device)
        if profile is not None:
            profile["batched"] = 1.0
            profile["wall_seconds"] = float(time.perf_counter() - started)
            profile["encode_batches"] = float(len(schedule.batches))
            profile["encoded_rows"] = float(schedule.encoded_rows)
            profile["route_rows"] = float(schedule.route_rows)
            profile["padding_rows"] = float(schedule.padding_rows)
            profile["baby_batches"] = float(len(ciphers))
            profile["baby_rotations"] = float(
                schedule.baby_rotations * len(ciphers)
            )
            profile["giant_groups"] = float(len(schedule.groups))
            profile["giant_rotations"] = float(
                schedule.giant_rotations * len(ciphers)
            )
            profile["logical_pt_ct"] = float(
                schedule.encoded_rows * len(ciphers)
            )
        return tuple(outputs)
    except Exception:
        for total in totals:
            release_if_supported(total)
        for output in outputs:
            release_if_supported(output)
        raise
    finally:
        for batch in baby_batches:
            release_if_supported(batch)


def _apply_sparse_routes_selected(
    ciphers: tuple[object, ...],
    *,
    routes: dict[int, np.ndarray | SparseMaskRoute],
    plan: SparseDiagonalPlan,
    slots: int,
    crypto_context,
    function_prefix: str,
    padded_plaintext_rows: int | None,
    use_batched: bool,
    batch_cap: int,
    fallback_on_error: bool,
    profile: dict[str, float] | None = None,
    rescale_outputs: bool = True,
) -> tuple[object, ...]:
    if not bool(use_batched):
        if profile is not None:
            profile["batched"] = 0.0
        return _apply_sparse_routes(
            ciphers,
            routes=routes,
            plan=plan,
            slots=int(slots),
            crypto_context=crypto_context,
            function_prefix=function_prefix,
            padded_plaintext_rows=padded_plaintext_rows,
            rescale_outputs=bool(rescale_outputs),
        )
    try:
        return _apply_sparse_routes_batched(
            ciphers,
            routes=routes,
            plan=plan,
            slots=int(slots),
            crypto_context=crypto_context,
            function_prefix=function_prefix,
            padded_plaintext_rows=padded_plaintext_rows,
            batch_cap=int(batch_cap),
            profile=profile,
            rescale_outputs=bool(rescale_outputs),
        )
    except Exception as exc:
        if not bool(fallback_on_error):
            raise
        warnings.warn(
            f"{function_prefix}: batched sparse BSGS failed; using the "
            f"correctness-first scheduler ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        if profile is not None:
            profile.clear()
            profile["batched"] = 0.0
            profile["fallbacks"] = 1.0
        return _apply_sparse_routes(
            ciphers,
            routes=routes,
            plan=plan,
            slots=int(slots),
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.fallback",
            padded_plaintext_rows=padded_plaintext_rows,
            rescale_outputs=bool(rescale_outputs),
        )


def _merge_scheduler_profile(
    profile: dict[str, float],
    *,
    label: str,
    stage_seconds: dict[str, float],
    scheduler_stats: dict[str, float],
) -> None:
    """Fold one sparse-scheduler profile into the stage and stats reports.

    ``wall_seconds`` is a duration and belongs with the stage timings; every
    other key is a scheduler counter, and the two aggregate counters
    (``fallbacks``, ``batched_stage_count``) accumulate across stages.
    """

    for name, value in profile.items():
        if name == "wall_seconds":
            stage_seconds[f"{label}.scheduler"] = float(value)
            continue
        scheduler_stats[f"{label}.{name}"] = float(value)
        if name == "fallbacks":
            scheduler_stats["fallbacks"] += float(value)
        elif name == "batched":
            scheduler_stats["batched_stage_count"] += float(value)


def _expand_c_feature_half(
    cipher,
    *,
    upper_sign: int,
    attention_layout: AttentionPairLayout,
    crypto_context,
):
    conjugated = transformed = shifted = None
    try:
        conjugated = crypto_context.fhe.homo_rotate(
            cipher, int(crypto_context.context.M) - 1, crypto_context.context
        )
        transformed = fhe.homo_mul_i(
            conjugated,
            crypto_context.context,
            negative=int(upper_sign) < 0,
        )
        shifted = crypto_context.fhe.homo_rotate(
            transformed,
            -int(attention_layout.payload_slots) // 2,
            crypto_context.context,
        )
        return crypto_context.fhe.homo_add(cipher, shifted, crypto_context.context)
    finally:
        release_if_supported(conjugated)
        release_if_supported(transformed)
        release_if_supported(shifted)


def _repeat_outputs(ciphers: tuple[object, ...], *, layout, crypto_context):
    outputs: list[object] = []
    try:
        for cipher in ciphers:
            outputs.append(
                repeat_payload_cipher(
                    cipher, layout=layout, crypto_context=crypto_context
                )
            )
        return tuple(outputs)
    except Exception:
        for output in outputs:
            release_if_supported(output)
        raise


def prepare_attention_inputs_fhe(
    carrier_ciphers: tuple[object, ...] | list[object],
    *,
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    cos: np.ndarray | None = None,
    sin: np.ndarray | None = None,
    model_config: Llama3CKKSConfig | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    verify: bool = False,
    expected_query: np.ndarray | None = None,
    expected_key: np.ndarray | None = None,
    expected_value: np.ndarray | None = None,
    function_prefix: str = "model.layers.0.self_attn.complex_token_attention",
) -> ComplexTokenAttentionResult:
    """Convert heterogeneous linear carriers directly to QK-ready layouts.

    This avoids materializing another persistent layout. Q/K use a
    lane/phase stage, a target gather fused with the first source-feature axis
    swap, and a second 31-diagonal axis swap.  K then fuses upper-half
    reconstruction into its Delta shear.  V uses the lane/phase
    stage, a tiny carrier gather, and one direct compact-to-Delta transform.
    Relative Q/K/V output depths are 3/4/3.

    Sparse-route scheduling is read from ``operator_config``: the release
    default batches mask encoding and reuses hoisted baby rotations
    (``batched_sparse_input_bsgs``, capped by ``sparse_input_batch_cap``).
    ``sparse_input_fallback`` keeps the correctness-first scheduler as the
    recovery path when the batched one raises; set it False to let backend
    errors surface instead.
    """

    model_config = Llama3CKKSConfig() if model_config is None else model_config
    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    # The stage below reads these as plain locals; unpack the schedule once.
    sparse_max_baby_offsets = int(operator_config.sparse_max_baby_offsets)
    qk_score_level_drop = int(operator_config.qk_score_level_drop)
    batched_sparse_bsgs = bool(operator_config.batched_sparse_input_bsgs)
    sparse_batch_cap = int(operator_config.sparse_input_batch_cap)
    sparse_scheduler_fallback = bool(operator_config.sparse_input_fallback)
    # RoPE phases are derived from the same config when the caller does not
    # supply them, so this has to happen before their shape is checked.
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must either both be provided or both omitted.")
    if cos is None:
        cos, sin = rope_cos_sin(
            seq_len=int(attention_layout.seq_len),
            head_dim=int(attention_layout.head_dim),
            theta=float(operator_config.rope_theta),
            start_pos=int(operator_config.rope_start_pos),
        )

    carrier_ciphers = tuple(carrier_ciphers)
    params = _layout_parameters(carrier_layout, attention_layout)
    if len(carrier_ciphers) != params["source_count"]:
        raise ValueError(
            f"expected {params['source_count']} linear carrier ciphers, got "
            f"{len(carrier_ciphers)}."
        )
    if not carrier_ciphers:
        raise ValueError("carrier_ciphers must not be empty.")
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in carrier_ciphers}
    if len(levels) != 1:
        raise ValueError("all linear carrier ciphers must enter at one level.")
    cos = np.asarray(cos, dtype=np.float64)
    sin = np.asarray(sin, dtype=np.float64)
    expected_phase_shape = (params["seq_len"], params["head_dim"])
    if cos.shape != expected_phase_shape or sin.shape != expected_phase_shape:
        raise ValueError(f"cos/sin must have shape {expected_phase_shape}.")
    value_routes = _value_delta_routes(params)
    value_plan = choose_sparse_diagonal_plan(
        tuple(value_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    lower_routes = _key_shear_routes(params, upper=False)
    upper_routes = _key_shear_routes(params, upper=True)
    lower_plan = choose_sparse_diagonal_plan(
        tuple(lower_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    upper_plan = choose_sparse_diagonal_plan(
        tuple(upper_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    axis2_masks, axis2_stride = _qk_axis_masks(params, stage=2)
    wall_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    scheduler_stats: dict[str, float] = {
        "enabled": float(batched_sparse_bsgs),
        "batch_cap": float(sparse_batch_cap),
        "fallback_allowed": float(sparse_scheduler_fallback),
        "fallbacks": 0.0,
        # Two key-shear calls plus one ordinary V call.
        "stage_count": 3.0,
        "batched_stage_count": 0.0,
    }
    lane_outputs: tuple[object, ...] = ()
    qk_axis1: tuple[object, ...] = ()
    q_compact: tuple[object, ...] = ()
    k_compact: tuple[object, ...] = ()
    v_compact: tuple[object, ...] = ()
    qk_low: tuple[object, ...] = ()
    q_full: tuple[object, ...] = ()
    q_outputs: tuple[object, ...] = ()
    k_lower: tuple[object, ...] = ()
    k_upper_sources: tuple[object, ...] = ()
    k_upper: tuple[object, ...] = ()
    k_combined: tuple[object, ...] = ()
    k_outputs: tuple[object, ...] = ()
    v_axis1_complex: tuple[object, ...] = ()
    v_complex: tuple[object, ...] = ()
    v_split: tuple[object, ...] = ()
    v_outputs: tuple[object, ...] = ()
    success = False
    try:
        start = time.perf_counter()
        lane_outputs = _apply_lane_phase(
            carrier_ciphers,
            params=params,
            cos=cos,
            sin=sin,
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.lane_phase",
        )
        sync_device(crypto_context.device)
        stage_seconds["lane_phase"] = time.perf_counter() - start

        start = time.perf_counter()
        qk_axis1 = _apply_qk_fused_gather_axis1(
            lane_outputs,
            params=params,
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.qk_gather_axis1",
        )
        sync_device(crypto_context.device)
        stage_seconds["qk_gather_axis1"] = time.perf_counter() - start

        # The attention entry budget is sized for the VALUE stream (which is
        # never re-bootstrapped and must reach PairFM pairing at six limbs).
        # The Q/K score stream is reset by the pre-exp softmax bootstrap and
        # only needs the tight schedule (K adapter 4 + QK 2 + preprocessing 1
        # + bootstrap floor 2), so any extra entry limbs can be shed here —
        # the earliest Q/K-only point — making the axis2 transform, expand,
        # key shear, and QK all proportionally cheaper.
        # LLAMA_FHE_QK_LEVEL_DROP overrides the scheduled value for sweeps.
        qk_level_drop = int(
            os.environ.get(
                "LLAMA_FHE_QK_LEVEL_DROP", str(int(qk_score_level_drop))
            )
        )
        if qk_level_drop > 0:
            start = time.perf_counter()
            dropped_qk: list[object] = []
            try:
                for cipher in qk_axis1:
                    target_limbs = (
                        int(cipher.state.cur_limbs) - qk_level_drop
                    )
                    if target_limbs <= 1:
                        raise ValueError(
                            "LLAMA_FHE_QK_LEVEL_DROP leaves too few limbs: "
                            f"cur_limbs={cipher.state.cur_limbs}, "
                            f"drop={qk_level_drop}."
                        )
                    target_state = cipher.state.replace(
                        cur_limbs=target_limbs,
                        scale_degree=1,
                        scaling_factor=None,
                    )
                    dropped_qk.append(
                        fhe.align_to(
                            cipher, target_state, crypto_context.context
                        )
                    )
            except Exception:
                for cipher in dropped_qk:
                    release_if_supported(cipher)
                raise
            for cipher in qk_axis1:
                release_if_supported(cipher)
            qk_axis1 = tuple(dropped_qk)
            sync_device(crypto_context.device)
            stage_seconds["qk_score_level_drop"] = (
                time.perf_counter() - start
            )

        start = time.perf_counter()
        v_compact = _apply_value_block_gather(
            lane_outputs,
            params=params,
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.value_gather",
        )
        for cipher in lane_outputs:
            release_if_supported(cipher)
        lane_outputs = ()
        sync_device(crypto_context.device)
        stage_seconds["value_gather"] = time.perf_counter() - start

        start = time.perf_counter()
        axis_profile: dict[str, float] = {}
        qk_low = apply_transform_by_level(
            qk_axis1,
            masks=axis2_masks,
            stride=axis2_stride,
            baby_steps=16,
            crypto_context=crypto_context,
            name=f"{function_prefix}.qk_axis2",
            profile=axis_profile,
        )
        for cipher in qk_axis1:
            release_if_supported(cipher)
        qk_axis1 = ()
        sync_device(crypto_context.device)
        stage_seconds["qk_axis2"] = time.perf_counter() - start
        for name, seconds in axis_profile.items():
            stage_seconds[f"qk_axis2.{name}"] = float(seconds)

        start = time.perf_counter()
        q_count = params["query_pairs"]
        q_expanded: list[object] = []
        try:
            for cipher in qk_low[:q_count]:
                q_expanded.append(
                    _expand_c_feature_half(
                        cipher,
                        upper_sign=1,
                        attention_layout=attention_layout,
                        crypto_context=crypto_context,
                    )
                )
            q_full = tuple(q_expanded)
            q_expanded = []
        finally:
            for cipher in q_expanded:
                release_if_supported(cipher)
        q_outputs = _repeat_outputs(
            q_full, layout=attention_layout, crypto_context=crypto_context
        )
        q_full = ()

        k_sources = qk_low[q_count:]
        lower_profile: dict[str, float] = {}
        k_lower = _apply_sparse_routes_selected(
            k_sources,
            routes=lower_routes,
            plan=lower_plan,
            slots=params["slots"],
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.key_shear_lower",
            padded_plaintext_rows=128,
            use_batched=batched_sparse_bsgs,
            batch_cap=sparse_batch_cap,
            fallback_on_error=sparse_scheduler_fallback,
            profile=lower_profile,
        )
        transformed_upper: list[object] = []
        try:
            for cipher in k_sources:
                conjugated = crypto_context.fhe.homo_rotate(
                    cipher,
                    int(crypto_context.context.M) - 1,
                    crypto_context.context,
                )
                try:
                    transformed_upper.append(
                        fhe.homo_mul_i(
                            conjugated,
                            crypto_context.context,
                            negative=True,
                        )
                    )
                finally:
                    release_if_supported(conjugated)
            k_upper_sources = tuple(transformed_upper)
            transformed_upper = []
        finally:
            for cipher in transformed_upper:
                release_if_supported(cipher)
        upper_profile: dict[str, float] = {}
        k_upper = _apply_sparse_routes_selected(
            k_upper_sources,
            routes=upper_routes,
            plan=upper_plan,
            slots=params["slots"],
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.key_shear_upper",
            padded_plaintext_rows=128,
            use_batched=batched_sparse_bsgs,
            batch_cap=sparse_batch_cap,
            fallback_on_error=sparse_scheduler_fallback,
            profile=upper_profile,
        )
        for cipher in k_upper_sources:
            release_if_supported(cipher)
        k_upper_sources = ()
        combined: list[object] = []
        try:
            for lower, upper in zip(k_lower, k_upper, strict=True):
                combined.append(
                    crypto_context.fhe.homo_add(
                        lower, upper, crypto_context.context
                    )
                )
            k_combined = tuple(combined)
            combined = []
        finally:
            for cipher in combined:
                release_if_supported(cipher)
            for cipher in k_lower + k_upper:
                release_if_supported(cipher)
            k_lower = k_upper = ()
        k_outputs = _repeat_outputs(
            k_combined, layout=attention_layout, crypto_context=crypto_context
        )
        k_combined = ()
        for cipher in qk_low:
            release_if_supported(cipher)
        qk_low = ()
        sync_device(crypto_context.device)
        stage_seconds["expand_and_key_shear"] = time.perf_counter() - start
        for label, sparse_profile in (
            ("key_shear_lower", lower_profile),
            ("key_shear_upper", upper_profile),
        ):
            _merge_scheduler_profile(
                sparse_profile,
                label=label,
                stage_seconds=stage_seconds,
                scheduler_stats=scheduler_stats,
            )

        start = time.perf_counter()
        value_profile: dict[str, float] = {}
        v_complex = _apply_sparse_routes_selected(
            v_compact,
            routes=value_routes,
            plan=value_plan,
            slots=params["slots"],
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.value_delta",
            padded_plaintext_rows=None,
            use_batched=batched_sparse_bsgs,
            batch_cap=sparse_batch_cap,
            fallback_on_error=sparse_scheduler_fallback,
            profile=value_profile,
        )
        for cipher in v_compact:
            release_if_supported(cipher)
        v_compact = ()
        value_split = split_complex_ciphers_twice(
            v_complex, crypto_context=crypto_context
        )
        value_real = value_split[0::2]
        value_imag = value_split[1::2]
        ordered: list[object] = []
        for real, imaginary in zip(value_real, value_imag, strict=True):
            ordered.extend((real, imaginary))
        v_split = tuple(ordered)
        v_outputs = _repeat_outputs(
            v_split, layout=attention_layout, crypto_context=crypto_context
        )
        v_split = ()
        for cipher in v_complex:
            release_if_supported(cipher)
        v_complex = ()
        sync_device(crypto_context.device)
        value_delta_label = "value_delta"
        stage_seconds[f"{value_delta_label}_and_split"] = time.perf_counter() - start
        _merge_scheduler_profile(
            value_profile,
            label=value_delta_label,
            stage_seconds=stage_seconds,
            scheduler_stats=scheduler_stats,
        )

        max_abs_diff = None
        if verify:
            if expected_query is None or expected_key is None or expected_value is None:
                raise ValueError(
                    "verify=True requires expected_query, expected_key, and expected_value."
                )
            query_rope = apply_rope_heads(
                expected_query,
                cos,
                sin,
                num_heads=int(attention_layout.query_heads),
                head_dim=int(attention_layout.head_dim),
            )
            key_rope = apply_rope_heads(
                expected_key,
                cos,
                sin,
                num_heads=int(attention_layout.key_value_heads),
                head_dim=int(attention_layout.head_dim),
            )
            actual_q = decrypt_complex_rows(q_outputs, crypto_context=crypto_context)
            actual_k = decrypt_complex_rows(k_outputs, crypto_context=crypto_context)
            actual_v = np.stack(
                [
                    np.asarray(crypto_context.decrypt(cipher))
                    for cipher in v_outputs
                ]
            )
            wanted_q = attention_layout.pack_query_c_feature_folded(query_rope)
            wanted_k = attention_layout.pack_key_delta_feature_folded(key_rope)
            wanted_v = attention_layout.pack_value_delta(expected_value)
            max_abs_diff = {
                "query_folded": float(np.max(np.abs(actual_q - wanted_q))),
                "key_folded": float(np.max(np.abs(actual_k - wanted_k))),
                "value": float(np.max(np.abs(actual_v - wanted_v))),
            }

        success = True
        return ComplexTokenAttentionResult(
            query_ciphers=q_outputs,
            key_ciphers=k_outputs,
            value_ciphers=v_outputs,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={name: float(seconds) for name, seconds in stage_seconds.items()},
            scheduler_stats={
                name: float(value) for name, value in scheduler_stats.items()
            },
            max_abs_diff=max_abs_diff,
        )
    finally:
        for group in (
            lane_outputs,
            qk_axis1,
            q_compact,
            k_compact,
            v_compact,
            qk_low,
            q_full,
            k_lower,
            k_upper_sources,
            k_upper,
            k_combined,
            v_axis1_complex,
            v_complex,
            v_split,
        ):
            for cipher in group:
                release_if_supported(cipher)
        if not success:
            for cipher in q_outputs + k_outputs + v_outputs:
                release_if_supported(cipher)
