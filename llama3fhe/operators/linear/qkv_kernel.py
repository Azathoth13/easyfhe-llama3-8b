from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from math import ceil

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import synchronize_device as sync_device
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.linear import LinearCarrierLayout

from . import (
    LLAMA3_LINEAR_SCHEDULES,
    LinearOperator,
    LinearOperatorConfig,
    linear_rotations,
    structured_linear_weight,
    structured_weight_packing_enabled,
)


@dataclass(frozen=True)
class _PreparedQKVWeights:
    hidden_states: np.ndarray
    q_weight_for_pack: np.ndarray
    k_weight_for_pack: np.ndarray
    v_weight_for_pack: np.ndarray
    seq_len: int
    input_width: int
    query_width: int
    key_value_width: int


@dataclass(frozen=True)
class _CarrierShape:
    query_fold_width: int
    key_fold_width: int
    carrier_width: int


def _pair_interleaved_mapping(
    pairs: tuple[tuple[int, int], ...], *, head_dim: int
) -> np.ndarray:
    values: list[int] = []
    for first, second in pairs:
        for feature in range(int(head_dim)):
            values.extend(
                (int(first) * int(head_dim) + feature,
                 int(second) * int(head_dim) + feature)
            )
    result = np.asarray(values, dtype=np.int64)
    if not np.array_equal(np.sort(result), np.arange(result.size)):
        raise ValueError("head pairs must cover every logical head exactly once.")
    return result


def _prepare_qkv_weights(
    hidden_states: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    attention_layout: AttentionPairLayout,
    input_column_scale: np.ndarray | None = None,
) -> _PreparedQKVWeights:
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be [seq,hidden], got {hidden_states.shape}.")
    seq_len, input_width = map(int, hidden_states.shape)
    q_weight = np.asarray(q_weight, dtype=np.float32)
    k_weight = np.asarray(k_weight, dtype=np.float32)
    v_weight = np.asarray(v_weight, dtype=np.float32)
    if q_weight.shape != (int(attention_layout.query_heads) * int(attention_layout.head_dim), input_width):
        raise ValueError(f"invalid Q weight shape {q_weight.shape}.")
    kv_shape = (
        int(attention_layout.key_value_heads) * int(attention_layout.head_dim),
        input_width,
    )
    if k_weight.shape != kv_shape or v_weight.shape != kv_shape:
        raise ValueError(
            f"K/V weights must both have shape {kv_shape}, got "
            f"{k_weight.shape}/{v_weight.shape}."
        )
    if int(attention_layout.seq_len) != seq_len:
        raise ValueError("QKV input sequence length does not match attention layout.")
    column_scale = None
    if input_column_scale is not None:
        column_scale = np.asarray(input_column_scale, dtype=np.float32).reshape(-1)
        if column_scale.shape != (input_width,):
            raise ValueError(
                "QKV input_column_scale must have shape "
                f"{(input_width,)}, got {column_scale.shape}."
            )
        if not np.all(np.isfinite(column_scale)):
            raise ValueError("QKV input_column_scale must be finite.")
    query_mapping = _pair_interleaved_mapping(
        attention_layout.query_head_pairs(), head_dim=int(attention_layout.head_dim)
    )
    kv_mapping = _pair_interleaved_mapping(
        attention_layout.key_value_head_pairs(), head_dim=int(attention_layout.head_dim)
    )
    q_weight_for_pack = q_weight[query_mapping]
    k_weight_for_pack = k_weight[kv_mapping]
    v_weight_for_pack = v_weight[kv_mapping]
    if column_scale is not None:
        q_weight_for_pack *= column_scale[None, :]
        k_weight_for_pack *= column_scale[None, :]
        v_weight_for_pack *= column_scale[None, :]
    return _PreparedQKVWeights(
        hidden_states=hidden_states,
        q_weight_for_pack=q_weight_for_pack,
        k_weight_for_pack=k_weight_for_pack,
        v_weight_for_pack=v_weight_for_pack,
        seq_len=seq_len,
        input_width=input_width,
        query_width=int(q_weight.shape[0]),
        key_value_width=int(k_weight.shape[0]),
    )


def _carrier_shape(
    prepared: _PreparedQKVWeights, *, head_dim: int
) -> _CarrierShape:
    if int(head_dim) <= 0 or int(head_dim) % 2:
        raise ValueError("head_dim must be a positive even value.")
    query_fold_width = int(prepared.query_width) // 2
    key_fold_width = int(prepared.key_value_width) // 2
    return _CarrierShape(
        query_fold_width=query_fold_width,
        key_fold_width=key_fold_width,
        carrier_width=(query_fold_width + key_fold_width + int(prepared.key_value_width) // 2),
    )


def _heterogeneous_qkv_weight(
    prepared: _PreparedQKVWeights,
    *,
    attention_layout: AttentionPairLayout,
    static: _CarrierShape,
) -> np.ndarray:
    head_dim = int(attention_layout.head_dim)
    folded = head_dim // 2
    pair_width = 2 * head_dim
    folded_pair_width = 2 * folded
    query_pairs = int(prepared.query_width) // pair_width
    kv_pairs = int(prepared.key_value_width) // pair_width
    key_base = int(static.query_fold_width)
    value_base = key_base + int(static.key_fold_width)
    combined = np.zeros(
        (int(static.carrier_width), int(prepared.input_width)),
        dtype=np.complex128,
    )
    for pair in range(query_pairs):
        dst = slice(pair * folded_pair_width, (pair + 1) * folded_pair_width)
        source = pair * pair_width
        combined[dst] = 0.5 * prepared.q_weight_for_pack[source : source + folded_pair_width]
        combined[dst] += 0.5j * prepared.q_weight_for_pack[
            source + folded_pair_width : source + 2 * folded_pair_width
        ]
    for pair in range(kv_pairs):
        dst = slice(
            key_base + pair * folded_pair_width,
            key_base + (pair + 1) * folded_pair_width,
        )
        source = pair * pair_width
        combined[dst] = prepared.k_weight_for_pack[source : source + folded_pair_width]
        combined[dst] -= 1j * prepared.k_weight_for_pack[
            source + folded_pair_width : source + 2 * folded_pair_width
        ]
    for carrier in range(kv_pairs // 2):
        dst = slice(
            value_base + carrier * pair_width,
            value_base + (carrier + 1) * pair_width,
        )
        combined[dst] = 0.5 * prepared.v_weight_for_pack[
            2 * carrier * pair_width : (2 * carrier + 1) * pair_width
        ]
        combined[dst] += 0.5j * prepared.v_weight_for_pack[
            (2 * carrier + 1) * pair_width : (2 * carrier + 2) * pair_width
        ]
    return combined


@dataclass(frozen=True)
class ComplexTokenDiagonalLayout:
    """Feature-major complex-token packing used by QKV projection.

    One ciphertext owns two groups of ``slots / dimension`` tokens.  For
    ciphertext ``c`` and lane ``u``::

        slot(d, u) = d * real_token_lanes + u
        Re(slot(d, u)) = H[2*L*c + u, d]
        Im(slot(d, u)) = H[2*L*c + L + u, d]

    The caller supplies persistent feature-major inputs, so no encrypted input
    layout conversion is part of this operator.
    """

    seq_len: int = 128
    dimension: int = 4096
    slots: int = 32768

    def __post_init__(self) -> None:
        if int(self.seq_len) <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}.")
        if int(self.dimension) <= 0 or int(self.dimension) & (int(self.dimension) - 1):
            raise ValueError(
                f"dimension must be a positive power of two, got {self.dimension}."
            )
        if int(self.slots) % int(self.dimension):
            raise ValueError(
                "complex token-diagonal slots must be divisible by dimension, got "
                f"{self.slots}/{self.dimension}."
            )

    @property
    def real_token_lanes(self) -> int:
        return int(self.slots) // int(self.dimension)

    @property
    def tokens_per_cipher(self) -> int:
        return 2 * self.real_token_lanes

    @property
    def cipher_count(self) -> int:
        return int(ceil(int(self.seq_len) / self.tokens_per_cipher))

    @property
    def recovered_layout(self) -> LinearCarrierLayout:
        return LinearCarrierLayout(
            seq_len=int(self.seq_len),
            dimension=int(self.dimension),
            slots=int(self.slots),
        )

    def slot(self, feature: int, lane: int) -> int:
        feature = int(feature)
        lane = int(lane)
        if not 0 <= feature < int(self.dimension):
            raise IndexError(
                f"feature={feature} is outside [0, {self.dimension})."
            )
        if not 0 <= lane < self.real_token_lanes:
            raise IndexError(
                f"lane={lane} is outside [0, {self.real_token_lanes})."
            )
        return feature * self.real_token_lanes + lane

    def pack(self, rows: np.ndarray, *, dtype=np.complex128) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float64)
        expected = (int(self.seq_len), int(self.dimension))
        if rows.shape != expected:
            raise ValueError(
                f"complex token-diagonal rows must have shape {expected}, got {rows.shape}."
            )
        if not np.issubdtype(np.dtype(dtype), np.complexfloating):
            raise ValueError(f"complex token-diagonal packing requires complex dtype, got {dtype}.")
        padded_tokens = self.cipher_count * self.tokens_per_cipher
        padded = np.zeros((padded_tokens, int(self.dimension)), dtype=np.float64)
        padded[: int(self.seq_len)] = rows
        groups = padded.reshape(
            self.cipher_count, self.tokens_per_cipher, int(self.dimension)
        )
        real = groups[:, : self.real_token_lanes].transpose(0, 2, 1)
        imaginary = groups[:, self.real_token_lanes :].transpose(0, 2, 1)
        return np.asarray(real + 1j * imaginary, dtype=dtype).reshape(
            self.cipher_count, int(self.slots)
        )

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        packed = np.asarray(packed)
        expected = (self.cipher_count, int(self.slots))
        if packed.shape != expected:
            raise ValueError(
                f"complex token-diagonal payload must have shape {expected}, got {packed.shape}."
            )
        feature_lane = packed.reshape(
            self.cipher_count, int(self.dimension), self.real_token_lanes
        )
        real = feature_lane.real.transpose(0, 2, 1)
        imaginary = feature_lane.imag.transpose(0, 2, 1)
        rows = np.concatenate((real, imaginary), axis=1).reshape(
            -1, int(self.dimension)
        )
        return np.asarray(rows[: int(self.seq_len)], dtype=np.float32)


@dataclass(frozen=True)
class ComplexTokenDiagonalQKVComplexity:
    input_cipher_count: int
    evaluated_cipher_count: int
    output_cipher_count: int
    dimension: int
    carrier_width: int
    reuse_baby_rotations: bool
    baby_steps: int
    giant_steps: int
    plaintext_chunks: int
    encoded_weight_plaintexts: int
    pt_ct_multiplications: int
    input_conjugations: int
    baby_rotations: int
    giant_rotations: int
    chunk_alignment_rotations: int
    rotations_including_conjugations: int
    projection_rescales: int
    recovery_additions: int
    recovery_subtractions: int
    recovery_imults: int
    multiplicative_depth: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


@dataclass
class ComplexTokenDiagonalQKVResult:
    """Recovered heterogeneous complex linear carriers in 8-token shards."""

    carrier_ciphers: tuple[object, ...]
    output_layout: LinearCarrierLayout
    carrier_width: int
    complexity: ComplexTokenDiagonalQKVComplexity
    wall_seconds: float
    stage_seconds: dict[str, float]
    output_levels: tuple[int, ...]
    max_abs_diff: float | None = None

    def release(self) -> None:
        for cipher in self.carrier_ciphers:
            release_if_supported(cipher)


def complex_token_diagonal_qkv_complexity(
    *,
    seq_len: int = 128,
    dimension: int = 4096,
    slots: int = 32768,
    carrier_width: int = 3072,
    baby_steps: int = LLAMA3_LINEAR_SCHEDULES.qkv.baby_steps,
    baby_anchor_step: int = LLAMA3_LINEAR_SCHEDULES.qkv.baby_anchor_step,
    max_plaintext_rows: int = LLAMA3_LINEAR_SCHEDULES.qkv.max_plaintext_rows,
    reuse_baby_rotations: bool = True,
) -> ComplexTokenDiagonalQKVComplexity:
    layout = ComplexTokenDiagonalLayout(
        seq_len=int(seq_len), dimension=int(dimension), slots=int(slots)
    )
    baby_steps = int(baby_steps)
    max_plaintext_rows = int(max_plaintext_rows)
    carrier_width = int(carrier_width)
    if baby_steps <= 0 or int(dimension) % baby_steps:
        raise ValueError(
            f"baby_steps={baby_steps} must divide dimension={dimension}."
        )
    if max_plaintext_rows < baby_steps:
        raise ValueError("max_plaintext_rows must hold one baby-step group.")
    if not 0 < carrier_width <= int(dimension):
        raise ValueError(
            f"carrier_width must be in [1, {dimension}], got {carrier_width}."
        )

    plan = LinearOperatorConfig(
        baby_steps=baby_steps,
        baby_anchor_step=int(baby_anchor_step),
        max_plaintext_rows=max_plaintext_rows,
        reuse_baby_rotations=bool(reuse_baby_rotations),
    ).plan(
        dimension=int(dimension),
        token_lanes=int(layout.real_token_lanes),
    )
    giant_steps = plan.giant_steps
    chunk_sizes = plan.chunk_sizes
    chunks = plan.plaintext_chunks
    evaluated = 2 * layout.cipher_count
    baby_rotation_rounds = 1 if bool(reuse_baby_rotations) else chunks
    baby_rotations = (
        evaluated * baby_rotation_rounds * max(0, baby_steps - 1)
    )
    giant_rotations = evaluated * sum(max(0, size - 1) for size in chunk_sizes)
    alignment_rotations = evaluated * max(0, chunks - 1)
    output_count = layout.recovered_layout.cipher_count
    second_group_outputs = max(0, output_count - layout.cipher_count)
    return ComplexTokenDiagonalQKVComplexity(
        input_cipher_count=layout.cipher_count,
        evaluated_cipher_count=evaluated,
        output_cipher_count=output_count,
        dimension=int(dimension),
        carrier_width=carrier_width,
        reuse_baby_rotations=bool(reuse_baby_rotations),
        baby_steps=baby_steps,
        giant_steps=giant_steps,
        plaintext_chunks=chunks,
        encoded_weight_plaintexts=int(dimension),
        pt_ct_multiplications=int(dimension) * evaluated,
        input_conjugations=layout.cipher_count,
        baby_rotations=baby_rotations,
        giant_rotations=giant_rotations,
        chunk_alignment_rotations=alignment_rotations,
        rotations_including_conjugations=(
            baby_rotations
            + giant_rotations
            + alignment_rotations
            + layout.cipher_count
        ),
        projection_rescales=evaluated,
        recovery_additions=layout.cipher_count,
        recovery_subtractions=second_group_outputs,
        recovery_imults=second_group_outputs,
        multiplicative_depth=1,
    )


def complex_token_diagonal_qkv_rotations(
    *,
    dimension: int = 4096,
    slots: int = 32768,
    real_token_lanes: int = 8,
    operator_config: LinearOperatorConfig | None = None,
) -> tuple[int, ...]:
    """Key union for the complex-token diagonal QKV kernel.

    The kernel and this planner must derive their union from the *same*
    config: a partial reconstruction here silently plans a different key set
    whenever a schedule value differs from the dataclass default, and the
    missing key only surfaces as a crash deep inside a rotation.
    """

    return linear_rotations(
        dimension=int(dimension),
        token_lanes=int(real_token_lanes),
        slots=int(slots),
        include_conjugation=True,
        operator_config=(
            LLAMA3_LINEAR_SCHEDULES.qkv
            if operator_config is None
            else operator_config
        ),
    )


def _complex_dtype(config: Llama3CKKSConfig) -> np.dtype:
    name = str(config.qkv_linear.dtype).lower()
    return np.dtype(np.complex64 if name in {"float32", "single"} else np.complex128)


def _projection_weights(
    hidden_states: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    attention_layout: AttentionPairLayout,
    input_column_scale: np.ndarray | None = None,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Return prepared metadata, desired carrier W, and 0.5*W padded square.

    If ``u = 0.5*W*z`` and ``v = 0.5*W*conj(z)``, then
    ``u+v = W*Re(z)`` and ``i*(v-u) = W*Im(z)``.  Absorbing the half here keeps
    recovery level-free.
    """

    prepared = _prepare_qkv_weights(
        hidden_states,
        q_weight,
        k_weight,
        v_weight,
        attention_layout=attention_layout,
        input_column_scale=input_column_scale,
    )
    static = _carrier_shape(prepared, head_dim=int(attention_layout.head_dim))
    carrier_weight = _heterogeneous_qkv_weight(
        prepared, attention_layout=attention_layout, static=static
    )
    working = np.zeros(
        (prepared.input_width, prepared.input_width), dtype=np.complex128
    )
    working[: int(static.carrier_width)] = 0.5 * carrier_weight
    return prepared, carrier_weight, working


def _structured_projection_weight(
    hidden_states: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    attention_layout: AttentionPairLayout,
    input_column_scale: np.ndarray | None = None,
):
    """Describe the Q/Q-K/K-V/V carrier matrix without materializing it."""

    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be [seq,hidden], got {hidden_states.shape}."
        )
    seq_len, input_width = map(int, hidden_states.shape)
    if int(attention_layout.seq_len) != seq_len:
        raise ValueError("QKV input sequence length does not match attention layout.")
    q_weight = np.asarray(q_weight, dtype=np.float32)
    k_weight = np.asarray(k_weight, dtype=np.float32)
    v_weight = np.asarray(v_weight, dtype=np.float32)
    query_width = int(attention_layout.query_heads) * int(
        attention_layout.head_dim
    )
    key_value_width = int(attention_layout.key_value_heads) * int(
        attention_layout.head_dim
    )
    if q_weight.shape != (query_width, input_width):
        raise ValueError(f"invalid Q weight shape {q_weight.shape}.")
    kv_shape = (key_value_width, input_width)
    if k_weight.shape != kv_shape or v_weight.shape != kv_shape:
        raise ValueError(
            f"K/V weights must both have shape {kv_shape}, got "
            f"{k_weight.shape}/{v_weight.shape}."
        )

    column_scale = np.ones((input_width,), dtype=np.float64)
    if input_column_scale is not None:
        column_scale = np.asarray(
            input_column_scale, dtype=np.float64
        ).reshape(-1)
        if column_scale.shape != (input_width,):
            raise ValueError(
                "QKV input_column_scale must have shape "
                f"{(input_width,)}, got {column_scale.shape}."
            )
        if not np.all(np.isfinite(column_scale)):
            raise ValueError("QKV input_column_scale must be finite.")

    head_dim = int(attention_layout.head_dim)
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even value.")
    folded_pair_width = head_dim
    pair_width = 2 * head_dim
    query_pairs = query_width // pair_width
    kv_pairs = key_value_width // pair_width
    query_fold_width = query_width // 2
    key_fold_width = key_value_width // 2
    value_fold_width = key_value_width // 2
    carrier_width = query_fold_width + key_fold_width + value_fold_width

    query_mapping = _pair_interleaved_mapping(
        attention_layout.query_head_pairs(), head_dim=head_dim
    )
    kv_mapping = _pair_interleaved_mapping(
        attention_layout.key_value_head_pairs(), head_dim=head_dim
    )
    real_source = np.full((input_width,), -1, dtype=np.int64)
    real_row = np.full((input_width,), -1, dtype=np.int64)
    real_scale = np.zeros((input_width,), dtype=np.float64)
    imag_source = np.full((input_width,), -1, dtype=np.int64)
    imag_row = np.full((input_width,), -1, dtype=np.int64)
    imag_scale = np.zeros((input_width,), dtype=np.float64)

    for pair in range(query_pairs):
        dst = np.arange(
            pair * folded_pair_width,
            (pair + 1) * folded_pair_width,
            dtype=np.int64,
        )
        source_start = pair * pair_width
        real_source[dst] = 0
        imag_source[dst] = 0
        real_row[dst] = query_mapping[source_start : source_start + folded_pair_width]
        imag_row[dst] = query_mapping[
            source_start + folded_pair_width : source_start + 2 * folded_pair_width
        ]
        # 0.5 carrier fold followed by the projection's 0.5 recovery scale.
        real_scale[dst] = 0.25
        imag_scale[dst] = 0.25

    key_base = query_fold_width
    for pair in range(kv_pairs):
        dst = np.arange(
            key_base + pair * folded_pair_width,
            key_base + (pair + 1) * folded_pair_width,
            dtype=np.int64,
        )
        source_start = pair * pair_width
        real_source[dst] = 1
        imag_source[dst] = 1
        real_row[dst] = kv_mapping[source_start : source_start + folded_pair_width]
        imag_row[dst] = kv_mapping[
            source_start + folded_pair_width : source_start + 2 * folded_pair_width
        ]
        real_scale[dst] = 0.5
        imag_scale[dst] = -0.5

    value_base = key_base + key_fold_width
    for carrier in range(kv_pairs // 2):
        dst = np.arange(
            value_base + carrier * pair_width,
            value_base + (carrier + 1) * pair_width,
            dtype=np.int64,
        )
        real_start = 2 * carrier * pair_width
        imag_start = (2 * carrier + 1) * pair_width
        real_source[dst] = 2
        imag_source[dst] = 2
        real_row[dst] = kv_mapping[real_start : real_start + pair_width]
        imag_row[dst] = kv_mapping[imag_start : imag_start + pair_width]
        real_scale[dst] = 0.25
        imag_scale[dst] = 0.25

    descriptor = structured_linear_weight(
        (q_weight, k_weight, v_weight),
        dimension=input_width,
        real_source=real_source,
        real_row=real_row,
        real_scale=real_scale,
        imag_source=imag_source,
        imag_row=imag_row,
        imag_scale=imag_scale,
        column_scale=column_scale,
    )
    return hidden_states, descriptor, carrier_width


def unpack_complex_carrier_shards(
    packed: np.ndarray,
    *,
    layout: LinearCarrierLayout,
    output_width: int,
) -> np.ndarray:
    """Unpack complex carrier shards without discarding the imaginary part."""

    packed = np.asarray(packed)
    expected = (layout.cipher_count, int(layout.slots))
    if packed.shape != expected:
        raise ValueError(
            f"complex carrier payload must have shape {expected}, got {packed.shape}."
        )
    output_width = int(output_width)
    if not 0 < output_width <= int(layout.dimension):
        raise ValueError(
            f"output_width must be in [1, {layout.dimension}], got {output_width}."
        )
    rows = packed.reshape(
        layout.cipher_count, int(layout.dimension), layout.token_lanes
    ).transpose(0, 2, 1).reshape(-1, int(layout.dimension))
    return np.asarray(
        rows[: int(layout.seq_len), :output_width], dtype=np.complex128
    )


def _decrypt_complex_ciphers(
    ciphers: tuple[object, ...], *, crypto_context
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for cipher in ciphers:
        imaginary_cipher = fhe.homo_mul_i(
            cipher,
            crypto_context.context,
            negative=True,
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


def qkv_fhe_complex_token_diagonal(
    hidden_states: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    attention_layout: AttentionPairLayout,
    config: Llama3CKKSConfig | None = None,
    operator_config: LinearOperatorConfig | None = None,
    extra_rotations: tuple[int, ...] | list[int] = (),
    crypto_context=None,
    input_ciphers: tuple[object, ...] | list[object] | None = None,
    verify: bool = True,
    function_prefix: str = "model.layers.0.self_attn.complex_token_diagonal_qkv",
    input_column_scale: np.ndarray | None = None,
) -> ComplexTokenDiagonalQKVResult:
    """Project 8 feature-major complex-token ciphertexts into QKV carriers.

    The same square complex plaintext matrix is encoded once and applied to the
    concatenated batch ``(z_0..z_7, conj(z_0)..conj(z_7))``.  Level-free
    add/subtract/``i`` recovery returns 16 ordinary 8-token carrier shards,
    while every carrier value remains complex for the heterogeneous
    Q/K-feature and V/V meanings.
    """

    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.qkv
        if operator_config is None
        else operator_config
    )
    # The stage code below reads the policy as plain locals; binding them here
    # keeps one object at the boundary.
    baby_steps = int(operator_config.baby_steps)
    baby_anchor_step = int(operator_config.baby_anchor_step)
    max_plaintext_rows = int(operator_config.max_plaintext_rows)
    reuse_baby_rotations = bool(operator_config.reuse_baby_rotations)
    hoist_strategy = str(operator_config.hoist_strategy)

    config = Llama3CKKSConfig() if config is None else config
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    wall_start = time.perf_counter()
    weight_transform_start = wall_start
    use_structured_weight = structured_weight_packing_enabled()
    carrier_weight = None
    if use_structured_weight:
        hidden_states, projection_weight, carrier_width = (
            _structured_projection_weight(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                attention_layout=attention_layout,
                input_column_scale=input_column_scale,
            )
        )
        seq_len, input_width = map(int, hidden_states.shape)
        if verify:
            carrier_weight = 2.0 * projection_weight.materialize(
                dtype=np.complex128
            )[:carrier_width]
    else:
        prepared, carrier_weight, projection_weight = _projection_weights(
            hidden_states,
            q_weight,
            k_weight,
            v_weight,
            attention_layout=attention_layout,
            input_column_scale=input_column_scale,
        )
        seq_len = int(prepared.seq_len)
        input_width = int(prepared.input_width)
        carrier_width = int(carrier_weight.shape[0])
    weight_transform_seconds = time.perf_counter() - weight_transform_start
    slots = 1 << (int(config.simulator.logN) - 1)
    input_layout = ComplexTokenDiagonalLayout(
        seq_len=seq_len, dimension=input_width, slots=slots
    )
    output_layout = input_layout.recovered_layout
    complexity = complex_token_diagonal_qkv_complexity(
        seq_len=seq_len,
        dimension=input_width,
        slots=slots,
        carrier_width=carrier_width,
        baby_steps=int(baby_steps),
        baby_anchor_step=int(baby_anchor_step),
        max_plaintext_rows=int(max_plaintext_rows),
        reuse_baby_rotations=bool(reuse_baby_rotations),
    )
    rotations = set(
        complex_token_diagonal_qkv_rotations(
            dimension=input_width,
            slots=slots,
            real_token_lanes=input_layout.real_token_lanes,
            # The same policy the kernel evaluates with, so the planned key
            # union cannot differ from the one the rotations actually need.
            operator_config=LinearOperatorConfig(
                baby_steps=int(baby_steps),
                baby_anchor_step=int(baby_anchor_step),
                max_plaintext_rows=int(max_plaintext_rows),
                reuse_baby_rotations=bool(reuse_baby_rotations),
                hoist_strategy=str(hoist_strategy),
            ),
        )
    )
    rotations.update(int(rotation) for rotation in extra_rotations if int(rotation))

    if crypto_context is None or input_ciphers is None:
        raise ValueError(
            "qkv_fhe_complex_token_diagonal requires an application-owned "
            "context and encrypted complex-token inputs."
        )

    # Match the public LD-QKV wall contract: exclude context/key generation,
    # but include input packing/encryption when this API owns the inputs.
    stage_seconds: dict[str, float] = {
        "context_setup": 0.0,
        "weight_transform_cpu": float(weight_transform_seconds),
    }
    owns_inputs = False
    input_ciphers = tuple(input_ciphers)
    if len(input_ciphers) != input_layout.cipher_count:
        raise ValueError(
            f"complex token-diagonal QKV expects {input_layout.cipher_count} inputs, "
            f"got {len(input_ciphers)}."
        )

    input_ciphers = tuple(input_ciphers)
    input_levels = {
        int(crypto_context.level_for_cipher(cipher)) for cipher in input_ciphers
    }
    if len(input_levels) != 1:
        if owns_inputs:
            for cipher in input_ciphers:
                release_if_supported(cipher)
        raise ValueError(
            "complex token-diagonal QKV inputs must enter at one common level."
        )
    conjugated_inputs: tuple[object, ...] = ()
    projected: tuple[object, ...] = ()
    outputs: list[object] = []
    projection_operator = LinearOperator(
        crypto_context=crypto_context,
        token_lanes=input_layout.real_token_lanes,
        config=LinearOperatorConfig(
            baby_steps=int(baby_steps),
            baby_anchor_step=int(baby_anchor_step),
            max_plaintext_rows=int(max_plaintext_rows),
            reuse_baby_rotations=bool(reuse_baby_rotations),
            hoist_strategy=str(hoist_strategy),
        ),
    )
    success = False
    try:
        start = time.perf_counter()
        conjugate_rotation = int(crypto_context.context.M) - 1
        conjugated_list: list[object] = []
        try:
            for cipher in input_ciphers:
                conjugated_list.append(
                    crypto_context.fhe.homo_rotate(
                        cipher, conjugate_rotation, crypto_context.context
                    )
                )
            conjugated_inputs = tuple(conjugated_list)
            conjugated_list = []
        finally:
            for cipher in conjugated_list:
                release_if_supported(cipher)
        sync_device(crypto_context.device)
        stage_seconds["input_conjugation"] = time.perf_counter() - start

        start = time.perf_counter()
        projected, profile = projection_operator.project(
            input_ciphers + conjugated_inputs,
            projection_weight,
            dtype=_complex_dtype(config),
        )
        sync_device(crypto_context.device)
        stage_seconds["paired_projection"] = time.perf_counter() - start
        for name, seconds in profile.items():
            stage_seconds[f"paired_projection_{name}"] = float(seconds)

        direct = projected[: input_layout.cipher_count]
        conjugate_evaluation = projected[input_layout.cipher_count :]
        start = time.perf_counter()
        for cipher_index, (left, right) in enumerate(
            zip(direct, conjugate_evaluation, strict=True)
        ):
            first_group = crypto_context.fhe.homo_add(
                left, right, crypto_context.context
            )
            outputs.append(first_group)
            second_group_token_base = (
                cipher_index * input_layout.tokens_per_cipher
                + input_layout.real_token_lanes
            )
            if second_group_token_base < int(input_layout.seq_len):
                delta = crypto_context.fhe.homo_sub(
                    right, left, crypto_context.context
                )
                try:
                    outputs.append(
                        fhe.homo_mul_i(delta, crypto_context.context)
                    )
                finally:
                    release_if_supported(delta)
        sync_device(crypto_context.device)
        stage_seconds["token_group_recovery"] = time.perf_counter() - start
        if len(outputs) != output_layout.cipher_count:
            raise RuntimeError(
                f"recovered {len(outputs)} carrier shards, expected {output_layout.cipher_count}."
            )

        carriers = None
        max_abs_diff = None
        if verify:
            start = time.perf_counter()
            decrypted = _decrypt_complex_ciphers(
                tuple(outputs), crypto_context=crypto_context
            )
            carriers = unpack_complex_carrier_shards(
                decrypted, layout=output_layout, output_width=carrier_width
            )
            if carrier_weight is None:
                raise RuntimeError("QKV verification is missing its CPU oracle.")
            expected = hidden_states.astype(np.float64) @ carrier_weight.T
            max_abs_diff = float(np.max(np.abs(carriers - expected)))
            sync_device(crypto_context.device)
            stage_seconds["verify_decrypt"] = time.perf_counter() - start

        success = True
        return ComplexTokenDiagonalQKVResult(
            carrier_ciphers=tuple(outputs),
            output_layout=output_layout,
            carrier_width=carrier_width,
            complexity=complexity,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={name: float(value) for name, value in stage_seconds.items()},
            output_levels=tuple(
                sorted(
                    {
                        int(crypto_context.level_for_cipher(cipher))
                        for cipher in outputs
                    }
                )
            ),
            max_abs_diff=max_abs_diff,
        )
    finally:
        projection_operator.close()
        for cipher in conjugated_inputs + projected:
            release_if_supported(cipher)
        if owns_inputs:
            for cipher in input_ciphers:
                release_if_supported(cipher)
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)
