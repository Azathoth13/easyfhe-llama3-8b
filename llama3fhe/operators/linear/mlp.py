from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from math import ceil

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import synchronize_device as sync_device
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.operators.nonlinear.elementwise import eval_silu_chebyshev_cipher

from ...approx.chebyshev import chebyshev_ps_mul_depth, eval_chebyshev_series_direct
from ..primitives import mul_cipher_rescale as _mul_cipher_rescale
from . import (
    LLAMA3_LINEAR_SCHEDULES,
    LinearOperator,
    LinearOperatorConfig,
    linear_rotations,
    pair_real_ciphers,
    split_complex_cipher_twice,
    structured_linear_weight,
    baby_reuse_across_pages_enabled,
    structured_weight_packing_enabled,
)


@dataclass(frozen=True)
class FeatureMajorMLPLayout:
    """Paged ``T=hidden_dim`` MLP layout around a canonical FM boundary.

    The public input and output use :class:`FeatureMajorPrefillLayout`.  The
    intermediate width is split into hidden-width feature pages while keeping
    the same private carrier geometry in every page. Inside the kernel, adjacent
    real token-shard ciphertexts are paired as ``x + i*y``.
    """

    canonical: FeatureMajorPrefillLayout
    intermediate_width: int = 14336

    def __post_init__(self) -> None:
        if int(self.intermediate_width) <= 0:
            raise ValueError(
                "intermediate_width must be positive, got "
                f"{self.intermediate_width}."
            )

    @property
    def page_width(self) -> int:
        return int(self.canonical.hidden_dim)

    @property
    def feature_page_count(self) -> int:
        return int(ceil(int(self.intermediate_width) / self.page_width))

    @property
    def paired_token_cipher_count(self) -> int:
        return int(ceil(int(self.canonical.cipher_count) / 2))

    @property
    def intermediate_real_cipher_count(self) -> int:
        return self.feature_page_count * int(self.canonical.cipher_count)

    @property
    def intermediate_complex_cipher_count(self) -> int:
        return self.feature_page_count * self.paired_token_cipher_count

    def active_width(self, feature_page: int) -> int:
        feature_page = int(feature_page)
        if not 0 <= feature_page < self.feature_page_count:
            raise IndexError(
                f"feature_page={feature_page} is outside "
                f"[0, {self.feature_page_count})."
            )
        return max(
            0,
            min(
                self.page_width,
                int(self.intermediate_width) - feature_page * self.page_width,
            ),
        )

    def real_cipher_slice(self, feature_page: int) -> slice:
        start = int(feature_page) * int(self.canonical.cipher_count)
        return slice(start, start + int(self.canonical.cipher_count))

    def complex_cipher_slice(self, feature_page: int) -> slice:
        start = int(feature_page) * self.paired_token_cipher_count
        return slice(start, start + self.paired_token_cipher_count)

    def pack_intermediate(
        self, values: np.ndarray, *, dtype=np.float64
    ) -> np.ndarray:
        values = np.asarray(values, dtype=dtype)
        expected = (int(self.canonical.seq_len), int(self.intermediate_width))
        if values.shape != expected:
            raise ValueError(
                f"intermediate values must have shape {expected}, got {values.shape}."
            )
        pages: list[np.ndarray] = []
        for page in range(self.feature_page_count):
            active = self.active_width(page)
            start = page * self.page_width
            padded = np.zeros(
                (int(self.canonical.seq_len), self.page_width), dtype=dtype
            )
            padded[:, :active] = values[:, start : start + active]
            pages.append(self.canonical.pack(padded, dtype=dtype))
        return np.concatenate(pages, axis=0)

    def unpack_intermediate(self, packed: np.ndarray) -> np.ndarray:
        packed = np.asarray(packed)
        expected = (self.intermediate_real_cipher_count, int(self.canonical.slots))
        if packed.shape != expected:
            raise ValueError(
                f"intermediate payload must have shape {expected}, got {packed.shape}."
            )
        values = np.empty(
            (int(self.canonical.seq_len), int(self.intermediate_width)),
            dtype=np.float32,
        )
        for page in range(self.feature_page_count):
            active = self.active_width(page)
            start = page * self.page_width
            physical = self.canonical.unpack(packed[self.real_cipher_slice(page)])
            values[:, start : start + active] = physical[:, :active]
        return values


@dataclass(frozen=True)
class FeatureMajorMLPComplexity:
    feature_pages: int
    canonical_real_ciphers: int
    paired_token_ciphers: int
    intermediate_complex_ciphers: int
    baby_steps: int
    plaintext_chunks: int
    reuse_baby_rotations_across_outputs: bool
    gate_up_encoded_weight_plaintexts: int
    gate_up_pt_ct_multiplications: int
    gate_up_baby_rotations: int
    gate_up_giant_rotations: int
    gate_up_chunk_alignment_rotations: int
    gate_up_input_conjugations: int
    gate_up_recovery_conjugations: int
    silu_polynomial_evaluations: int
    silu_polynomial_degree: int
    silu_split_conjugations: int
    up_split_conjugations: int
    gated_ct_ct_multiplications: int
    gated_relinearizations: int
    down_encoded_weight_plaintexts: int
    down_pt_ct_multiplications: int
    down_baby_rotations: int
    down_giant_rotations: int
    down_chunk_alignment_rotations: int
    down_accumulation_additions: int
    final_split_conjugations: int
    encoded_weight_plaintexts: int
    pt_ct_multiplications: int
    rotations_including_conjugations: int
    multiplicative_depth: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


@dataclass
class FeatureMajorMLPResult:
    ciphers: tuple[object, ...]
    complexity: FeatureMajorMLPComplexity
    wall_seconds: float
    stage_seconds: dict[str, float]
    output_levels: tuple[int, ...]
    output: np.ndarray | None = None
    expected: np.ndarray | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None

    def release(self) -> None:
        for cipher in self.ciphers:
            release_if_supported(cipher)


def _chunk_rotation_counts(
    dimension: int, *, baby_steps: int, max_plaintext_rows: int
) -> tuple[int, int, int]:
    giant_steps = int(ceil(int(dimension) / int(baby_steps)))
    chunk_giants = max(1, int(max_plaintext_rows) // int(baby_steps))
    sizes = [
        min(chunk_giants, giant_steps - base)
        for base in range(0, giant_steps, chunk_giants)
    ]
    return (
        len(sizes),
        sum(max(0, size - 1) for size in sizes),
        max(0, len(sizes) - 1),
    )


def mlp_complexity(
    layout: FeatureMajorPrefillLayout | None = None,
    *,
    intermediate_width: int = 14336,
    silu_polynomial_degree: int = 48,
    silu_multiplicative_depth: int = 6,
    baby_steps: int = LLAMA3_LINEAR_SCHEDULES.gate_up.baby_steps,
    max_plaintext_rows: int = LLAMA3_LINEAR_SCHEDULES.gate_up.max_plaintext_rows,
    reuse_baby_rotations_across_outputs: bool = True,
    fuse_silu_domain_map: bool = False,
) -> FeatureMajorMLPComplexity:
    """Audit MLP cost; ``silu_multiplicative_depth`` is the PS core only.

    The ordinary Chebyshev-domain map consumes one additional level.  When
    fused into the gate projection that level disappears, while the supplied
    Paterson--Stockmeyer core depth is unchanged.
    """

    layout = FeatureMajorPrefillLayout() if layout is None else layout
    mlp_layout = FeatureMajorMLPLayout(layout, int(intermediate_width))
    dimension = int(layout.hidden_dim)
    baby_steps = int(baby_steps)
    if baby_steps <= 0 or dimension % baby_steps:
        raise ValueError(
            f"baby_steps={baby_steps} must divide hidden_dim={dimension}."
        )
    if int(max_plaintext_rows) < baby_steps:
        raise ValueError("max_plaintext_rows must hold one baby-step group.")
    if int(silu_multiplicative_depth) < 0:
        raise ValueError("silu_multiplicative_depth must be non-negative.")
    chunks, giants_one, alignment_one = _chunk_rotation_counts(
        dimension,
        baby_steps=baby_steps,
        max_plaintext_rows=int(max_plaintext_rows),
    )
    pairs = mlp_layout.paired_token_cipher_count
    pages = mlp_layout.feature_page_count
    evaluated = 2 * pairs
    baby_rounds = 1 if bool(reuse_baby_rotations_across_outputs) else pages

    gate_rows = pages * dimension
    gate_baby = evaluated * baby_rounds * max(0, baby_steps - 1)
    gate_giant = evaluated * pages * giants_one
    gate_alignment = evaluated * pages * alignment_one
    intermediate_complex = pages * pairs

    down_rows = pages * dimension
    down_inputs = intermediate_complex
    down_baby = down_inputs * max(0, baby_steps - 1)
    down_giant = down_inputs * giants_one
    down_alignment = down_inputs * alignment_one

    gate_input_conjugations = pairs
    gate_recovery_conjugations = intermediate_complex
    silu_split_conjugations = intermediate_complex
    up_split_conjugations = intermediate_complex
    final_split_conjugations = pairs
    rotations = (
        gate_baby
        + gate_giant
        + gate_alignment
        + gate_input_conjugations
        + gate_recovery_conjugations
        + silu_split_conjugations
        + up_split_conjugations
        + down_baby
        + down_giant
        + down_alignment
        + final_split_conjugations
    )
    return FeatureMajorMLPComplexity(
        feature_pages=pages,
        canonical_real_ciphers=int(layout.cipher_count),
        paired_token_ciphers=pairs,
        intermediate_complex_ciphers=intermediate_complex,
        baby_steps=baby_steps,
        plaintext_chunks=chunks,
        reuse_baby_rotations_across_outputs=bool(
            reuse_baby_rotations_across_outputs
        ),
        gate_up_encoded_weight_plaintexts=gate_rows,
        gate_up_pt_ct_multiplications=gate_rows * evaluated,
        gate_up_baby_rotations=gate_baby,
        gate_up_giant_rotations=gate_giant,
        gate_up_chunk_alignment_rotations=gate_alignment,
        gate_up_input_conjugations=gate_input_conjugations,
        gate_up_recovery_conjugations=gate_recovery_conjugations,
        silu_polynomial_evaluations=2 * intermediate_complex,
        silu_polynomial_degree=int(silu_polynomial_degree),
        silu_split_conjugations=silu_split_conjugations,
        up_split_conjugations=up_split_conjugations,
        gated_ct_ct_multiplications=2 * intermediate_complex,
        gated_relinearizations=2 * intermediate_complex,
        down_encoded_weight_plaintexts=down_rows,
        down_pt_ct_multiplications=down_rows * pairs,
        down_baby_rotations=down_baby,
        down_giant_rotations=down_giant,
        down_chunk_alignment_rotations=down_alignment,
        down_accumulation_additions=max(0, pages - 1) * pairs,
        final_split_conjugations=final_split_conjugations,
        encoded_weight_plaintexts=gate_rows + down_rows,
        pt_ct_multiplications=gate_rows * evaluated + down_rows * pairs,
        rotations_including_conjugations=rotations,
        multiplicative_depth=(
            1
            + int(silu_multiplicative_depth)
            + int(not bool(fuse_silu_domain_map))
            + 1
            + 1
        ),
    )


def mlp_rotations(
    layout: FeatureMajorPrefillLayout | None = None,
    *,
    operator_config: LinearOperatorConfig | None = None,
) -> tuple[int, ...]:
    layout = FeatureMajorPrefillLayout() if layout is None else layout
    return linear_rotations(
        dimension=int(layout.hidden_dim),
        token_lanes=int(layout.tokens_per_cipher),
        slots=int(layout.slots),
        include_conjugation=True,
        operator_config=(
            LLAMA3_LINEAR_SCHEDULES.gate_up
            if operator_config is None
            else operator_config
        ),
    )


def _pair_real_slots_numpy(
    packed: np.ndarray, *, layout: FeatureMajorPrefillLayout
) -> np.ndarray:
    packed = np.asarray(packed)
    expected = (int(layout.cipher_count), int(layout.slots))
    if packed.shape != expected:
        raise ValueError(f"real FM payload must have shape {expected}, got {packed.shape}.")
    paired: list[np.ndarray] = []
    for index in range(0, int(layout.cipher_count), 2):
        imaginary = (
            packed[index + 1]
            if index + 1 < int(layout.cipher_count)
            else np.zeros((int(layout.slots),), dtype=packed.dtype)
        )
        paired.append(
            np.asarray(packed[index], dtype=np.complex128)
            + 1j * np.asarray(imaginary, dtype=np.complex128)
        )
    return np.stack(paired)


def _split_half_slots_numpy(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.complex128)
    outputs: list[np.ndarray] = []
    for row in packed:
        outputs.append(np.asarray(row + np.conjugate(row), dtype=np.complex128).real)
        outputs.append(
            np.asarray(-1j * (row - np.conjugate(row)), dtype=np.complex128).real
        )
    return np.stack(outputs)


def mlp_packed_numpy(
    hidden_states: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
    *,
    layout: FeatureMajorPrefillLayout | None = None,
    silu=None,
    input_column_scale: np.ndarray | None = None,
    fuse_silu_domain_map: bool = False,
    silu_fit_interval: tuple[float, float] | None = None,
) -> np.ndarray:
    """Execute the proposed carrier algebra and return canonical FM slots.

    This is deliberately more explicit than a direct NumPy MLP.  It exercises
    the real-token pairing, ``A*z/A*conj(z)`` gate/up recovery, nonlinear split,
    and half-scaled down split used by the FHE path.
    """

    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    gate_weight = np.asarray(gate_weight, dtype=np.float64)
    up_weight = np.asarray(up_weight, dtype=np.float64)
    down_weight = np.asarray(down_weight, dtype=np.float64)
    silu = (
        (lambda value: value / (1.0 + np.exp(-value)))
        if silu is None
        else silu
    )
    if input_column_scale is None:
        column_scale = (
            np.ones((int(layout.hidden_dim),), dtype=np.float64)
            if layout is not None
            else None
        )
    else:
        column_scale = np.asarray(input_column_scale, dtype=np.float64).reshape(-1)
    layout = (
        FeatureMajorPrefillLayout(
            seq_len=int(hidden_states.shape[0]),
            hidden_dim=int(hidden_states.shape[1]),
            slots=int(hidden_states.shape[1]) * int(hidden_states.shape[0]),
        )
        if layout is None
        else layout
    )
    expected_hidden = (int(layout.seq_len), int(layout.hidden_dim))
    if hidden_states.shape != expected_hidden:
        raise ValueError(
            f"hidden_states must have shape {expected_hidden}, got {hidden_states.shape}."
        )
    if column_scale is None:
        column_scale = np.ones((int(layout.hidden_dim),), dtype=np.float64)
    if column_scale.shape != (int(layout.hidden_dim),):
        raise ValueError(
            "MLP input_column_scale must have shape "
            f"{(int(layout.hidden_dim),)}, got {column_scale.shape}."
        )
    if not np.all(np.isfinite(column_scale)):
        raise ValueError("MLP input_column_scale must be finite.")
    gate_weight = gate_weight * column_scale[None, :]
    up_weight = up_weight * column_scale[None, :]
    if bool(fuse_silu_domain_map):
        if silu_fit_interval is None:
            raise ValueError(
                "silu_fit_interval is required when fuse_silu_domain_map=True."
            )
        fit_lo, fit_hi = (float(v) for v in silu_fit_interval)
        gate_alpha = 2.0 / (fit_hi - fit_lo)
        gate_beta = -1.0 - 2.0 * fit_lo / (fit_hi - fit_lo)
        original_silu = silu
        silu = lambda value: original_silu((value - gate_beta) / gate_alpha)
    else:
        gate_alpha = 1.0
        gate_beta = 0.0
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have the same shape.")
    intermediate_width, hidden_dim = gate_weight.shape
    if hidden_dim != int(layout.hidden_dim):
        raise ValueError("gate/up input width must match layout.hidden_dim.")
    if down_weight.shape != (hidden_dim, intermediate_width):
        raise ValueError(
            f"down_weight must have shape {(hidden_dim, intermediate_width)}, "
            f"got {down_weight.shape}."
        )
    mlp_layout = FeatureMajorMLPLayout(layout, intermediate_width)
    paired = _pair_real_slots_numpy(layout.pack(hidden_states), layout=layout)
    token_lanes = int(layout.tokens_per_cipher)
    paired_rows = paired.reshape(
        mlp_layout.paired_token_cipher_count,
        int(layout.hidden_dim),
        token_lanes,
    ).transpose(0, 2, 1)

    gated_complex_pages: list[np.ndarray] = []
    for page in range(mlp_layout.feature_page_count):
        active = mlp_layout.active_width(page)
        start = page * mlp_layout.page_width
        projection = np.zeros(
            (mlp_layout.page_width, mlp_layout.page_width),
            dtype=np.complex128,
        )
        projection[:active] = 0.25 * (
            gate_alpha * gate_weight[start : start + active]
            + 1j * up_weight[start : start + active]
        )
        for z in paired_rows:
            direct = z @ projection.T
            conjugate_evaluation = np.conjugate(z) @ projection.T
            conjugated_output = np.conjugate(conjugate_evaluation)
            gate_half = direct + conjugated_output
            up_half = -1j * (direct - conjugated_output)
            gate_real = gate_half + np.conjugate(gate_half)
            gate_imag = (-1j * (gate_half - np.conjugate(gate_half))).real
            up_real = up_half + np.conjugate(up_half)
            up_imag = (-1j * (up_half - np.conjugate(up_half))).real
            gated_complex_pages.append(
                np.asarray(
                    silu(gate_real.real + gate_beta) * up_real.real
                    + 1j * silu(gate_imag + gate_beta) * up_imag,
                    dtype=np.complex128,
                )
            )

    down_accumulators = [
        np.zeros((token_lanes, int(layout.hidden_dim)), dtype=np.complex128)
        for _ in range(mlp_layout.paired_token_cipher_count)
    ]
    for page in range(mlp_layout.feature_page_count):
        active = mlp_layout.active_width(page)
        start = page * mlp_layout.page_width
        projection = np.zeros(
            (mlp_layout.page_width, mlp_layout.page_width), dtype=np.float64
        )
        projection[:, :active] = 0.5 * down_weight[:, start : start + active]
        page_start = page * mlp_layout.paired_token_cipher_count
        for pair in range(mlp_layout.paired_token_cipher_count):
            down_accumulators[pair] += (
                gated_complex_pages[page_start + pair] @ projection.T
            )

    half_slots = np.stack(
        [value.T.reshape(int(layout.slots)) for value in down_accumulators]
    )
    real_slots = _split_half_slots_numpy(half_slots)[: int(layout.cipher_count)]
    return np.asarray(real_slots, dtype=np.float64)


def _add_profile(
    destination: dict[str, float], prefix: str, profile: dict[str, float]
) -> None:
    for name, seconds in profile.items():
        key = f"{prefix}_{name}"
        destination[key] = float(destination.get(key, 0.0) + float(seconds))


def mlp_fhe(
    hidden_states: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
    *,
    silu_entry: dict,
    layout: FeatureMajorPrefillLayout,
    config: Llama3CKKSConfig | None = None,
    operator_config: LinearOperatorConfig | None = None,
    crypto_context=None,
    input_ciphers: tuple[object, ...] | list[object] | None = None,
    verify: bool = False,
    input_column_scale: np.ndarray | None = None,
    fuse_silu_domain_map: bool = False,
) -> FeatureMajorMLPResult:
    """Run the selected ``T=hidden_dim`` feature-major SwiGLU MLP.

    Gate/up uses one persistent :class:`LinearOperator`, so its baby rotations
    are computed on the first output page and reused by the remaining three.
    Down uses a separate operator because every feature page consumes different
    intermediate ciphertexts and therefore has no cross-page baby identity to
    reuse.
    """

    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.gate_up
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
    config_slots = 1 << (int(config.simulator.logN) - 1)
    if int(layout.slots) != config_slots:
        raise ValueError(
            "feature-major MLP layout/config slot mismatch: "
            f"layout.slots={layout.slots}, config slots={config_slots}."
        )
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    gate_weight = np.asarray(gate_weight, dtype=np.float32)
    up_weight = np.asarray(up_weight, dtype=np.float32)
    down_weight = np.asarray(down_weight, dtype=np.float32)
    expected_hidden = (int(layout.seq_len), int(layout.hidden_dim))
    if hidden_states.shape != expected_hidden:
        raise ValueError(
            f"hidden_states must have shape {expected_hidden}, got {hidden_states.shape}."
        )
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have the same shape.")
    intermediate_width, hidden_dim = gate_weight.shape
    if hidden_dim != int(layout.hidden_dim):
        raise ValueError("gate/up input width must match layout.hidden_dim.")
    if down_weight.shape != (hidden_dim, intermediate_width):
        raise ValueError(
            f"down_weight must have shape {(hidden_dim, intermediate_width)}, "
            f"got {down_weight.shape}."
        )
    if int(layout.cipher_count) % 2:
        raise ValueError("FHE feature-major MLP requires an even canonical cipher count.")
    if not bool(reuse_baby_rotations):
        raise ValueError(
            "the feature-major FHE operator requires "
            "reuse_baby_rotations=True."
        )

    coefficients = np.asarray(silu_entry["coefficients"], dtype=np.float64)
    lower_bound, upper_bound = (
        float(value) for value in silu_entry["fit_interval"]
    )
    column_scale = None
    if input_column_scale is not None:
        column_scale = np.asarray(input_column_scale, dtype=np.float32).reshape(-1)
        if column_scale.shape != (hidden_dim,):
            raise ValueError(
                "MLP input_column_scale must have shape "
                f"{(hidden_dim,)}, got {column_scale.shape}."
            )
        if not np.all(np.isfinite(column_scale)):
            raise ValueError("MLP input_column_scale must be finite.")
    gate_alpha = 2.0 / (upper_bound - lower_bound)
    gate_beta = -1.0 - 2.0 * lower_bound / (upper_bound - lower_bound)
    mlp_layout = FeatureMajorMLPLayout(layout, intermediate_width)
    complexity = mlp_complexity(
        layout,
        intermediate_width=intermediate_width,
        silu_polynomial_degree=max(0, int(coefficients.size) - 1),
        silu_multiplicative_depth=chebyshev_ps_mul_depth(coefficients),
        baby_steps=int(baby_steps),
        max_plaintext_rows=int(max_plaintext_rows),
        reuse_baby_rotations_across_outputs=True,
        fuse_silu_domain_map=bool(fuse_silu_domain_map),
    )
    if crypto_context is None or input_ciphers is None:
        raise ValueError(
            "mlp_fhe requires an application-owned context "
            "and encrypted feature-major inputs."
        )
    context_slots = int(crypto_context.max_slots)
    if context_slots != int(layout.slots):
        raise ValueError(
            "feature-major MLP layout/context slot mismatch: "
            f"layout.slots={layout.slots}, context max_slots={context_slots}."
        )

    wall_start = time.perf_counter()
    stage: dict[str, float] = {}
    owns_inputs = False
    input_ciphers = tuple(input_ciphers)
    if len(input_ciphers) != int(layout.cipher_count):
        raise ValueError(
            f"expected {layout.cipher_count} input ciphers, got {len(input_ciphers)}."
        )
    input_levels = {
        int(crypto_context.level_for_cipher(cipher)) for cipher in input_ciphers
    }
    if len(input_levels) != 1:
        if owns_inputs:
            for cipher in input_ciphers:
                release_if_supported(cipher)
        raise ValueError(
            "feature-major MLP inputs must enter at one common level."
        )
    paired_inputs: tuple[object, ...] = ()
    conjugated_inputs: tuple[object, ...] = ()
    gate_half: list[object] = []
    up_half: list[object] = []
    gated_complex: list[object] = []
    down_accumulators: list[object | None] = [
        None for _ in range(mlp_layout.paired_token_cipher_count)
    ]
    outputs: list[object] = []
    gate_up_operator = LinearOperator(
        crypto_context=crypto_context,
        token_lanes=int(layout.tokens_per_cipher),
        config=LinearOperatorConfig(
            baby_steps=int(baby_steps),
            baby_anchor_step=int(baby_anchor_step),
            max_plaintext_rows=int(max_plaintext_rows),
            reuse_baby_rotations=bool(reuse_baby_rotations),
            hoist_strategy=str(hoist_strategy),
        ),
    )
    down_operator = LinearOperator(
        crypto_context=crypto_context,
        token_lanes=int(layout.tokens_per_cipher),
        config=LinearOperatorConfig(
            baby_steps=int(baby_steps),
            baby_anchor_step=int(baby_anchor_step),
            max_plaintext_rows=int(max_plaintext_rows),
            reuse_baby_rotations=bool(reuse_baby_rotations),
            hoist_strategy=str(hoist_strategy),
        ),
    )
    success = False

    def debug_complex_range(label: str, ciphers) -> None:
        if os.environ.get("LLAMA_FHE_STAGE_DIAGNOSTICS", "") != "1":
            return
        only = os.environ.get("LLAMA_FHE_STAGE_DIAGNOSTICS_LAYER", "").strip()
        current = os.environ.get("LLAMA_FHE_CURRENT_LAYER", "").strip()
        if only and current and int(only) != int(current):
            return
        values = np.stack(
            [np.asarray(crypto_context.decrypt(cipher)) for cipher in ciphers]
        )
        layer_tag = current if current else "?"
        print(
            f"[mlp-diagnostic] layer={int(layer_tag) + 1 if layer_tag != '?' else '?'} "
            f"stage={label} "
            f"real_max={float(np.max(np.abs(values.real)))} "
            f"imag_max={float(np.max(np.abs(values.imag)))} "
            f"real_mean={float(np.mean(np.abs(values.real)))} "
            f"imag_mean={float(np.mean(np.abs(values.imag)))}",
            flush=True,
        )

    try:
        start = time.perf_counter()
        paired_inputs = pair_real_ciphers(
            input_ciphers, crypto_context=crypto_context
        )
        conjugated_list: list[object] = []
        try:
            for cipher in paired_inputs:
                conjugated_list.append(
                    crypto_context.fhe.homo_rotate(
                        cipher,
                        int(crypto_context.context.M) - 1,
                        crypto_context.context,
                    )
                )
            conjugated_inputs = tuple(conjugated_list)
            conjugated_list = []
        finally:
            for cipher in conjugated_list:
                release_if_supported(cipher)
        sync_device(crypto_context.device)
        stage["input_pair_and_conjugate"] = time.perf_counter() - start

        dtype = np.dtype(np.complex128)
        use_structured_weight = structured_weight_packing_enabled()
        for page in range(mlp_layout.feature_page_count):
            weight_transform_start = time.perf_counter()
            active = mlp_layout.active_width(page)
            feature_start = page * mlp_layout.page_width
            if use_structured_weight:
                output_rows = np.arange(
                    mlp_layout.page_width, dtype=np.int64
                )
                active_rows = output_rows < active
                source_map = np.where(active_rows, 0, -1)
                source_rows = np.where(active_rows, output_rows, -1)
                weight = structured_linear_weight(
                    (
                        gate_weight[feature_start : feature_start + active],
                        up_weight[feature_start : feature_start + active],
                    ),
                    dimension=mlp_layout.page_width,
                    real_source=source_map,
                    real_row=source_rows,
                    real_scale=(
                        0.25
                        * (
                            gate_alpha
                            if bool(fuse_silu_domain_map)
                            else 1.0
                        )
                    ),
                    imag_source=np.where(active_rows, 1, -1),
                    imag_row=source_rows,
                    imag_scale=0.25,
                    column_scale=(
                        1.0 if column_scale is None else column_scale
                    ),
                )
            else:
                weight = np.zeros(
                    (mlp_layout.page_width, mlp_layout.page_width),
                    dtype=np.complex128,
                )
                gate_page = gate_weight[
                    feature_start : feature_start + active
                ].astype(np.complex128)
                up_page = up_weight[
                    feature_start : feature_start + active
                ].astype(np.complex128)
                if column_scale is not None:
                    gate_page *= column_scale[None, :]
                    up_page *= column_scale[None, :]
                weight[:active] = 0.25 * (
                    (gate_alpha if bool(fuse_silu_domain_map) else 1.0)
                    * gate_page
                    + 1j * up_page
                )
            stage["gate_up_weight_transform_cpu"] = float(
                stage.get("gate_up_weight_transform_cpu", 0.0)
                + time.perf_counter()
                - weight_transform_start
            )
            projected, profile = gate_up_operator.project(
                paired_inputs + conjugated_inputs,
                weight,
                dtype=dtype,
            )
            if not baby_reuse_across_pages_enabled():
                gate_up_operator.clear_baby_cache()
            _add_profile(stage, "gate_up_projection", profile)
            try:
                pairs = mlp_layout.paired_token_cipher_count
                for direct, conjugate_evaluation in zip(
                    projected[:pairs], projected[pairs:], strict=True
                ):
                    conjugated_output = crypto_context.fhe.homo_rotate(
                        conjugate_evaluation,
                        int(crypto_context.context.M) - 1,
                        crypto_context.context,
                    )
                    try:
                        gate_half.append(
                            crypto_context.fhe.homo_add(
                                direct, conjugated_output, crypto_context.context
                            )
                        )
                        delta = crypto_context.fhe.homo_sub(
                            direct, conjugated_output, crypto_context.context
                        )
                        try:
                            up_half.append(
                                fhe.homo_mul_i(
                                    delta,
                                    crypto_context.context,
                                    negative=True,
                                )
                            )
                        finally:
                            release_if_supported(delta)
                    finally:
                        release_if_supported(conjugated_output)
            finally:
                for cipher in projected:
                    release_if_supported(cipher)
        gate_up_operator.clear_baby_cache()
        debug_complex_range("gate", gate_half)
        debug_complex_range("up", up_half)

        fine_profile = os.environ.get("LLAMA_FHE_FINE_PROFILE", "") == "1"
        fine_totals: dict[str, float] = {
            "silu_gated.split_complex": 0.0,
            "silu_gated.domain_map": 0.0,
            "silu_gated.silu_polynomial": 0.0,
            "silu_gated.gate_mul_up": 0.0,
            "silu_gated.repack_complex": 0.0,
        }

        def _fine_add(name: str, started: float) -> float:
            sync_device(crypto_context.device)
            finished = time.perf_counter()
            fine_totals[name] += float(finished - started)
            return finished

        start = time.perf_counter()
        if fine_profile:
            sync_device(crypto_context.device)
            fine_started = time.perf_counter()
        for gate_cipher, up_cipher in zip(gate_half, up_half, strict=True):
            gate_real = gate_imag = up_real = up_imag = None
            mapped_real = mapped_imag = None
            activated_real = activated_imag = None
            gated_real = gated_imag = imaginary_i = None
            try:
                gate_real, gate_imag = split_complex_cipher_twice(
                    gate_cipher, crypto_context=crypto_context
                )
                up_real, up_imag = split_complex_cipher_twice(
                    up_cipher, crypto_context=crypto_context
                )
                if fine_profile:
                    fine_started = _fine_add(
                        "silu_gated.split_complex", fine_started
                    )
                if bool(fuse_silu_domain_map):
                    beta_real = fhe.encode_scalar(
                        gate_beta,
                        cur_limbs=int(gate_real.state.cur_limbs),
                        scale_degree=1,
                        scaling_factor=gate_real.state.scaling_factor,
                        context=crypto_context.context,
                    )
                    beta_imag = fhe.encode_scalar(
                        gate_beta,
                        cur_limbs=int(gate_imag.state.cur_limbs),
                        scale_degree=1,
                        scaling_factor=gate_imag.state.scaling_factor,
                        context=crypto_context.context,
                    )
                    try:
                        mapped_real = fhe.homo_add_scalar(
                            gate_real, beta_real, crypto_context.context
                        )
                        mapped_imag = fhe.homo_add_scalar(
                            gate_imag, beta_imag, crypto_context.context
                        )
                    except Exception:
                        release_if_supported(mapped_real)
                        release_if_supported(mapped_imag)
                        mapped_real = mapped_imag = None
                        raise
                    release_if_supported(gate_real)
                    release_if_supported(gate_imag)
                    gate_real, gate_imag = mapped_real, mapped_imag
                    mapped_real = mapped_imag = None
                if fine_profile:
                    fine_started = _fine_add(
                        "silu_gated.domain_map", fine_started
                    )
                activated_real = eval_silu_chebyshev_cipher(
                    gate_real,
                    coefficients,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    crypto_context=crypto_context,
                    input_is_chebyshev_mapped=bool(fuse_silu_domain_map),
                )
                activated_imag = eval_silu_chebyshev_cipher(
                    gate_imag,
                    coefficients,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    crypto_context=crypto_context,
                    input_is_chebyshev_mapped=bool(fuse_silu_domain_map),
                )
                if fine_profile:
                    fine_started = _fine_add(
                        "silu_gated.silu_polynomial", fine_started
                    )
                gated_real = _mul_cipher_rescale(
                    activated_real, up_real, crypto_context=crypto_context
                )
                gated_imag = _mul_cipher_rescale(
                    activated_imag, up_imag, crypto_context=crypto_context
                )
                if fine_profile:
                    fine_started = _fine_add(
                        "silu_gated.gate_mul_up", fine_started
                    )
                imaginary_i = fhe.homo_mul_i(
                    gated_imag, crypto_context.context
                )
                gated_complex.append(
                    crypto_context.fhe.homo_add(
                        gated_real, imaginary_i, crypto_context.context
                    )
                )
                if fine_profile:
                    fine_started = _fine_add(
                        "silu_gated.repack_complex", fine_started
                    )
            finally:
                for cipher in (
                    gate_real,
                    gate_imag,
                    mapped_real,
                    mapped_imag,
                    up_real,
                    up_imag,
                    activated_real,
                    activated_imag,
                    gated_real,
                    gated_imag,
                    imaginary_i,
                ):
                    release_if_supported(cipher)
        sync_device(crypto_context.device)
        stage["silu_gated"] = time.perf_counter() - start
        if fine_profile:
            for key, value in fine_totals.items():
                stage[key] = float(value)
        debug_complex_range("gated", gated_complex)
        for cipher in gate_half + up_half:
            release_if_supported(cipher)
        gate_half = []
        up_half = []

        for page in range(mlp_layout.feature_page_count):
            page_inputs = tuple(
                gated_complex[mlp_layout.complex_cipher_slice(page)]
            )
            weight_transform_start = time.perf_counter()
            active = mlp_layout.active_width(page)
            feature_start = page * mlp_layout.page_width
            if use_structured_weight:
                columns = np.full(
                    (mlp_layout.page_width,), -1, dtype=np.int64
                )
                columns[:active] = np.arange(active, dtype=np.int64)
                rows = np.arange(mlp_layout.page_width, dtype=np.int64)
                weight = structured_linear_weight(
                    (down_weight[:, feature_start : feature_start + active],),
                    dimension=mlp_layout.page_width,
                    real_source=np.zeros(
                        (mlp_layout.page_width,), dtype=np.int64
                    ),
                    real_row=rows,
                    real_scale=0.5,
                    column=columns,
                )
            else:
                weight = np.zeros(
                    (mlp_layout.page_width, mlp_layout.page_width),
                    dtype=np.float64,
                )
                weight[:, :active] = 0.5 * down_weight[
                    :, feature_start : feature_start + active
                ]
            stage["down_weight_transform_cpu"] = float(
                stage.get("down_weight_transform_cpu", 0.0)
                + time.perf_counter()
                - weight_transform_start
            )
            projected, profile = down_operator.project(
                page_inputs,
                weight,
                dtype=np.dtype(np.float64),
            )
            if not baby_reuse_across_pages_enabled():
                down_operator.clear_baby_cache()
            _add_profile(stage, "down_projection", profile)
            projected_terms: list[object | None] = list(projected)
            try:
                for index, term in enumerate(projected_terms):
                    if term is None:
                        raise RuntimeError("down projection returned an empty term.")
                    current = down_accumulators[index]
                    if current is None:
                        down_accumulators[index] = term
                        projected_terms[index] = None
                        continue
                    combined = crypto_context.fhe.homo_add(
                        current, term, crypto_context.context
                    )
                    release_if_supported(current)
                    release_if_supported(term)
                    down_accumulators[index] = combined
                    projected_terms[index] = None
            finally:
                for term in projected_terms:
                    release_if_supported(term)
            # Every down page consumes a distinct intermediate cipher set.
            # Its baby rotations cannot hit on the next page, so retaining
            # them until the whole MLP closes only increases the live set.
            down_operator.clear_baby_cache()
        for cipher in gated_complex:
            release_if_supported(cipher)
        gated_complex = []

        start = time.perf_counter()
        for carrier in down_accumulators:
            if carrier is None:
                raise RuntimeError("down projection produced an empty accumulator.")
            real, imaginary = split_complex_cipher_twice(
                carrier, crypto_context=crypto_context
            )
            outputs.extend((real, imaginary))
        outputs = outputs[: int(layout.cipher_count)]
        sync_device(crypto_context.device)
        stage["final_split"] = time.perf_counter() - start
        debug_complex_range("down_real", outputs)

        output = expected = None
        max_abs_diff = mean_abs_diff = None
        if verify:
            start = time.perf_counter()
            decrypted = np.stack(
                [
                    np.asarray(crypto_context.decrypt(cipher), dtype=np.float64)
                    for cipher in outputs
                ]
            )
            output = layout.unpack(decrypted)
            scaled_hidden = (
                hidden_states
                if column_scale is None
                else hidden_states * column_scale[None, :]
            )
            gate_reference = scaled_hidden @ gate_weight.T
            up_reference = scaled_hidden @ up_weight.T
            activated_reference = eval_chebyshev_series_direct(
                gate_reference,
                coefficients,
                lower_bound,
                upper_bound,
            ).astype(np.float32, copy=False)
            expected = (
                (activated_reference * up_reference) @ down_weight.T
            ).astype(np.float32, copy=False)
            difference = output - expected
            max_abs_diff = float(np.max(np.abs(difference)))
            mean_abs_diff = float(np.mean(np.abs(difference)))
            sync_device(crypto_context.device)
            stage["verify_decrypt"] = time.perf_counter() - start

        success = True
        return FeatureMajorMLPResult(
            ciphers=tuple(outputs),

            complexity=complexity,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={key: float(value) for key, value in stage.items()},
            output_levels=tuple(
                sorted(
                    {
                        int(crypto_context.level_for_cipher(cipher))
                        for cipher in outputs
                    }
                )
            ),
            output=output,
            expected=expected,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
        )
    finally:
        gate_up_operator.close()
        down_operator.close()
        for cipher in paired_inputs + conjugated_inputs:
            release_if_supported(cipher)
        if owns_inputs:
            for cipher in input_ciphers:
                release_if_supported(cipher)
        for cipher in gate_half + up_half + gated_complex:
            release_if_supported(cipher)
        for cipher in down_accumulators:
            release_if_supported(cipher)
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)
