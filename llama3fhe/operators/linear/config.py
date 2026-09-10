from __future__ import annotations

"""Operator-owned scheduling for feature-major diagonal linear maps."""

import os
from dataclasses import dataclass, field

_CACHE_MODES = {"none", "middle", "plain", "both"}
_HOIST_STRATEGIES = {"normal", "ext_normal", "ext_double_hoist"}
_RAW_WEIGHT_POLICIES = {"device", "host"}


def structured_weight_packing_enabled() -> bool:
    """Whether the Triton packer reads model weights through a descriptor.

    The default. ``LLAMA_FHE_STRUCTURED_LINEAR_PACK=0`` selects the earlier
    path, which materializes a dense square physical matrix on the CPU first:
    measured 4.0-4.5% slower per layer and +159 MiB of allocator peak
    It is retained as the rollback for that default, so this predicate exists
    once rather than as three separate environment reads.
    """

    return os.environ.get("LLAMA_FHE_STRUCTURED_LINEAR_PACK", "1") != "0"


def linear_encode_reuse_enabled() -> bool:
    """Whether one encoded diagonal batch is reused across input ciphertexts.

    The default. ``LLAMA_FHE_REUSE_LINEAR_ENCODE=0`` re-encodes the same
    diagonal chunk once per input ciphertext (the no-reuse formula in the
    linear ablation). Baby/MAC/merge are unchanged.
    """

    return os.environ.get("LLAMA_FHE_REUSE_LINEAR_ENCODE", "1") != "0"


def baby_reuse_across_pages_enabled() -> bool:
    """Whether gate/up (and down) pages share one baby-rotation cache.

    The default. ``LLAMA_FHE_REUSE_BABY_ACROSS_PAGES=0`` clears the operator
    baby cache after each physical page so later pages recompute baby
    rotations. Chunk-level reuse inside one page is unchanged.
    """

    return os.environ.get("LLAMA_FHE_REUSE_BABY_ACROSS_PAGES", "1") != "0"


@dataclass(frozen=True)
class LinearOperatorConfig:
    """Execution and cache policy owned by :class:`LinearOperator`.

    There are only two baby-schedule parameters: ``baby_steps`` (B) and
    ``baby_anchor_step``.  The latter is the number of logical baby steps in
    each fast-rotation block.  For ``B=128, anchor=32``, the operator reaches
    anchors 0/32/64/96 sequentially and runs one local fast rotation covering
    0..31 from each anchor.  The physical spacing is inferred from
    ``token_lanes``; no third schedule parameter exists.  Set the anchor to
    ``-1`` for one traditional unbounded fast batch.

    ConstantBundle policy also belongs here: a weight owns cached constants,
    but the operator decides which preparation stages should be retained.
    """

    baby_steps: int = 32
    baby_anchor_step: int = 32
    max_plaintext_rows: int = 512
    reuse_baby_rotations: bool = True
    hoist_strategy: str = "normal"
    constant_cache_mode: str = "none"
    raw_weight_policy: str = "device"

    def __post_init__(self) -> None:
        if int(self.baby_steps) <= 0:
            raise ValueError("baby_steps must be positive.")
        if int(self.baby_anchor_step) in (0, 1):
            raise ValueError("baby_anchor_step must be at least two or -1.")
        if int(self.baby_anchor_step) < -1:
            raise ValueError("baby_anchor_step must be positive or -1.")
        if int(self.max_plaintext_rows) < int(self.baby_steps):
            raise ValueError("max_plaintext_rows must hold one baby-step group.")
        if str(self.hoist_strategy).lower() not in _HOIST_STRATEGIES:
            raise ValueError(
                "hoist_strategy must be 'normal', 'ext_normal', or "
                "'ext_double_hoist'."
            )
        if str(self.constant_cache_mode) not in _CACHE_MODES:
            raise ValueError(
                "constant_cache_mode must be one of "
                f"{sorted(_CACHE_MODES)}, got {self.constant_cache_mode!r}."
            )
        if str(self.raw_weight_policy) not in _RAW_WEIGHT_POLICIES:
            raise ValueError(
                "raw_weight_policy must be 'device' or 'host', got "
                f"{self.raw_weight_policy!r}."
            )

    def plan(self, *, dimension: int, token_lanes: int) -> "LinearPlan":
        return LinearPlan(
            dimension=int(dimension),
            token_lanes=int(token_lanes),
            baby_steps=int(self.baby_steps),
            baby_anchor_step=int(self.baby_anchor_step),
            max_plaintext_rows=int(self.max_plaintext_rows),
        )


def _production_linear_config() -> LinearOperatorConfig:
    """A100-selected production schedule for one 4096-wide FM linear.

    ``baby_steps=64`` re-tuned at dnum>=3 (2026-08-16 sweep: 64 beats the old
    dnum=2 optimum of 32 by 6.7% on layer0 and is the largest split the
    backend's 64-key innerproduct path admits; 16 and 128 both lose, and
    anchor variants 32/-1 measured worse/neutral).
    """

    return LinearOperatorConfig(
        baby_steps=64,
        baby_anchor_step=64,
        max_plaintext_rows=512,
        reuse_baby_rotations=True,
        hoist_strategy="normal",
        constant_cache_mode="none",
        raw_weight_policy="device",
    )


@dataclass(frozen=True)
class Llama3LinearSchedules:
    """Named operator schedules for the four Llama3 projection boundaries.

    These are separate even though the current measured optimum is identical.
    Gate/up and down have different input reuse, so future backend changes can
    tune them independently without moving execution policy onto a weight.
    """

    qkv: LinearOperatorConfig = field(default_factory=_production_linear_config)
    w_o: LinearOperatorConfig = field(default_factory=_production_linear_config)
    gate_up: LinearOperatorConfig = field(default_factory=_production_linear_config)
    down: LinearOperatorConfig = field(default_factory=_production_linear_config)


LLAMA3_LINEAR_SCHEDULES = Llama3LinearSchedules()


@dataclass(frozen=True)
class LinearChunk:
    index: int
    giant_base: int
    giant_count: int

    @property
    def name(self) -> str:
        return f"chunk.{int(self.index)}"


@dataclass(frozen=True)
class BabyBlock:
    start: int
    width: int


@dataclass(frozen=True)
class LinearPlan:
    """Validated physical BSGS schedule, independent of any weight."""

    dimension: int
    token_lanes: int
    baby_steps: int
    baby_anchor_step: int
    max_plaintext_rows: int

    def __post_init__(self) -> None:
        if int(self.dimension) <= 0:
            raise ValueError("dimension must be positive.")
        if int(self.token_lanes) <= 0:
            raise ValueError("token_lanes must be positive.")
        if int(self.baby_steps) <= 0:
            raise ValueError("baby_steps must be positive.")
        if int(self.dimension) % int(self.baby_steps):
            raise ValueError(
                f"baby_steps={self.baby_steps} must divide "
                f"dimension={self.dimension}."
            )
        if int(self.baby_anchor_step) in (0, 1) or int(self.baby_anchor_step) < -1:
            raise ValueError("baby_anchor_step must be at least two or -1.")
        if int(self.max_plaintext_rows) < int(self.baby_steps):
            raise ValueError("max_plaintext_rows must hold one baby-step group.")

    @property
    def giant_steps(self) -> int:
        return int(self.dimension) // int(self.baby_steps)

    @property
    def chunk_giants(self) -> int:
        return max(1, int(self.max_plaintext_rows) // int(self.baby_steps))

    @property
    def giant_offset(self) -> int:
        return int(self.baby_steps) * int(self.token_lanes)

    @property
    def baby_offsets(self) -> tuple[int, ...]:
        return tuple(
            baby * int(self.token_lanes) for baby in range(int(self.baby_steps))
        )

    @property
    def baby_blocks(self) -> tuple[BabyBlock, ...]:
        width = (
            int(self.baby_anchor_step)
            if self.anchor_enabled
            else int(self.baby_steps)
        )
        return tuple(
            BabyBlock(start=start, width=min(width, int(self.baby_steps) - start))
            for start in range(0, int(self.baby_steps), width)
        )

    @property
    def baby_rotation_count(self) -> int:
        """Rotation-like operations used to produce all baby steps once."""

        return max(0, int(self.baby_steps) - 1)

    @property
    def baby_key_offsets(self) -> tuple[int, ...]:
        block_width = int(self.baby_blocks[0].width)
        offsets = {
            step * int(self.token_lanes) for step in range(1, block_width)
        }
        if len(self.baby_blocks) > 1:
            offsets.add(int(self.baby_anchor_step) * int(self.token_lanes))
        return tuple(sorted(offsets))

    @property
    def anchor_enabled(self) -> bool:
        return bool(
            int(self.baby_anchor_step) >= 2
            and int(self.baby_steps) > int(self.baby_anchor_step)
        )

    @property
    def chunks(self) -> tuple[LinearChunk, ...]:
        return tuple(
            LinearChunk(
                index=index,
                giant_base=giant_base,
                giant_count=min(
                    self.chunk_giants, self.giant_steps - giant_base
                ),
            )
            for index, giant_base in enumerate(
                range(0, self.giant_steps, self.chunk_giants)
            )
        )

    @property
    def chunk_sizes(self) -> tuple[int, ...]:
        return tuple(chunk.giant_count for chunk in self.chunks)

    @property
    def plaintext_chunks(self) -> int:
        return len(self.chunks)

    @property
    def cache_key(self) -> tuple[int, ...]:
        return (
            int(self.dimension),
            int(self.token_lanes),
            int(self.baby_steps),
            int(self.max_plaintext_rows),
        )


def linear_rotations(
    *,
    dimension: int,
    token_lanes: int,
    slots: int | None = None,
    include_conjugation: bool = False,
    operator_config: LinearOperatorConfig | None = None,
) -> tuple[int, ...]:
    """Return the exact evaluation-key union for one operator schedule."""

    operator_config = (
        LinearOperatorConfig() if operator_config is None else operator_config
    )
    plan = operator_config.plan(
        dimension=int(dimension), token_lanes=int(token_lanes)
    )
    rotations = set(plan.baby_key_offsets)
    strategy = str(operator_config.hoist_strategy).lower()
    max_chunk_giants = max(chunk.giant_count for chunk in plan.chunks)
    if max_chunk_giants > 1:
        if strategy == "ext_double_hoist":
            # Double hoist directly rotates every local giant accumulator in
            # one chunk, so each local offset needs its own key.
            rotations.update(
                step * plan.giant_offset
                for step in range(1, int(max_chunk_giants))
            )
        else:
            # normal/ext_normal use a sequential Horner chain and therefore
            # repeat one giant rotation key within each chunk.
            rotations.add(plan.giant_offset)
    # Independently align each completed plaintext chunk in the global giant
    # axis.  These keys are required by every strategy.
    rotations.update(
        int(chunk.giant_base) * plan.giant_offset
        for chunk in plan.chunks
        if int(chunk.giant_base)
    )
    if include_conjugation:
        if slots is None:
            raise ValueError("slots are required when conjugation is requested.")
        rotations.add(4 * int(slots) - 1)
    return tuple(sorted(int(value) for value in rotations if int(value)))
