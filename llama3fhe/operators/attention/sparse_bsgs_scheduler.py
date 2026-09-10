from __future__ import annotations

"""Pure scheduling helpers for the attention input sparse BSGS kernels.

The objects in this module contain only integer route metadata.  They make it
possible to test batching and operation-count preservation without importing
or constructing an FHE context.  Ciphertext execution lives in ``input.py``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseMaskRoute:
    """Destination slots and their public coefficients for one diagonal.

    A bare integer array is the shorthand for unit coefficients, and is what
    the release schedule's V-delta and key-shear routes use. This explicit
    form carries a per-destination coefficient, including complex ones; the
    scheduler is layout-agnostic and supports it, and
    ``test_batched_sparse_bsgs_fhe_matches_reference_with_complex_routes``
    holds that contract even though the release routes are all unit.
    """

    destinations: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        destinations = np.asarray(self.destinations, dtype=np.int64)
        coefficients = np.asarray(self.coefficients)
        if destinations.ndim != 1 or coefficients.ndim != 1:
            raise ValueError("sparse route destinations and coefficients must be 1-D")
        if destinations.shape != coefficients.shape:
            raise ValueError(
                "sparse route destinations and coefficients must have equal length"
            )
        if len(np.unique(destinations)) != destinations.size:
            raise ValueError("one sparse diagonal cannot write a destination twice")
        object.__setattr__(self, "destinations", destinations)
        object.__setattr__(self, "coefficients", coefficients)


SparseRouteValue = np.ndarray | SparseMaskRoute


def sparse_route_arrays(route: SparseRouteValue) -> tuple[np.ndarray, np.ndarray]:
    """Normalize either route form to (destinations, coefficients)."""

    if isinstance(route, SparseMaskRoute):
        return route.destinations, route.coefficients
    destinations = np.asarray(route, dtype=np.int64)
    if destinations.ndim != 1:
        raise ValueError("sparse route destinations must be 1-D")
    return destinations, np.ones(destinations.shape, dtype=np.float64)


@dataclass(frozen=True)
class SparseRouteGroup:
    """All sparse diagonals accumulated before one giant-step rotation."""

    giant_offset: int
    diagonal_offsets: tuple[int, ...]
    baby_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.diagonal_offsets:
            raise ValueError("a sparse route group must not be empty")
        if len(self.diagonal_offsets) != len(self.baby_offsets):
            raise ValueError("diagonal_offsets and baby_offsets must have equal length")

    @property
    def row_count(self) -> int:
        return len(self.diagonal_offsets)


@dataclass(frozen=True)
class SparseRouteBatch:
    """One bounded GPU plaintext-encoding batch of complete giant groups."""

    groups: tuple[SparseRouteGroup, ...]
    padding_rows: int = 0

    def __post_init__(self) -> None:
        if not self.groups and int(self.padding_rows) <= 0:
            raise ValueError("a sparse route batch must contain routes or padding")
        if int(self.padding_rows) < 0:
            raise ValueError("padding_rows must be non-negative")

    @property
    def route_rows(self) -> int:
        return sum(group.row_count for group in self.groups)

    @property
    def encoded_rows(self) -> int:
        return self.route_rows + int(self.padding_rows)


@dataclass(frozen=True)
class SparseRouteSchedule:
    """A sparse BSGS route partition with an explicit encode-batch cap."""

    modulus: int
    baby_offsets: tuple[int, ...]
    groups: tuple[SparseRouteGroup, ...]
    batches: tuple[SparseRouteBatch, ...]
    batch_cap: int
    padding_rows: int

    @property
    def diagonal_offsets(self) -> tuple[int, ...]:
        return tuple(
            offset
            for group in self.groups
            for offset in group.diagonal_offsets
        )

    @property
    def route_rows(self) -> int:
        return sum(group.row_count for group in self.groups)

    @property
    def encoded_rows(self) -> int:
        return self.route_rows + int(self.padding_rows)

    @property
    def giant_rotations(self) -> int:
        return sum(int(group.giant_offset) != 0 for group in self.groups)

    @property
    def baby_rotations(self) -> int:
        return len(self.baby_offsets) - int(0 in self.baby_offsets)


def canonical_rotation(offset: int, slots: int) -> int:
    value = int(offset) % int(slots)
    if value >= int(slots) // 2:
        value -= int(slots)
    return value


def build_sparse_route_schedule(
    *,
    diagonal_offsets: tuple[int, ...] | list[int],
    baby_offsets: tuple[int, ...] | list[int],
    modulus: int,
    slots: int,
    batch_cap: int = 32,
    padded_plaintext_rows: int | None = None,
) -> SparseRouteSchedule:
    """Group sparse routes and greedily pack complete groups into GPU batches.

    Giant groups are deliberately never split.  That preserves exactly one
    post-accumulation giant rotation for each nonzero group.  The selected
    attention plans have at most 15 routes in one group, so the production
    cap of 32 can usually encode two V groups or four K groups together.
    """

    slots = int(slots)
    modulus = int(modulus)
    batch_cap = int(batch_cap)
    if slots <= 0:
        raise ValueError("slots must be positive")
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    if not 1 <= batch_cap <= 32:
        raise ValueError("batch_cap must be in [1, 32]")

    diagonals = tuple(int(offset) for offset in diagonal_offsets)
    babies = tuple(int(offset) for offset in baby_offsets)
    if not diagonals:
        raise ValueError("diagonal_offsets must not be empty")
    if len(set(diagonals)) != len(diagonals):
        raise ValueError("diagonal_offsets must be unique")
    if not babies or len(set(babies)) != len(babies):
        raise ValueError("baby_offsets must be non-empty and unique")

    by_giant: dict[int, list[tuple[int, int]]] = {}
    for offset in diagonals:
        baby = int(offset) % modulus
        if baby not in babies:
            raise ValueError(
                f"diagonal {offset} requires baby offset {baby}, which is absent"
            )
        giant = canonical_rotation(int(offset) - baby, slots)
        by_giant.setdefault(giant, []).append((offset, baby))

    groups = tuple(
        SparseRouteGroup(
            giant_offset=int(giant),
            diagonal_offsets=tuple(int(offset) for offset, _ in entries),
            baby_offsets=tuple(int(baby) for _, baby in entries),
        )
        for giant, entries in sorted(by_giant.items())
    )
    largest_group = max(group.row_count for group in groups)
    if largest_group > batch_cap:
        raise ValueError(
            f"batch_cap={batch_cap} cannot hold the largest complete giant "
            f"group ({largest_group} rows)"
        )

    route_rows = len(diagonals)
    padded_rows = (
        route_rows
        if padded_plaintext_rows is None
        else int(padded_plaintext_rows)
    )
    if padded_rows < route_rows:
        raise ValueError(
            f"padded_plaintext_rows={padded_rows} is below the "
            f"{route_rows} nonzero routes"
        )
    padding_left = padded_rows - route_rows

    # Pack complete groups first: a group is never split, which is what keeps
    # exactly one post-accumulation giant rotation per nonzero group.
    packed: list[tuple[SparseRouteGroup, ...]] = []
    current: list[SparseRouteGroup] = []
    current_rows = 0
    for group in groups:
        if current and current_rows + group.row_count > batch_cap:
            packed.append(tuple(current))
            current = []
            current_rows = 0
        current.append(group)
        current_rows += group.row_count
    if current:
        packed.append(tuple(current))

    # Then place the padding rows: spare capacity in the packed batches first,
    # from the last backwards, and only then zero-only batches. Padding still
    # produces the audited number of zero PT-CT uses.
    padding_by_batch = [0] * len(packed)
    for index in range(len(packed) - 1, -1, -1):
        if padding_left <= 0:
            break
        used = sum(group.row_count for group in packed[index])
        take = min(batch_cap - used, padding_left)
        padding_by_batch[index] = int(take)
        padding_left -= int(take)
    zero_only: list[int] = []
    while padding_left > 0:
        take = min(batch_cap, padding_left)
        zero_only.append(int(take))
        padding_left -= int(take)

    batches = tuple(
        [
            SparseRouteBatch(groups=group_tuple, padding_rows=padding)
            for group_tuple, padding in zip(
                packed, padding_by_batch, strict=True
            )
        ]
        + [
            SparseRouteBatch(groups=(), padding_rows=padding)
            for padding in zero_only
        ]
    )
    if any(batch.encoded_rows > batch_cap for batch in batches):
        raise RuntimeError("constructed sparse route batch exceeds its cap")

    return SparseRouteSchedule(
        modulus=modulus,
        baby_offsets=babies,
        groups=groups,
        batches=batches,
        batch_cap=batch_cap,
        padding_rows=padded_rows - route_rows,
    )


def materialize_sparse_batch_masks(
    batch: SparseRouteBatch,
    *,
    routes: dict[int, SparseRouteValue],
    slots: int,
) -> tuple[np.ndarray, tuple[tuple[SparseRouteGroup, int, int], ...]]:
    """Build one bounded dense-mask batch from sparse destination indices."""

    slots = int(slots)
    route_values = [
        sparse_route_arrays(routes[int(offset)])
        for group in batch.groups
        for offset in group.diagonal_offsets
    ]
    dtype = np.result_type(
        np.float64,
        *(coefficients.dtype for _, coefficients in route_values),
    )
    rows = np.zeros((batch.encoded_rows, slots), dtype=dtype)
    metadata: list[tuple[SparseRouteGroup, int, int]] = []
    cursor = 0
    for group in batch.groups:
        start = cursor
        for offset in group.diagonal_offsets:
            if int(offset) not in routes:
                raise KeyError(f"route destinations for diagonal {offset} are missing")
            # Equivalent to writing the destination mask and subsequently
            # applying np.roll(mask, giant_offset), without copying a dense
            # slot row a second time.
            destinations, coefficients = sparse_route_arrays(
                routes[int(offset)]
            )
            destinations = (
                destinations + int(group.giant_offset)
            ) % slots
            rows[cursor, destinations] = coefficients
            cursor += 1
        metadata.append((group, start, cursor))
    if cursor + int(batch.padding_rows) != batch.encoded_rows:
        raise RuntimeError("sparse batch mask cursor does not match encoded rows")
    return rows, tuple(metadata)


__all__ = [
    "SparseMaskRoute",
    "SparseRouteBatch",
    "SparseRouteGroup",
    "SparseRouteSchedule",
    "build_sparse_route_schedule",
    "canonical_rotation",
    "materialize_sparse_batch_masks",
    "sparse_route_arrays",
]
