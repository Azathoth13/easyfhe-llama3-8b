from __future__ import annotations

"""Logical-to-physical maps for online diagonal weight packing.

The feature-major linear kernels consume a square physical matrix, but model
weights rarely arrive in that form.  This descriptor represents each physical
entry as a (possibly complex) pair of reads from one or more real model-weight
matrices::

    W_phys[o, i] = column_scale[i] * (
        real_scale[o] * sources[real_source[o]][real_row[o], column[i]]
        + 1j * imag_scale[o]
          * sources[imag_source[o]][imag_row[o], column[i]]
    )

``source == -1`` or ``column == -1`` denotes public zero padding.  Keeping the
map explicit lets the Triton packer perform permutation, slicing, scaling and
real/imaginary fusion while it emits BSGS diagonals, without first allocating
a dense square physical matrix on the CPU.
"""

from dataclasses import dataclass

import numpy as np


def _index_vector(value, *, dimension: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64).reshape(-1)
    if result.shape != (int(dimension),):
        raise ValueError(
            f"{name} must have shape {(int(dimension),)}, got {result.shape}."
        )
    return np.ascontiguousarray(result)


def _scale_vector(value, *, dimension: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (int(dimension),):
        raise ValueError(
            f"{name} must have shape {(int(dimension),)}, got {result.shape}."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class StructuredLinearWeight:
    """A square physical linear map backed by real logical weight sources."""

    dimension: int
    sources: tuple[np.ndarray, ...]
    real_source: np.ndarray
    real_row: np.ndarray
    real_scale: np.ndarray
    imag_source: np.ndarray
    imag_row: np.ndarray
    imag_scale: np.ndarray
    column: np.ndarray
    column_scale: np.ndarray

    def __post_init__(self) -> None:
        dimension = int(self.dimension)
        if dimension <= 0:
            raise ValueError("structured linear dimension must be positive.")
        sources = tuple(np.asarray(source) for source in self.sources)
        if not 1 <= len(sources) <= 3:
            raise ValueError("structured linear weight requires one to three sources.")
        for index, source in enumerate(sources):
            if source.ndim != 2:
                raise ValueError(
                    f"source {index} must be a matrix, got shape {source.shape}."
                )
            if np.iscomplexobj(source) or not np.issubdtype(
                source.dtype, np.floating
            ):
                raise TypeError(
                    f"source {index} must be a real floating matrix, got {source.dtype}."
                )
        source_dtypes = {np.dtype(source.dtype) for source in sources}
        if len(source_dtypes) != 1:
            raise TypeError(
                "structured linear sources must use one common dtype, got "
                f"{sorted(str(value) for value in source_dtypes)}."
            )

        real_source = _index_vector(
            self.real_source, dimension=dimension, name="real_source"
        )
        real_row = _index_vector(
            self.real_row, dimension=dimension, name="real_row"
        )
        imag_source = _index_vector(
            self.imag_source, dimension=dimension, name="imag_source"
        )
        imag_row = _index_vector(
            self.imag_row, dimension=dimension, name="imag_row"
        )
        column = _index_vector(self.column, dimension=dimension, name="column")
        real_scale = _scale_vector(
            self.real_scale, dimension=dimension, name="real_scale"
        )
        imag_scale = _scale_vector(
            self.imag_scale, dimension=dimension, name="imag_scale"
        )
        column_scale = _scale_vector(
            self.column_scale, dimension=dimension, name="column_scale"
        )

        source_count = len(sources)
        for name, source_ids, rows in (
            ("real", real_source, real_row),
            ("imag", imag_source, imag_row),
        ):
            if np.any((source_ids < -1) | (source_ids >= source_count)):
                raise ValueError(
                    f"{name}_source values must be -1 or in [0, {source_count})."
                )
            if np.any((source_ids >= 0) & (rows < 0)):
                raise ValueError(f"active {name}_row values must be non-negative.")
            for source_index, source in enumerate(sources):
                selected = rows[source_ids == source_index]
                if selected.size and int(np.max(selected)) >= int(source.shape[0]):
                    raise ValueError(
                        f"{name}_row exceeds source {source_index} row count "
                        f"{source.shape[0]}."
                    )

        if np.any(column < -1):
            raise ValueError("column values must be -1 or non-negative.")
        for source_index, source in enumerate(sources):
            source_used = np.any(real_source == source_index) or np.any(
                imag_source == source_index
            )
            active_columns = column[column >= 0]
            if (
                source_used
                and active_columns.size
                and int(np.max(active_columns)) >= int(source.shape[1])
            ):
                raise ValueError(
                    f"column exceeds source {source_index} width {source.shape[1]}."
                )

        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "real_source", real_source)
        object.__setattr__(self, "real_row", real_row)
        object.__setattr__(self, "real_scale", real_scale)
        object.__setattr__(self, "imag_source", imag_source)
        object.__setattr__(self, "imag_row", imag_row)
        object.__setattr__(self, "imag_scale", imag_scale)
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "column_scale", column_scale)


    @property
    def is_complex(self) -> bool:
        return bool(np.any(self.imag_source >= 0))

    def materialize(self, *, dtype=None) -> np.ndarray:
        """Materialize the physical matrix as a CPU oracle/fallback."""

        if dtype is None:
            dtype = np.complex128 if self.is_complex else np.float64
        dtype = np.dtype(dtype)
        if self.is_complex and not np.issubdtype(dtype, np.complexfloating):
            raise TypeError("a structured weight with imaginary rows needs complex dtype.")
        output = np.zeros(
            (int(self.dimension), int(self.dimension)), dtype=dtype
        )
        active_columns = self.column >= 0
        columns = self.column[active_columns]
        column_scale = self.column_scale[active_columns]
        for source_index, source in enumerate(self.sources):
            rows = np.flatnonzero(self.real_source == source_index)
            if rows.size and columns.size:
                values = source[np.ix_(self.real_row[rows], columns)]
                output[np.ix_(rows, np.flatnonzero(active_columns))] += (
                    self.real_scale[rows, None]
                    * values
                    * column_scale[None, :]
                )
            rows = np.flatnonzero(self.imag_source == source_index)
            if rows.size and columns.size:
                values = source[np.ix_(self.imag_row[rows], columns)]
                output[np.ix_(rows, np.flatnonzero(active_columns))] += (
                    1j
                    * self.imag_scale[rows, None]
                    * values
                    * column_scale[None, :]
                )
        return output


def structured_linear_weight(
    sources,
    *,
    dimension: int,
    real_source,
    real_row,
    real_scale=1.0,
    imag_source=None,
    imag_row=None,
    imag_scale=0.0,
    column=None,
    column_scale=1.0,
) -> StructuredLinearWeight:
    """Build a descriptor, broadcasting scalar scales and zero padding."""

    dimension = int(dimension)

    def index_or(value, fill: int) -> np.ndarray:
        if value is None:
            return np.full((dimension,), int(fill), dtype=np.int64)
        return np.asarray(value, dtype=np.int64)

    def scale_or_vector(value) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            return np.full((dimension,), float(array), dtype=np.float64)
        return array

    return StructuredLinearWeight(
        dimension=dimension,
        sources=tuple(np.asarray(source) for source in sources),
        real_source=index_or(real_source, -1),
        real_row=index_or(real_row, -1),
        real_scale=scale_or_vector(real_scale),
        imag_source=index_or(imag_source, -1),
        imag_row=index_or(imag_row, -1),
        imag_scale=scale_or_vector(imag_scale),
        column=(
            np.arange(dimension, dtype=np.int64)
            if column is None
            else np.asarray(column, dtype=np.int64)
        ),
        column_scale=scale_or_vector(column_scale),
    )
