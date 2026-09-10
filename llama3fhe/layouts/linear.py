from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _positive_power_of_two(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}.")
    return value


@dataclass(frozen=True)
class LinearCarrierLayout:
    """Private feature-major carrier used between QKV and attention.

    Slots use ``slot(dim, token_lane)``.  Unlike
    :class:`~llama3fhe.layouts.feature_major.FeatureMajorPrefillLayout`, the
    ``dimension`` is the power-of-two physical row width (4096 in production).
    Operators may use only an active prefix, currently 3072 heterogeneous QKV
    channels.  This is not a persistent model layout and never crosses a
    residual boundary.
    """

    seq_len: int
    dimension: int
    slots: int = 32768

    def __post_init__(self) -> None:
        _positive_power_of_two(self.dimension, name="dimension")
        if int(self.seq_len) <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}.")
        if int(self.slots) % int(self.dimension):
            raise ValueError(
                "linear carrier layout requires slots divisible by dimension, "
                f"got {self.slots}/{self.dimension}."
            )

    @property
    def token_lanes(self) -> int:
        return int(self.slots) // int(self.dimension)

    @property
    def cipher_count(self) -> int:
        return (
            int(self.seq_len) + self.token_lanes - 1
        ) // self.token_lanes

    def slot(self, dim: int, token_lane: int) -> int:
        dim = int(dim)
        token_lane = int(token_lane)
        if not 0 <= dim < int(self.dimension):
            raise IndexError(f"dim={dim} is outside [0, {self.dimension}).")
        if not 0 <= token_lane < self.token_lanes:
            raise IndexError(
                f"token_lane={token_lane} is outside [0, {self.token_lanes})."
            )
        return dim * self.token_lanes + token_lane

    def pack(self, rows: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        rows = np.asarray(rows, dtype=dtype)
        expected = (int(self.seq_len), int(self.dimension))
        if rows.shape != expected:
            raise ValueError(
                f"linear carrier rows must have shape {expected}, got {rows.shape}."
            )
        padded_rows = self.cipher_count * self.token_lanes
        padded = np.zeros((padded_rows, int(self.dimension)), dtype=dtype)
        padded[: int(self.seq_len)] = rows
        return (
            padded.reshape(
                self.cipher_count,
                self.token_lanes,
                int(self.dimension),
            )
            .transpose(0, 2, 1)
            .reshape(self.cipher_count, int(self.slots))
        )

    def unpack(
        self,
        packed: np.ndarray,
        *,
        output_dim: int | None = None,
    ) -> np.ndarray:
        packed = np.asarray(packed)
        expected = (self.cipher_count, int(self.slots))
        if packed.shape != expected:
            raise ValueError(
                f"linear carrier payload must have shape {expected}, got {packed.shape}."
            )
        output_dim = int(
            self.dimension if output_dim is None else output_dim
        )
        if not 0 < output_dim <= int(self.dimension):
            raise ValueError(
                f"output_dim must be in [1, {self.dimension}], got {output_dim}."
            )
        rows = (
            packed.reshape(
                self.cipher_count,
                int(self.dimension),
                self.token_lanes,
            )
            .transpose(0, 2, 1)
            .reshape(-1, int(self.dimension))
        )
        return np.asarray(
            rows[: int(self.seq_len), :output_dim],
            dtype=np.float32,
        )


__all__ = ["LinearCarrierLayout"]
