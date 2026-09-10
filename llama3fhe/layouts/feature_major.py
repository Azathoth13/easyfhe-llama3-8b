from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linear import _positive_power_of_two


@dataclass(frozen=True)
class FeatureMajorPrefillLayout:
    """Canonical real feature-major prefill layout.

    A ciphertext owns a contiguous shard of tokens.  Within that ciphertext,
    the token shard is the fast axis and model feature is the slow axis::

        slot(dim, local_token) = dim * tokens_per_cipher + local_token

    The production contract is ``S=128, D=4096, slots=32768``: eight tokens
    per ciphertext and sixteen real ciphertexts.  Smaller power-of-two shapes
    are supported so the same contract can be exercised in smoke tests.
    """

    seq_len: int = 128
    hidden_dim: int = 4096
    slots: int = 32768

    def __post_init__(self) -> None:
        _positive_power_of_two(self.hidden_dim, name="hidden_dim")
        if int(self.seq_len) <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}.")
        if int(self.slots) % int(self.hidden_dim):
            raise ValueError(
                "feature-major layout requires slots divisible by hidden_dim, "
                f"got slots={self.slots}, hidden_dim={self.hidden_dim}."
            )

    @property
    def tokens_per_cipher(self) -> int:
        return int(self.slots) // int(self.hidden_dim)

    @property
    def cipher_count(self) -> int:
        return (
            int(self.seq_len) + self.tokens_per_cipher - 1
        ) // self.tokens_per_cipher

    def active_tokens(self, cipher_index: int) -> int:
        cipher_index = int(cipher_index)
        if not 0 <= cipher_index < self.cipher_count:
            raise IndexError(
                f"cipher_index={cipher_index} is outside [0, {self.cipher_count})."
            )
        return max(
            0,
            min(
                self.tokens_per_cipher,
                int(self.seq_len) - cipher_index * self.tokens_per_cipher,
            ),
        )

    def slot(self, dim: int, local_token: int) -> int:
        dim = int(dim)
        local_token = int(local_token)
        if not 0 <= dim < int(self.hidden_dim):
            raise IndexError(f"dim={dim} is outside [0, {self.hidden_dim}).")
        if not 0 <= local_token < self.tokens_per_cipher:
            raise IndexError(
                "local_token="
                f"{local_token} is outside [0, {self.tokens_per_cipher})."
            )
        return dim * self.tokens_per_cipher + local_token

    def pack(self, rows: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        rows = np.asarray(rows, dtype=dtype)
        expected = (int(self.seq_len), int(self.hidden_dim))
        if rows.shape != expected:
            raise ValueError(
                f"feature-major rows must have shape {expected}, got {rows.shape}."
            )
        padded_rows = self.cipher_count * self.tokens_per_cipher
        padded = np.zeros((padded_rows, int(self.hidden_dim)), dtype=dtype)
        padded[: int(self.seq_len)] = rows
        return (
            padded.reshape(
                self.cipher_count,
                self.tokens_per_cipher,
                int(self.hidden_dim),
            )
            .transpose(0, 2, 1)
            .reshape(self.cipher_count, int(self.slots))
        )

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        packed = np.asarray(packed)
        expected = (self.cipher_count, int(self.slots))
        if packed.shape != expected:
            raise ValueError(
                f"feature-major payload must have shape {expected}, got {packed.shape}."
            )
        rows = (
            packed.reshape(
                self.cipher_count,
                int(self.hidden_dim),
                self.tokens_per_cipher,
            )
            .transpose(0, 2, 1)
            .reshape(-1, int(self.hidden_dim))
        )
        return np.asarray(rows[: int(self.seq_len)], dtype=np.float32)
