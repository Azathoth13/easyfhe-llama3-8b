from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttentionQKVPayload:
    query_c: np.ndarray
    key_delta: np.ndarray
    value_delta: np.ndarray


@dataclass(frozen=True)
class AttentionPairLayout:
    """Two-head interleaved Attention ``C/Delta`` working layouts."""

    seq_len: int = 128
    head_dim: int = 128
    query_heads: int = 32
    key_value_heads: int = 8
    slots: int = 32768

    def __post_init__(self) -> None:
        for name in ("seq_len", "head_dim", "query_heads", "key_value_heads", "slots"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if int(self.query_heads) % 2 or int(self.key_value_heads) % 2:
            raise ValueError("two-head interleaving requires even Q and KV head counts.")
        if int(self.query_heads) % int(self.key_value_heads):
            raise ValueError("query_heads must be divisible by key_value_heads for GQA.")
        if int(self.seq_len) != int(self.head_dim):
            raise ValueError("the current C/Delta design requires seq_len == head_dim.")
        if 2 * int(self.seq_len) * int(self.head_dim) > int(self.slots):
            raise ValueError(
                "one two-head payload does not fit: "
                f"2*{self.seq_len}*{self.head_dim} > slots={self.slots}."
            )
        if int(self.slots) % self.payload_slots:
            raise ValueError(
                f"slots={self.slots} must be divisible by the two-head payload size={self.payload_slots}."
            )

    @property
    def payload_slots(self) -> int:
        return 2 * int(self.seq_len) * int(self.head_dim)

    @property
    def payload_repetitions(self) -> int:
        return int(self.slots) // self.payload_slots

    @property
    def gqa_ratio(self) -> int:
        return int(self.query_heads) // int(self.key_value_heads)

    @property
    def query_pair_count(self) -> int:
        return int(self.query_heads) // 2

    @property
    def key_value_pair_count(self) -> int:
        return int(self.key_value_heads) // 2


    def query_head_pairs(self) -> tuple[tuple[int, int], ...]:
        ratio = self.gqa_ratio
        pairs: list[tuple[int, int]] = []
        for kv_pair in range(self.key_value_pair_count):
            q_base = 2 * ratio * kv_pair
            for offset in range(ratio):
                pairs.append((q_base + offset, q_base + ratio + offset))
        return tuple(pairs)

    def key_value_head_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((2 * pair, 2 * pair + 1) for pair in range(self.key_value_pair_count))

    def _logical_heads(self, values: np.ndarray, *, heads: int, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        flat_shape = (int(self.seq_len), int(heads) * int(self.head_dim))
        head_shape = (int(self.seq_len), int(heads), int(self.head_dim))
        if values.shape == flat_shape:
            return values.reshape(head_shape)
        if values.shape == head_shape:
            return values
        raise ValueError(f"{name} must have shape {flat_shape} or {head_shape}, got {values.shape}.")

    def _repeat_payload(self, base: np.ndarray) -> np.ndarray:
        base = np.asarray(base)
        if base.ndim != 2 or base.shape[1] != self.payload_slots:
            raise ValueError(f"base payload must have shape [pairs, {self.payload_slots}], got {base.shape}.")
        return np.tile(base, (1, self.payload_repetitions))

    def _base_payload(self, packed: np.ndarray, *, pairs: int, name: str) -> np.ndarray:
        packed = np.asarray(packed)
        expected = (int(pairs), int(self.slots))
        if packed.shape != expected:
            raise ValueError(f"{name} payload must have shape {expected}, got {packed.shape}.")
        base = packed[:, : self.payload_slots]
        for repetition in range(1, self.payload_repetitions):
            start = repetition * self.payload_slots
            if not np.allclose(
                base,
                packed[:, start : start + self.payload_slots],
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError(f"{name} repeated payload copy {repetition} does not match copy 0.")
        return base

    def pack_query_c(self, query: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        query = self._logical_heads(query, heads=int(self.query_heads), name="query")
        packed = np.zeros((self.query_pair_count, self.payload_slots), dtype=dtype)
        tokens = np.arange(int(self.seq_len), dtype=np.int64)
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                slots = 2 * (
                    np.arange(int(self.head_dim), dtype=np.int64)[:, None] * int(self.seq_len) + tokens[None, :]
                ) + lane
                packed[pair_index, slots.reshape(-1)] = query[:, head, :].T.reshape(-1)
        return self._repeat_payload(packed)

    def unpack_query_c(self, packed: np.ndarray) -> np.ndarray:
        packed = self._base_payload(packed, pairs=self.query_pair_count, name="Q_C")
        query = np.zeros((int(self.seq_len), int(self.query_heads), int(self.head_dim)), dtype=np.float32)
        tokens = np.arange(int(self.seq_len), dtype=np.int64)
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                slots = 2 * (
                    np.arange(int(self.head_dim), dtype=np.int64)[:, None] * int(self.seq_len) + tokens[None, :]
                ) + lane
                query[:, head, :] = packed[pair_index, slots.reshape(-1)].reshape(
                    int(self.head_dim), int(self.seq_len)
                ).T
        return query.reshape(int(self.seq_len), -1)

    def pack_query_c_feature_folded(
        self, query: np.ndarray, *, scale: float = 0.5, dtype=np.complex128
    ) -> np.ndarray:
        """Pack the selected complex QK operand in full ``C`` layout.

        Every feature row ``d`` contains

        ``scale * (Q[d] + i*Q[(d+h/2) mod h])``.

        Keeping both feature halves is intentional: the Delta QK reduction
        uses every physical first-axis row as a distinct output delta row.
        """

        if int(self.head_dim) % 2:
            raise ValueError("complex Q feature folding requires even head_dim.")
        values = self._logical_heads(
            query, heads=int(self.query_heads), name="query"
        )
        packed = np.zeros(
            (self.query_pair_count, self.payload_slots), dtype=dtype
        )
        tokens = np.arange(int(self.seq_len), dtype=np.int64)
        features = np.arange(int(self.head_dim), dtype=np.int64)
        paired_features = (features + int(self.head_dim) // 2) % int(
            self.head_dim
        )
        slots_base = 2 * (
            features[:, None] * int(self.seq_len) + tokens[None, :]
        )
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                real = values[:, head, :].T
                imaginary = values[:, head, paired_features].T
                packed[pair_index, (slots_base + lane).reshape(-1)] = (
                    float(scale) * (real + 1j * imaginary)
                ).reshape(-1)
        return self._repeat_payload(packed)

    def _pack_delta(self, values: np.ndarray, *, name: str, dtype=np.float64) -> np.ndarray:
        values = self._logical_heads(values, heads=int(self.key_value_heads), name=name)
        packed = np.zeros((self.key_value_pair_count, self.payload_slots), dtype=dtype)
        queries = np.arange(int(self.seq_len), dtype=np.int64)
        features = np.arange(int(self.head_dim), dtype=np.int64)
        source_tokens = (queries[None, :] + features[:, None]) % int(self.seq_len)
        slots_base = 2 * (features[:, None] * int(self.seq_len) + queries[None, :])
        for pair_index, heads in enumerate(self.key_value_head_pairs()):
            for lane, head in enumerate(heads):
                packed[pair_index, (slots_base + lane).reshape(-1)] = values[source_tokens, head, features[:, None]].reshape(-1)
        return self._repeat_payload(packed)


    def pack_key_delta(self, key: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        return self._pack_delta(key, name="key", dtype=dtype)

    def pack_key_delta_feature_folded(
        self, key: np.ndarray, *, dtype=np.complex128
    ) -> np.ndarray:
        """Pack ``K_Delta[d] - i*K_Delta[d+d_h/2]`` for folded QK."""

        if int(self.head_dim) % 2:
            raise ValueError("complex K feature folding requires even head_dim.")
        values = self._logical_heads(
            key, heads=int(self.key_value_heads), name="key"
        )
        packed = np.zeros(
            (self.key_value_pair_count, self.payload_slots), dtype=dtype
        )
        queries = np.arange(int(self.seq_len), dtype=np.int64)
        features = np.arange(int(self.head_dim), dtype=np.int64)
        paired_features = (features + int(self.head_dim) // 2) % int(
            self.head_dim
        )
        source_tokens = (queries[None, :] + features[:, None]) % int(
            self.seq_len
        )
        slots_base = 2 * (
            features[:, None] * int(self.seq_len) + queries[None, :]
        )
        for pair_index, heads in enumerate(self.key_value_head_pairs()):
            for lane, head in enumerate(heads):
                real = values[source_tokens, head, features[:, None]]
                imaginary = values[
                    source_tokens, head, paired_features[:, None]
                ]
                packed[pair_index, (slots_base + lane).reshape(-1)] = (
                    real - 1j * imaginary
                ).reshape(-1)
        return self._repeat_payload(packed)


    def pack_value_delta(self, value: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        return self._pack_delta(value, name="value", dtype=dtype)


    def pack_score_delta(self, scores: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        """Pack logical scores ``[Q heads, query, key]`` as ``S_Delta[d,q]``."""

        scores = np.asarray(scores, dtype=np.float32)
        expected = (int(self.query_heads), int(self.seq_len), int(self.seq_len))
        if scores.shape != expected:
            raise ValueError(f"scores must have shape {expected}, got {scores.shape}.")
        packed = np.zeros((self.query_pair_count, self.payload_slots), dtype=dtype)
        queries = np.arange(int(self.seq_len), dtype=np.int64)
        deltas = np.arange(int(self.seq_len), dtype=np.int64)
        keys = (queries[None, :] + deltas[:, None]) % int(self.seq_len)
        slots_base = 2 * (deltas[:, None] * int(self.seq_len) + queries[None, :])
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                packed[pair_index, (slots_base + lane).reshape(-1)] = scores[head, queries[None, :], keys].reshape(-1)
        return self._repeat_payload(packed)

    def unpack_score_delta(self, packed: np.ndarray) -> np.ndarray:
        packed = self._base_payload(packed, pairs=self.query_pair_count, name="S_Delta/P_Delta")
        scores = np.zeros(
            (int(self.query_heads), int(self.seq_len), int(self.seq_len)), dtype=np.float32
        )
        queries = np.arange(int(self.seq_len), dtype=np.int64)
        deltas = np.arange(int(self.seq_len), dtype=np.int64)
        keys = (queries[None, :] + deltas[:, None]) % int(self.seq_len)
        slots_base = 2 * (deltas[:, None] * int(self.seq_len) + queries[None, :])
        for pair_index, heads in enumerate(self.query_head_pairs()):
            for lane, head in enumerate(heads):
                scores[head, queries[None, :], keys] = packed[
                    pair_index, (slots_base + lane).reshape(-1)
                ].reshape(int(self.seq_len), int(self.seq_len))
        return scores

    def pack_output_c(self, output: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        return self.pack_query_c(output, dtype=dtype)

    def unpack_output_c(self, packed: np.ndarray) -> np.ndarray:
        return self.unpack_query_c(packed)

    def pack_qkv(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, *, dtype=np.float64) -> AttentionQKVPayload:
        return AttentionQKVPayload(
            query_c=self.pack_query_c(query, dtype=dtype),
            key_delta=self.pack_key_delta(key, dtype=dtype),
            value_delta=self.pack_value_delta(value, dtype=dtype),
        )
