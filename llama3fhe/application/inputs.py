"""Turning a prompt or a fixture into the model's plaintext input.

Everything the runner needs before encryption: resolving token ids from a
prompt or a JSON fixture, the padding token, and the synthetic hidden states
used when no checkpoint is available. Nothing here is encrypted yet — the
boundary is deliberate, because these are the only places a real prompt exists
in the clear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..layouts import AttentionPairLayout, FeatureMajorPrefillLayout
from ..model import LayerWeights
from .checkpoint import CheckpointMetadata


def _padding_token(metadata: CheckpointMetadata, tokenizer) -> int:
    for value in (
        getattr(tokenizer, "pad_token_id", None),
        metadata.pad_token_id,
        getattr(tokenizer, "eos_token_id", None),
    ):
        if value is not None:
            return int(value)
    eos = metadata.eos_token_id
    if isinstance(eos, tuple) and eos:
        return int(eos[0])
    if isinstance(eos, int):
        return eos
    raise ValueError("checkpoint has no EOS/PAD token for 128-token padding.")


def _load_prompt_ids(
    prompt: str,
    *,
    model_dir: Path,
    metadata: CheckpointMetadata,
    seq_len: int,
    pad_side: str,
    add_special_tokens: bool,
) -> tuple[np.ndarray, object, int]:
    """Tokenize and deterministically truncate/pad one prompt to prefill size."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "--prompt requires transformers; use --input-ids-file otherwise."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ids = np.asarray(
        tokenizer.encode(
            str(prompt), add_special_tokens=bool(add_special_tokens)
        ),
        dtype=np.int64,
    )
    raw_count = int(ids.size)
    if raw_count == 0:
        raise ValueError("prompt tokenization produced no tokens.")
    if ids.size > int(seq_len):
        ids = ids[-int(seq_len) :]
    elif ids.size < int(seq_len):
        padding = np.full(
            (int(seq_len) - ids.size,),
            _padding_token(metadata, tokenizer),
            dtype=np.int64,
        )
        ids = (
            np.concatenate((padding, ids))
            if str(pad_side) == "left"
            else np.concatenate((ids, padding))
        )
    return ids, tokenizer, raw_count


def _synthetic_inputs(
    seed: int,
    *,
    layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> tuple[np.ndarray, LayerWeights]:
    """Create deterministic full-shape data for the one-layer system preset."""

    rng = np.random.default_rng(int(seed))
    seq_len, hidden_dim = int(layout.seq_len), int(layout.hidden_dim)
    hidden = rng.normal(
        scale=0.01, size=(seq_len, hidden_dim)
    ).astype(np.float32)

    def weight(shape, scale=0.002):
        return rng.normal(scale=scale, size=shape).astype(np.float32)

    kv_dim = int(attention_layout.key_value_heads) * int(
        attention_layout.head_dim
    )
    return hidden, LayerWeights(
        input_layernorm=rng.normal(
            1.0, 0.01, size=(hidden_dim,)
        ).astype(np.float32),
        q_proj=weight((hidden_dim, hidden_dim)),
        k_proj=weight((kv_dim, hidden_dim)),
        v_proj=weight((kv_dim, hidden_dim)),
        o_proj=weight((hidden_dim, hidden_dim)),
        post_attention_layernorm=rng.normal(
            1.0, 0.01, size=(hidden_dim,)
        ).astype(np.float32),
        gate_proj=weight((14336, hidden_dim)),
        up_proj=weight((14336, hidden_dim)),
        down_proj=weight((hidden_dim, 14336)),
    )