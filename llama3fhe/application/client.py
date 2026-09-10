from __future__ import annotations

"""Client-side plaintext I/O shared by the executable applications.

Nothing in this module performs homomorphic evaluation or owns an EasyFHE
context.  It only translates between user/checkpoint data and the canonical
feature-major encrypted boundary.
"""

import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..layouts.feature_major import FeatureMajorPrefillLayout
from .checkpoint import CheckpointMetadata, load_norm_weight

if TYPE_CHECKING:
    from .checkpoint import WeightLoader


def load_input_ids(path: str | Path, *, seq_len: int) -> np.ndarray:
    """Load exactly ``seq_len`` token IDs from a JSON list or object."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("input_ids") if isinstance(payload, dict) else payload
    if values is None:
        raise ValueError("input-ID JSON object must contain an 'input_ids' field.")
    ids = np.asarray(values, dtype=np.int64).reshape(-1)
    if ids.size != int(seq_len):
        raise ValueError(
            f"input IDs must contain exactly {seq_len} entries, got {ids.size}."
        )
    return ids


def decrypt_hidden(
    ciphers,
    *,
    crypto_context,
    layout: FeatureMajorPrefillLayout,
) -> np.ndarray:
    """Decrypt canonical FM ciphertexts into logical ``[token, feature]`` rows."""

    ciphers = tuple(ciphers)
    if len(ciphers) != int(layout.cipher_count):
        raise ValueError(
            f"feature-major decode expects {layout.cipher_count} ciphers, "
            f"got {len(ciphers)}."
        )
    packed = np.stack(
        [np.asarray(crypto_context.decrypt(cipher)) for cipher in ciphers],
        axis=0,
    )
    return layout.unpack(packed)


def final_logits(
    hidden: np.ndarray,
    *,
    loader: "WeightLoader",
    metadata: CheckpointMetadata,
    token_index: int = -1,
) -> np.ndarray:
    """Apply the client-side exact final RMSNorm and language-model head."""

    hidden = np.asarray(hidden, dtype=np.float32)
    hidden_size = int(metadata.hidden_size)
    if hidden.ndim != 2 or hidden.shape[1] != hidden_size or hidden.shape[0] == 0:
        raise ValueError(
            "final hidden state must be a non-empty [token, hidden] matrix with "
            f"hidden={hidden_size}, got {hidden.shape}."
        )
    gamma = load_norm_weight(loader, "model.norm.weight", hidden_size)
    token_index = int(token_index)
    if token_index < 0:
        token_index += int(hidden.shape[0])
    if not 0 <= token_index < int(hidden.shape[0]):
        raise IndexError(
            f"token_index={token_index} is outside [0, {hidden.shape[0]})."
        )
    last = hidden[token_index]
    inverse_rms = 1.0 / np.sqrt(
        np.mean(last.astype(np.float64) ** 2) + float(metadata.rms_norm_eps)
    )
    normalized = np.asarray(last * inverse_rms, dtype=np.float32) * gamma
    head_name = (
        "lm_head.weight"
        if loader.has_tensor("lm_head.weight")
        else "model.embed_tokens.weight"
    )
    head = loader.get(head_name, out_dtype=np.float32)
    if head.ndim != 2 or head.shape[1] != hidden_size:
        raise ValueError(
            f"{head_name} must have shape [vocab, {metadata.hidden_size}], "
            f"got {head.shape}."
        )
    return np.asarray(head @ normalized, dtype=np.float32)


def top_k_logits(logits: np.ndarray, count: int) -> list[dict[str, object]]:
    """Return stable descending token/logit rows without sorting the full vocab."""

    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits.size == 0:
        raise ValueError("logits must not be empty.")
    count = int(count)
    if count <= 0:
        raise ValueError("top-k count must be positive.")
    count = min(count, int(logits.size))
    indices = np.argpartition(-logits, count - 1)[:count]
    indices = indices[np.argsort(-logits[indices], kind="stable")]
    return [
        {"token_id": int(index), "logit": float(logits[index])}
        for index in indices
    ]


def write_json_report(
    payload: dict[str, object], output_path: str | Path | None
) -> None:
    """Print a report and optionally persist the identical JSON payload."""

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def median(values) -> float:
    """Return a float median for timing/report values."""

    return float(statistics.median(float(value) for value in values))


def stage_medians(samples: list[dict[str, object]]) -> dict[str, float]:
    """Aggregate the union of named operator stages across measured runs."""

    names = sorted(
        {
            str(name)
            for sample in samples
            for name in dict(sample["stage_seconds"]).keys()
        }
    )
    return {
        name: median(
            dict(sample["stage_seconds"]).get(name, 0.0)
            for sample in samples
        )
        for name in names
    }


def nested_timing_sum(value, *, suffixes: tuple[str, ...]) -> float:
    """Sum numeric timing leaves whose names end with ``suffixes``."""

    if isinstance(value, (list, tuple)):
        return sum(
            nested_timing_sum(child, suffixes=suffixes) for child in value
        )
    if not isinstance(value, dict):
        return 0.0
    total = 0.0
    for key, child in value.items():
        if isinstance(child, (int, float)) and str(key).endswith(suffixes):
            total += float(child)
        elif isinstance(child, (dict, list, tuple)):
            total += nested_timing_sum(child, suffixes=suffixes)
    return total


def begin_cuda_allocator_sample(device: str) -> int | None:
    """Reset CUDA peak accounting and return the current allocator usage."""

    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
        return int(torch.cuda.memory_allocated())
    except (ImportError, RuntimeError):
        return None


def end_cuda_allocator_sample(device: str) -> int | None:
    """Return the EasyFHE/PyTorch allocator peak for the active sample."""

    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        return None


__all__ = [
    "decrypt_hidden",
    "begin_cuda_allocator_sample",
    "end_cuda_allocator_sample",
    "final_logits",
    "load_input_ids",
    "median",
    "nested_timing_sum",
    "stage_medians",
    "top_k_logits",
    "write_json_report",
]
