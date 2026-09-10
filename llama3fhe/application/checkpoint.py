from __future__ import annotations

"""Prepared-checkpoint metadata and lazy per-layer weight access."""

import json
import math
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..model import LayerWeights

RUNTIME_EXPORT_METADATA = "export_metadata.json"
RUNTIME_EXPORT_FORMAT = "llama3_8b_feature_major_quarot_v1"


_SIMPLE_DTYPES: dict[str, np.dtype[Any]] = {
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "I8": np.dtype("i1"),
    "I16": np.dtype("<i2"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
    "U8": np.dtype("u1"),
    "U16": np.dtype("<u2"),
    "U32": np.dtype("<u4"),
    "U64": np.dtype("<u8"),
    "BOOL": np.dtype("?"),
}


@dataclass(frozen=True)
class CheckpointMetadata:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    eos_token_id: int | tuple[int, ...] | None
    pad_token_id: int | None

    @classmethod
    def from_json(cls, path: str | Path) -> "CheckpointMetadata":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint config must be a JSON object: {path}.")
        required = (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                f"checkpoint config {path} is missing fields: {', '.join(missing)}."
            )
        raw_eos = payload.get("eos_token_id")
        if isinstance(raw_eos, list) and not raw_eos:
            raw_eos = None
        eos = (
            tuple(int(value) for value in raw_eos)
            if isinstance(raw_eos, list)
            else None if raw_eos is None else int(raw_eos)
        )
        return cls(
            hidden_size=int(payload["hidden_size"]),
            intermediate_size=int(payload["intermediate_size"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            num_key_value_heads=int(payload["num_key_value_heads"]),
            num_hidden_layers=int(payload.get("num_hidden_layers", 32)),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-5)),
            rope_theta=float(payload.get("rope_theta", 500000.0)),
            eos_token_id=eos,
            pad_token_id=(
                None
                if payload.get("pad_token_id") is None
                else int(payload["pad_token_id"])
            ),
        )

    def validate_llama3_8b(self, *, num_layers: int | None = None) -> None:
        expected = (4096, 14336, 32, 8)
        actual = (
            int(self.hidden_size),
            int(self.intermediate_size),
            int(self.num_attention_heads),
            int(self.num_key_value_heads),
        )
        if actual != expected:
            raise ValueError(
                f"release expects Llama-3-8B geometry {expected}, got {actual}."
            )
        requested = int(self.num_hidden_layers if num_layers is None else num_layers)
        if not 0 < requested <= int(self.num_hidden_layers):
            raise ValueError(
                f"requested {requested} layers from a "
                f"{self.num_hidden_layers}-layer checkpoint."
            )
        if not math.isfinite(float(self.rms_norm_eps)) or float(
            self.rms_norm_eps
        ) <= 0:
            raise ValueError(
                f"rms_norm_eps must be finite and positive, got {self.rms_norm_eps}."
            )
        if not math.isfinite(float(self.rope_theta)) or float(self.rope_theta) <= 0:
            raise ValueError(
                f"rope_theta must be finite and positive, got {self.rope_theta}."
            )


def validate_runtime_checkpoint(model_dir: str | Path) -> dict[str, object]:
    """Require the full-fused QuaRot export consumed by the FHE graph."""

    model_dir = Path(model_dir)
    path = model_dir / RUNTIME_EXPORT_METADATA
    if not path.is_file():
        raise ValueError(
            f"{model_dir} is not a prepared feature-major checkpoint: "
            f"missing {RUNTIME_EXPORT_METADATA}. Prepare a full-fused QuaRot "
            "checkpoint as described in assets/weights/README.md."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"runtime export metadata must be an object: {path}.")
    if payload.get("format") != RUNTIME_EXPORT_FORMAT:
        raise ValueError(
            f"unsupported runtime checkpoint format {payload.get('format')!r}; "
            f"expected {RUNTIME_EXPORT_FORMAT!r}."
        )
    if payload.get("full_fuse") is not True:
        raise ValueError(
            "runtime checkpoint must set full_fuse=true; online QuaRot "
            "Hadamard operators are intentionally absent from the FHE graph."
        )
    if payload.get("runtime_weight_layout") != "ordinary_out_in":
        raise ValueError("runtime weights must retain ordinary [out,in] layout.")
    if payload.get("he_packing") != "prepared_online_by_linear_operator":
        raise ValueError(
            "runtime checkpoint must delegate HE packing to the online linear operator."
        )
    return payload


class SafeTensorReader:
    """Validated, memory-mapped access to one safetensors file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None
        self._mmap = None
        self.header: dict[str, Any] = {}
        self.data_start = 0
        try:
            self._file = self.path.open("rb")
            if self.path.stat().st_size < 8:
                raise ValueError(
                    f"safetensors file is shorter than its header: {self.path}."
                )
            self._mmap = mmap.mmap(
                self._file.fileno(), length=0, access=mmap.ACCESS_READ
            )
            header_len = int(struct.unpack("<Q", self._mmap[:8])[0])
            header_end = 8 + header_len
            if header_end > len(self._mmap):
                raise ValueError(
                    f"safetensors header exceeds file size in {self.path}."
                )
            payload = json.loads(self._mmap[8:header_end].decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(
                    f"safetensors header must be an object: {self.path}."
                )
            self.header = payload
            self.data_start = header_end
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        mapping, self._mmap = self._mmap, None
        file_obj, self._file = self._file, None
        if mapping is not None:
            mapping.close()
        if file_obj is not None:
            file_obj.close()

    def __enter__(self) -> "SafeTensorReader":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def keys(self) -> tuple[str, ...]:
        return tuple(name for name in self.header if name != "__metadata__")

    def _tensor_info(self, name: str) -> dict[str, Any]:
        if name not in self.header:
            raise KeyError(f"Tensor not found in {self.path.name}: {name}")
        info = self.header[name]
        if name == "__metadata__" or not isinstance(info, dict):
            raise KeyError(f"{name} is not a real tensor entry.")
        return info

    def get_tensor(
        self,
        name: str,
        out_dtype: np.dtype | None = np.float32,
    ) -> np.ndarray:
        """Decode a named tensor, including portable BF16 conversion."""

        if self._mmap is None:
            raise RuntimeError(f"safetensors reader is closed: {self.path}.")
        info = self._tensor_info(name)
        try:
            dtype_tag = str(info["dtype"])
            shape = tuple(int(value) for value in info["shape"])
            rel_start, rel_end = (
                int(value) for value in info["data_offsets"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid safetensors metadata for {name} in {self.path.name}."
            ) from exc
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"tensor {name} has a negative shape: {shape}.")
        if rel_start < 0 or rel_end < rel_start:
            raise ValueError(
                f"tensor {name} has invalid data offsets {(rel_start, rel_end)}."
            )
        start = self.data_start + rel_start
        end = self.data_start + rel_end
        if end > len(self._mmap):
            raise ValueError(f"tensor {name} extends past {self.path.name}.")
        itemsize = (
            2
            if dtype_tag == "BF16"
            else (
                _SIMPLE_DTYPES[dtype_tag].itemsize
                if dtype_tag in _SIMPLE_DTYPES
                else None
            )
        )
        if itemsize is None:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_tag}")
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * int(itemsize)
        if end - start != expected_bytes:
            raise ValueError(
                f"tensor {name} stores {end - start} bytes, expected "
                f"{expected_bytes} for shape={shape}, dtype={dtype_tag}."
            )
        buffer = memoryview(self._mmap)[start:end]
        if dtype_tag == "BF16":
            raw = np.frombuffer(buffer, dtype=np.dtype("<u2")).reshape(shape)
            value = (raw.astype(np.uint32) << 16).view(np.float32)
            if out_dtype is not None and np.dtype(out_dtype) != np.float32:
                return value.astype(out_dtype)
            return value.copy()
        value = np.frombuffer(
            buffer, dtype=_SIMPLE_DTYPES[dtype_tag]
        ).reshape(shape)
        if out_dtype is not None and np.dtype(out_dtype) != value.dtype:
            return value.astype(out_dtype)
        return value.copy()


class WeightLoader:
    """Resolve single or indexed safetensors files, mapping shards lazily."""

    def __init__(
        self,
        model_dir: str | Path,
        tensor_dtype: np.dtype = np.float32,
    ):
        self.model_dir = Path(model_dir)
        self.tensor_dtype = np.dtype(tensor_dtype)
        self.readers: dict[str, SafeTensorReader] = {}
        self.weight_map: dict[str, str] | None = None
        self._single_filename: str | None = None
        self._closed = False
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(
                payload.get("weight_map"), dict
            ):
                raise ValueError(f"invalid safetensors index: {index_path}.")
            self.weight_map = {
                str(name): str(filename)
                for name, filename in payload["weight_map"].items()
            }
            if not self.weight_map:
                raise ValueError(f"empty safetensors weight map: {index_path}.")
        else:
            path = self.model_dir / "model.safetensors"
            if not path.exists():
                raise FileNotFoundError(
                    "Neither model.safetensors nor model.safetensors.index.json "
                    f"found in {self.model_dir}"
                )
            self._single_filename = path.name

    def _reader(self, filename: str) -> SafeTensorReader:
        if self._closed:
            raise RuntimeError("weight loader is closed.")
        reader = self.readers.get(filename)
        if reader is None:
            reader = SafeTensorReader(self.model_dir / filename)
            self.readers[filename] = reader
        return reader

    def close(self) -> None:
        readers, self.readers = tuple(self.readers.values()), {}
        self._closed = True
        for reader in readers:
            reader.close()

    def __enter__(self) -> "WeightLoader":
        if self._closed:
            raise RuntimeError("weight loader is closed.")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def has_tensor(self, name: str) -> bool:
        if self._closed:
            raise RuntimeError("weight loader is closed.")
        if self.weight_map is not None:
            return name in self.weight_map
        return name in self._reader(str(self._single_filename))

    def get(
        self,
        name: str,
        out_dtype: np.dtype | None = None,
    ) -> np.ndarray:
        out_dtype = (
            self.tensor_dtype if out_dtype is None else np.dtype(out_dtype)
        )
        if self.weight_map is None:
            return self._reader(str(self._single_filename)).get_tensor(
                name, out_dtype=out_dtype
            )
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"Tensor {name} not found in sharded weight map.")
        return self._reader(filename).get_tensor(name, out_dtype=out_dtype)


def load_layer_weights(loader: WeightLoader, layer_idx: int) -> LayerWeights:
    """Load and shape-check the nine arrays consumed by one layer."""

    layer_idx = int(layer_idx)
    if layer_idx < 0:
        raise ValueError("layer_idx must be non-negative.")
    prefix = f"model.layers.{layer_idx}"

    def load(name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = loader.get(f"{prefix}.{name}")
        if value.shape != shape:
            raise ValueError(
                f"{prefix}.{name} must have shape {shape}, got {value.shape}."
            )
        return value

    return LayerWeights(
        input_layernorm=load_norm_weight(
            loader, f"{prefix}.input_layernorm.weight", 4096
        ),
        q_proj=load("self_attn.q_proj.weight", (4096, 4096)),
        k_proj=load("self_attn.k_proj.weight", (1024, 4096)),
        v_proj=load("self_attn.v_proj.weight", (1024, 4096)),
        o_proj=load("self_attn.o_proj.weight", (4096, 4096)),
        post_attention_layernorm=load_norm_weight(
            loader, f"{prefix}.post_attention_layernorm.weight", 4096
        ),
        gate_proj=load("mlp.gate_proj.weight", (14336, 4096)),
        up_proj=load("mlp.up_proj.weight", (14336, 4096)),
        down_proj=load("mlp.down_proj.weight", (4096, 14336)),
    )


def load_norm_weight(
    loader: WeightLoader,
    name: str,
    hidden_size: int,
) -> np.ndarray:
    """Load a normal gamma or expand QuaRot's scalar RMSN placeholder.

    QuaRot fuses gamma into adjacent Linear weights and replaces every
    ``LlamaRMSNorm`` with its weightless ``RMSN`` module.  That module registers
    a shape-``(1,)`` zero parameter only for state-dict compatibility; its
    forward pass does not use the parameter.  The equivalent explicit FHE
    RMSNorm therefore has unit gamma, not zero gamma.
    """

    hidden_size = int(hidden_size)
    value = np.asarray(loader.get(name), dtype=np.float32)
    if value.shape == (hidden_size,):
        return value
    if value.shape == (1,):
        scalar = float(value[0])
        if scalar == 0.0:
            scalar = 1.0
        return np.full((hidden_size,), scalar, dtype=np.float32)
    raise ValueError(
        f"{name} must have shape {(hidden_size,)} or a fused scalar placeholder, "
        f"got {value.shape}."
    )


__all__ = [
    "CheckpointMetadata",
    "SafeTensorReader",
    "WeightLoader",
    "load_layer_weights",
    "load_norm_weight",
    "validate_runtime_checkpoint",
]
