"""Streaming one layer's weights at a time, one layer ahead.

The 32-layer model never holds all its weights: each layer is loaded, used and
released while the next one loads on a background thread. That double buffer is
why a 15 GiB checkpoint runs in a fraction of that, and why the loader has to
be explicit about ownership — a layer's arrays are dropped as soon as the layer
that needed them is done.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from .. import Llama3CKKSConfig
from ..layouts import AttentionPairLayout, FeatureMajorPrefillLayout
from ..model import LayerWeights, TransformerLayer
from .approx_config import ApproximationConfig
from .checkpoint import WeightLoader, load_layer_weights


class DoubleBufferedLayerStream:
    """One-layer lookahead for raw checkpoint arrays.

    The worker reads and converts layer ``i+1`` while the main thread executes
    layer ``i`` on CUDA. Two host slots are sufficient: the current immutable
    ``LayerWeights`` object and one future result. Operator-specific GPU
    packing remains online and is intentionally not hidden by this loader.
    """

    def __init__(
        self,
        *,
        loader: WeightLoader | None,
        synthetic_weights: LayerWeights | None,
        approx_config: ApproximationConfig,
        config: Llama3CKKSConfig,
        feature_layout: FeatureMajorPrefillLayout,
        attention_layout: AttentionPairLayout,
        num_layers: int,
        rope_theta: float,
    ) -> None:
        self.loader = loader
        self.synthetic_weights = synthetic_weights
        self.approx_config = approx_config
        self.config = config
        self.feature_layout = feature_layout
        self.attention_layout = attention_layout
        self.num_layers = int(num_layers)
        self.rope_theta = float(rope_theta)
        self.index = 0
        self._executor = (
            None
            if loader is None
            else ThreadPoolExecutor(max_workers=1, thread_name_prefix="weight-prefetch")
        )
        self._future: Future | None = None
        self._load_rows: dict[int, dict[str, object]] = {}
        self._compute_windows: dict[int, tuple[float, float]] = {}
        self._closed = False
        if self._executor is not None and self.num_layers:
            self._future = self._submit(0)

    def _submit(self, layer_idx: int) -> Future:
        submitted = time.perf_counter()

        def load():
            started = time.perf_counter()
            weights = load_layer_weights(self.loader, layer_idx)  # type: ignore[arg-type]
            finished = time.perf_counter()
            return weights, {
                "layer": int(layer_idx),
                "submitted_at": float(submitted),
                "load_started_at": float(started),
                "load_finished_at": float(finished),
                "load_seconds": float(finished - started),
                "raw_bytes": int(
                    sum(
                        int(getattr(weights, name).nbytes)
                        for name in LayerWeights.__dataclass_fields__
                    )
                ),
                "worker_thread": int(threading.get_ident()),
            }

        return self._executor.submit(load)  # type: ignore[union-attr]

    def __iter__(self):
        return self

    def __next__(self) -> TransformerLayer:
        if self.index >= self.num_layers:
            raise StopIteration
        layer_idx = int(self.index)
        if self.loader is None:
            weights = self.synthetic_weights
            if weights is None:
                raise RuntimeError("synthetic layer stream has no weights")
        else:
            if self._future is None:
                raise RuntimeError("raw-weight prefetch future is missing")
            wait_started = time.perf_counter()
            weights, row = self._future.result()
            wait_finished = time.perf_counter()
            row["consumer_wait_seconds"] = float(wait_finished - wait_started)
            row["consumer_thread"] = int(threading.get_ident())
            self._load_rows[layer_idx] = row
        self.index += 1
        self._future = (
            self._submit(self.index)
            if self._executor is not None and self.index < self.num_layers
            else None
        )
        return TransformerLayer(
            index=layer_idx,
            weights=weights,
            approximations=self.approx_config.for_layer(layer_idx),
            config=self.config,
            feature_layout=self.feature_layout,
            attention_layout=self.attention_layout,
            rope_theta=self.rope_theta,
        )

    def record_compute_window(
        self, layer_idx: int, started: float, finished: float
    ) -> None:
        self._compute_windows[int(layer_idx)] = (float(started), float(finished))

    def stats(self) -> dict[str, object]:
        rows = []
        for layer_idx in sorted(self._load_rows):
            row = dict(self._load_rows[layer_idx])
            previous = self._compute_windows.get(layer_idx - 1)
            overlap = 0.0
            if previous is not None:
                overlap = max(
                    0.0,
                    min(float(row["load_finished_at"]), previous[1])
                    - max(float(row["load_started_at"]), previous[0]),
                )
            row["overlap_with_previous_layer_seconds"] = float(overlap)
            row["asynchronous"] = bool(
                layer_idx > 0
                and (
                    int(row["worker_thread"]) != int(row["consumer_thread"])
                    and overlap > 0.0
                )
            )
            rows.append(row)
        adjacent_bytes = [
            int(rows[index]["raw_bytes"]) + int(rows[index + 1]["raw_bytes"])
            for index in range(max(0, len(rows) - 1))
        ]
        return {
            "enabled": bool(self.loader is not None and self.num_layers > 1),
            "lookahead_layers": 1 if self.loader is not None else 0,
            "layers": rows,
            "total_load_seconds": float(
                sum(float(row["load_seconds"]) for row in rows)
            ),
            "total_consumer_wait_seconds": float(
                sum(float(row["consumer_wait_seconds"]) for row in rows)
            ),
            "total_overlap_seconds": float(
                sum(float(row["overlap_with_previous_layer_seconds"]) for row in rows)
            ),
            "peak_two_slot_host_bytes": (
                max(adjacent_bytes)
                if adjacent_bytes
                else (int(rows[0]["raw_bytes"]) if rows else 0)
            ),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._future = None


__all__ = ["DoubleBufferedLayerStream"]
