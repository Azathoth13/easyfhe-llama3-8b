"""The Algorithm-2 softmax artifact, as the encrypted kernel consumes it.

``SoftmaxPolyConfig`` is a typed view of ``assets/polynomials/
softmax_poly_coeffs.json``: the per-layer shift/clamp constants, the exp
polynomial entries, the two-stage inverse-square-root polynomials, and the
per-layer Alg2 round ladder. The encrypted kernel
(:mod:`llama3fhe.operators.attention.softmax`) reads exactly this surface;
plaintext evaluation of the same mathematics lives with the exact references
in :mod:`llama3fhe.approx.attention`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

_REQUIRED_LAYER_KEYS = ("U_hat", "stats_min", "coefficients", "fit_interval")


@dataclass(frozen=True)
class SoftmaxPolyConfig:
    """Release Alg2 softmax constants, keyed by layer index."""

    k: int
    k_by_layer: Dict[int, int]
    layer_U_hat: Dict[int, float]
    layer_stats_min: Dict[int, float]
    layer_exp: Dict[int, Dict[str, Any]]
    invsqrt_alg2: Dict[str, Dict[str, Any]]

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> SoftmaxPolyConfig:
        layer_U_hat: Dict[int, float] = {}
        layer_stats_min: Dict[int, float] = {}
        layer_exp: Dict[int, Dict[str, Any]] = {}
        k_by_layer: Dict[int, int] = {}
        for key, row in artifact.get("layers", {}).items():
            layer = int(key)
            missing = [name for name in _REQUIRED_LAYER_KEYS if name not in row]
            if missing:
                raise ValueError(
                    f"softmax artifact layer {layer} is missing {missing}."
                )
            layer_U_hat[layer] = float(row["U_hat"])
            layer_stats_min[layer] = float(row["stats_min"])
            layer_exp[layer] = row
            if "k" in row:
                k_by_layer[layer] = int(row["k"])
        # The top-level ladder overrides / fills gaps: {"0": 4, "1": 3, ...}.
        for key, value in (artifact.get("k_by_layer") or {}).items():
            k_by_layer[int(key)] = int(value)
        invsqrt = artifact.get("invsqrt_alg2", {})
        for stage in ("j1_rough", "j2_precise"):
            if stage not in invsqrt:
                raise ValueError(f"softmax artifact is missing invsqrt {stage}.")
        return cls(
            k=int(artifact.get("k", 2)),
            k_by_layer=k_by_layer,
            layer_U_hat=layer_U_hat,
            layer_stats_min=layer_stats_min,
            layer_exp=layer_exp,
            invsqrt_alg2=invsqrt,
        )

    def k_for(self, layer_idx: int) -> int:
        """Alg2 round count for ``layer_idx`` (ladder entry, else global)."""

        return int(self.k_by_layer.get(int(layer_idx), self.k))

    def U_hat_for(self, layer_idx: int) -> float:
        return self.layer_U_hat[int(layer_idx)]

    def stats_min_for(self, layer_idx: int) -> float:
        return self.layer_stats_min[int(layer_idx)]

    def poly_for_layer(
        self, poly_entry: Dict[str, Any], layer_idx: int
    ) -> Dict[str, Any]:
        """Select a per-layer polynomial variant when the entry carries one.

        Public because the encrypted kernel needs exactly this selection: it
        used to reach for the private name through getattr and carry its own
        fallback copy of the logic, which could then diverge silently.
        """

        by_layer = poly_entry.get("by_layer")
        if isinstance(by_layer, dict):
            return by_layer.get(str(int(layer_idx)), poly_entry)
        return poly_entry


def load_softmax_poly_coeffs(path: str | Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


__all__ = ["SoftmaxPolyConfig", "load_softmax_poly_coeffs"]
