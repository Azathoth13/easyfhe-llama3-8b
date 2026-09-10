from __future__ import annotations

"""Load nonlinear approximation artifacts and select each layer's config."""

from pathlib import Path

from ..approx.rmsnorm import get_layer_coeffs, load_rmsnorm_poly_coeffs
from ..approx.silu import load_silu_poly_coeffs
from ..approx.softmax import (
    SoftmaxPolyConfig,
    load_softmax_poly_coeffs,
)
from ..model import LayerApproximations


class ApproximationConfig:
    """Immutable approximation config with cached per-layer selection."""

    def __init__(
        self,
        *,
        rmsnorm_path: str | Path,
        silu_path: str | Path,
        softmax_path: str | Path,
    ) -> None:
        self.paths = {
            "rmsnorm": str(Path(rmsnorm_path)),
            "silu": str(Path(silu_path)),
            "softmax": str(Path(softmax_path)),
        }
        self._rmsnorm = load_rmsnorm_poly_coeffs(rmsnorm_path)
        self._silu = load_silu_poly_coeffs(silu_path)
        self._softmax = SoftmaxPolyConfig.from_artifact(
            load_softmax_poly_coeffs(softmax_path)
        )
        self._layers: dict[int, LayerApproximations] = {}

    def for_layer(self, layer_idx: int) -> LayerApproximations:
        layer_idx = int(layer_idx)
        if layer_idx < 0:
            raise ValueError("layer_idx must be non-negative.")
        cached = self._layers.get(layer_idx)
        if cached is not None:
            return cached
        key = str(layer_idx)
        per_layer = self._silu.get("per_layer", {})
        try:
            silu = per_layer[key]["tiers"]["standard"]
        except KeyError as exc:
            raise KeyError(
                f"SiLU artifact has no standard tier for layer {key}."
            ) from exc
        selected = LayerApproximations(
            input_norm=get_layer_coeffs(self._rmsnorm, "attn", key),
            post_attention_norm=get_layer_coeffs(self._rmsnorm, "ffn", key),
            silu=silu,
            softmax=self._softmax,
        )
        self._layers[layer_idx] = selected
        return selected

    def validate_layers(self, count: int) -> None:
        """Fail before context/key generation if any requested layer is absent."""

        count = int(count)
        if count <= 0:
            raise ValueError("layer count must be positive.")
        for layer_idx in range(count):
            self.for_layer(layer_idx)
            if (
                layer_idx not in self._softmax.layer_U_hat
                or layer_idx not in self._softmax.layer_stats_min
                or layer_idx not in self._softmax.layer_exp
            ):
                raise KeyError(
                    f"Softmax artifact has no complete entry for layer {layer_idx}."
                )

    def degree_summary(self, layer_idx: int) -> dict[str, int]:
        """Return the three layer-specific polynomial degrees for reports."""

        selected = self.for_layer(layer_idx)
        return {
            "input_rmsnorm": len(selected.input_norm["coefficients"]) - 1,
            "post_attention_rmsnorm": (
                len(selected.post_attention_norm["coefficients"]) - 1
            ),
            "silu": len(selected.silu["coefficients"]) - 1,
        }


__all__ = ["ApproximationConfig"]
