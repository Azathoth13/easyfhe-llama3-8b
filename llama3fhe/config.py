from __future__ import annotations

"""Runtime configuration for the feature-major release path.

Operator schedules live beside their operators. This module contains only the
shared EasyFHE context and plaintext encoding settings that cross module
boundaries.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimulatorConfig:
    logN: int = 16
    maxLevelsRemaining: int = 37
    level_budget: tuple[int, int] = (4, 4)
    dnum: int = 3
    dcrtBits: int = 59
    firstMod: int = 60
    secretKeyDist: str = "SPARSE_TERNARY"
    rescaleTech: str = "FIXEDMANUAL"
    device: str = "cuda"

    def rescale_policy(self) -> str:
        value = str(self.rescaleTech).lower()
        if value in {"fixedmanual", "fixed_manual", "manual"}:
            return "manual"
        if value in {"fixedauto", "fixed_auto", "auto"}:
            return "auto"
        raise ValueError(f"unsupported EasyFHE rescale policy: {value!r}")

    def validate(self) -> None:
        if int(self.logN) <= 1:
            raise ValueError("simulator.logN must be greater than one.")
        if int(self.maxLevelsRemaining) <= 0:
            raise ValueError("simulator.maxLevelsRemaining must be positive.")
        if int(self.dnum) <= 0 or int(self.dcrtBits) <= 0 or int(self.firstMod) <= 0:
            raise ValueError("simulator dnum/modulus parameters must be positive.")
        self.rescale_policy()


@dataclass
class QKVLinearConfig:
    """Encoding metadata shared by the feature-major linear frontends."""

    input_level: int = 0
    dtype: str = "float64"


@dataclass
class LayoutTransformConfig:
    dtype: str = "float64"


@dataclass
class Llama3CKKSConfig:
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    qkv_linear: QKVLinearConfig = field(default_factory=QKVLinearConfig)
    layout_transform: LayoutTransformConfig = field(
        default_factory=LayoutTransformConfig
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "Llama3CKKSConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        simulator = data.get("simulator", {})
        qkv = data.get("qkv_linear", {})
        transform = data.get("layout_transform", {})
        level_budget = tuple(
            int(value) for value in simulator.get("level_budget", (4, 4))
        )
        if len(level_budget) != 2:
            raise ValueError("simulator.level_budget must contain two integers.")
        result = cls(
            simulator=SimulatorConfig(
                logN=int(simulator.get("logN", 16)),
                maxLevelsRemaining=int(simulator.get("maxLevelsRemaining", 37)),
                level_budget=level_budget,
                dnum=int(simulator.get("dnum", 3)),
                dcrtBits=int(simulator.get("dcrtBits", 59)),
                firstMod=int(simulator.get("firstMod", 60)),
                secretKeyDist=str(
                    simulator.get("secretKeyDist", "SPARSE_TERNARY")
                ),
                rescaleTech=str(simulator.get("rescaleTech", "FIXEDMANUAL")),
                device=str(simulator.get("device", "cuda")),
            ),
            qkv_linear=QKVLinearConfig(
                input_level=int(qkv.get("input_level", 0)),
                dtype=str(qkv.get("dtype", "float64")),
            ),
            layout_transform=LayoutTransformConfig(
                dtype=str(transform.get("dtype", "float64")),
            ),
        )
        result.simulator.validate()
        return result


__all__ = [
    "LayoutTransformConfig",
    "Llama3CKKSConfig",
    "QKVLinearConfig",
    "SimulatorConfig",
]
