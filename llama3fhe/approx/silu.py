from __future__ import annotations

"""Loader for the release SiLU approximation artifact.

Runtime SiLU evaluation lives in
:mod:`llama3fhe.operators.nonlinear.elementwise`; this module only reads the
release artifact.
"""

import json
from pathlib import Path
from typing import Any


def load_silu_poly_coeffs(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


__all__ = ["load_silu_poly_coeffs"]
