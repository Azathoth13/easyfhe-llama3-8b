#!/usr/bin/env bash
# Layer-0 quick check: audit + optional synthetic encrypted smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[check_layer0] audit-only..."
"$PYTHON_BIN" run_layer0.py --audit-only

if [[ "${LAYER0_SYNTHETIC:-1}" == "1" ]]; then
  echo "[check_layer0] synthetic encrypted Layer-0..."
  "$PYTHON_BIN" run_layer0.py --weights-source synthetic
fi

echo "[check_layer0] DONE"
