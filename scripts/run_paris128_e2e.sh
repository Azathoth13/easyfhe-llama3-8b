#!/usr/bin/env bash
# Canonical Paris128 32-layer encrypted decode.
# Reference (NVIDIA A100 80GB): ~511 s wall · peak ~65 GiB · token 12366.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${MODEL_DIR:-$ROOT/assets/weights/Meta-Llama-3-8B-QuaRot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_IDS="${INPUT_IDS:-assets/paris128_example.json}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "error: model dir not found: $MODEL_DIR (see docs/SETUP.md)" >&2
  exit 1
fi

echo "[paris128] model=$MODEL_DIR"
echo "[paris128] expect token_id=12366 ( Paris)"
echo "[paris128] reference wall ~511s on A100 80GB; peak GPU ~65 GiB"

"$PYTHON_BIN" run_model.py \
  --model-dir "$MODEL_DIR" \
  --input-ids-file "$INPUT_IDS" \
  --num-layers 32 \
  --expect-token-id 12366

echo "[paris128] PASS"
