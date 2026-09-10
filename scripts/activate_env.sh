#!/usr/bin/env bash
# Activate the environment created by setup_env.sh.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ODIN_FHE_ENV_DIR:-$ROOT/.venv}"
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
  echo "Missing environment: $ENV_DIR; run ./scripts/setup_env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export ODIN_FHE_ROOT="$ROOT"
export MODEL_DIR="${MODEL_DIR:-$ROOT/assets/weights/Meta-Llama-3-8B-QuaRot}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${EASYFHE_SOURCE_DIR:-}" ]]; then
  export PYTHONPATH="${EASYFHE_SOURCE_DIR}:${ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
fi
echo "Odin FHE Llama 3 environment activated"
echo "  MODEL_DIR=$MODEL_DIR"
