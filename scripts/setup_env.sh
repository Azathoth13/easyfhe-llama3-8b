#!/usr/bin/env bash
# Create a Python environment for the Odin FHE Llama 3 runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ODIN_FHE_ENV_DIR:-$ROOT/.venv}"
SOURCE_DIR="${EASYFHE_SOURCE_DIR:-}"
WHEEL="${EASYFHE_WHEEL:-}"

if [[ -z "$SOURCE_DIR" && -f "$HOME/EasyFHE/easyfhe/__init__.py" ]]; then
  SOURCE_DIR="$HOME/EasyFHE"
fi
if [[ -z "$SOURCE_DIR" && -z "$WHEEL" ]]; then
  echo "Set EASYFHE_SOURCE_DIR to a compiled EasyFHE checkout or EASYFHE_WHEEL to a local wheel." >&2
  exit 1
fi
if [[ -n "$SOURCE_DIR" && ! -f "$SOURCE_DIR/easyfhe/__init__.py" ]]; then
  echo "Invalid EASYFHE_SOURCE_DIR=$SOURCE_DIR" >&2
  exit 2
fi
if [[ -n "$WHEEL" && ! -f "$WHEEL" ]]; then
  echo "Invalid EASYFHE_WHEEL=$WHEEL" >&2
  exit 2
fi

if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
  python3.12 -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$ROOT"
if [[ -n "$SOURCE_DIR" ]]; then
  echo "Using EasyFHE source checkout: $SOURCE_DIR"
else
  python -m pip install --no-deps --no-index "$WHEEL"
fi

echo "Odin FHE Llama 3 environment is ready."
echo "  source $ROOT/scripts/activate_env.sh"
