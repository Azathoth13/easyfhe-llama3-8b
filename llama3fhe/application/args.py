"""Command-line surface for the encrypted Llama-3 runners.

One parser serves both entry points: ``run_model.py`` gets the full 32-layer
defaults, ``run_layer0.py`` passes ``layer0_preset=True`` for the single-layer
defaults. Keeping it here means the flag surface can be read without paging
through the orchestration, and the orchestration never has to mention argparse.

Every flag that selects a *performance* default is documented by its help
text; this module states what the flag does without depending on benchmark
archives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = PROJECT_ROOT / "assets/feature_major.json"

DEFAULT_RMS = PROJECT_ROOT / "assets/polynomials/rmsnorm_poly_coeffs.json"
DEFAULT_SILU = PROJECT_ROOT / "assets/polynomials/silu_poly_coeffs.json"
DEFAULT_SOFTMAX = PROJECT_ROOT / "assets/polynomials/softmax_poly_coeffs.json"
DEFAULT_INPUT_IDS = PROJECT_ROOT / "assets/paris128_example.json"


def _fixture_expected_token_id(path: str | Path) -> int | None:
    """Return an optional next-token assertion embedded in an input fixture."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    value = payload.get("expected_next_token_id")
    return None if value is None else int(value)


def build_parser(*, layer0_preset: bool = False) -> argparse.ArgumentParser:
    """Build the shared CLI; the Layer0 executable changes defaults only."""

    scope = "one layer" if layer0_preset else "a streamed model"
    parser = argparse.ArgumentParser(
        description=f"Run {scope} of persistent feature-major Llama-3-8B."
    )
    parser.add_argument("--model-dir", type=Path)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--input-ids-file",
        type=Path,
        help=(
            "JSON token fixture. run_model.py defaults to "
            "assets/paris128_example.json when neither input option is "
            "provided."
        ),
    )
    sources.add_argument(
        "--prompt",
        help="Tokenize this prompt instead of using the canonical fixture.",
    )
    parser.set_defaults(
        default_input_ids_file=None if layer0_preset else DEFAULT_INPUT_IDS
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    if layer0_preset:
        parser.set_defaults(num_layers=1)
    else:
        parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument(
        "--weights-source",
        choices=("real", "synthetic"),
        default="synthetic" if layer0_preset else "real",
    )
    parser.add_argument("--seed", type=int, default=26081461)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--input-level",
        type=int,
        default=None,
        help=(
            "Initial ciphertext level. Auto selects L0 for an ordinary "
            "single-layer run, L(depth-1) for --steady-state-layer, and "
            "the bootstrap output level for a streamed multi-layer run."
        ),
    )
    parser.add_argument(
        "--steady-state-layer",
        action="store_true",
        help=(
            "Run Layer0 weights with the same input refresh and attention "
            "checkpoint schedule as a later transformer layer."
        ),
    )
    parser.add_argument(
        "--softmax-bootstrap-position",
        choices=("auto", "pre", "post", "off"),
        default="auto",
        help=(
            "Place the steady-state Softmax main-stream refresh before or "
            "after exp, disable it, or use auto (pre for steady state)."
        ),
    )
    parser.add_argument(
        "--softmax-deep-main-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Refresh the steady-layer Softmax main checkpoint through a "
            "second (3,3)-budget bootstrap program whose smaller depth "
            "frees two extra output levels. The PV/PairFM chain then "
            "arrives at the 6-limb threshold and skips the paired-input "
            "re-bootstrap entirely."
        ),
    )
    parser.add_argument(
        "--pairfm-bootstrap-mode",
        choices=("auto", "always", "off"),
        default="auto",
        help=(
            "Refresh PV output before PairFM always, never, or only when "
            "PairFM + W_O would leave fewer than two limbs."
        ),
    )
    parser.add_argument(
        "--pairfm-post-refresh-input-limbs",
        type=int,
        default=None,
        help=(
            "After a PairFM checkpoint, modulus-drop the refreshed carriers "
            "to this limb count without changing their CKKS scale. Defaults "
            "to the 6 limbs the PairFM + W_O suffix needs. Set 0 to disable "
            "the drop."
        ),
    )
    parser.add_argument(
        "--softmax-complex-pre-exp-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pair adjacent real score ciphertexts into complex values for "
            "the pre-exp checkpoint and fuse the exp domain map into score "
            "preprocessing."
        ),
    )
    parser.add_argument(
        "--softmax-reuse-round-square",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse each Alg2 round's row-sum square in the main update, "
            "computing y^2*lambda^2 instead of recomputing (y*lambda)^2."
        ),
    )
    parser.add_argument(
        "--softmax-direct-slim-inactive-fill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fill inactive slim polynomial rows directly after the disjoint "
            "slim pack, avoiding a redundant active-row mask/rescale."
        ),
    )
    parser.add_argument(
        "--softmax-fuse-slim-chebyshev-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fuse the inverse-square-root Chebyshev domain map into the "
            "existing slim-pack plaintext and fill inactive rows by addition."
        ),
    )
    parser.add_argument(
        "--adaptive-rmsnorm-slim-refresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refresh the single RMSNorm rsqrt slim ciphertext when needed, "
            "instead of refreshing the 16 materialized normalized outputs "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--rmsnorm2-slim-bootstrap-log-slots",
        type=int,
        default=None,
        help=(
            "Optional sparse bootstrap program for the RMSN2 rsqrt slim "
            "refresh only (e.g. 12). RMSN1 recovery stays on the full-slot "
            "program. Default: reuse the main program."
        ),
    )
    parser.add_argument(
        "--batched-sparse-attention-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Batch sparse attention-input masks and reuse each input's "
            "hoisted baby rotations (GPU batch cap 32; default: enabled)."
        ),
    )
    parser.add_argument(
        "--fuse-rmsnorm-output-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Absorb RMSNorm gamma/paired-bootstrap scale into the following "
            "QKV or gate/up plaintext weights (default: enabled)."
        ),
    )
    parser.add_argument(
        "--fuse-silu-domain-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Absorb the SiLU Chebyshev alpha into gate plaintext weights "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--bootstrap-post-levels",
        type=int,
        help=(
            "Bootstrap output levels (default: depth-18 for the (4,4) program)."
        ),
    )
    parser.add_argument(
        "--softmax-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        default=2,
        help="Native bootstrap passes per Softmax refresh (default: 2).",
    )
    parser.add_argument(
        "--softmax-main-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Native passes for the 16/8-cipher main Softmax checkpoint "
            "only (default: 1; single-pass halves the dominant softmax "
            "bootstrap cost at unchanged layer numerics)."
        ),
    )
    parser.add_argument(
        "--softmax-lambda-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Native passes for compact Alg2 lambda refreshes only "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--softmax-single-pass-match-two-pass-level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After a one-pass Softmax bootstrap, cheaply modulus-drop one "
            "limb so downstream kernels see the same level as Meta-BTS "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--residual-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Native bootstrap passes per residual refresh (default: 1; use "
            "2 for the higher-precision reference schedule)."
        ),
    )
    parser.add_argument(
        "--input-residual-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        help="Override native passes for the layer-input residual checkpoint.",
    )
    parser.add_argument(
        "--post-attention-residual-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        help="Override native passes for the post-attention residual checkpoint.",
    )
    parser.add_argument(
        "--softmax-bootstrap-precision",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--residual-bootstrap-precision",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--residual-bootstrap-level-drop",
        type=int,
        default=0,
        help=(
            "Maximum modulus-only levels dropped after input/post-attention "
            "residual bootstraps. The layer clamps this per RMSNorm/consumer "
            "so attention and MLP retain their required limb budgets "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--preserve-prebootstrap-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep each pre-norm identity branch on its original ciphertext "
            "instead of materializing a residual copy from the RMSNorm "
            "bootstrap output (default: enabled)."
        ),
    )
    parser.add_argument(
        "--profile-device-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record allocated/reserved allocator memory and total device "
            "usage at major layer boundaries (default: disabled)."
        ),
    )
    parser.add_argument(
        "--profile-operator-breakdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep each measured layer's nested operator timing tree in the "
            "JSON report (default: disabled)."
        ),
    )
    parser.add_argument(
        "--plaintext-cache-gb",
        type=float,
        default=0.0,
        help=(
            "Session budget for cross-layer encoded public masks. Disabled "
            "by default because the current A100 two-layer profile is faster "
            "without it."
        ),
    )
    parser.add_argument(
        "--plaintext-cache-max-source-mb",
        type=float,
        default=1.0,
        help=(
            "Largest raw public-mask bundle eligible for the session cache; "
            "larger BSGS tables stay transient (default: 1 MiB)."
        ),
    )
    parser.add_argument(
        "--attention-rotation-chunk-size",
        type=int,
        choices=(1, 2, 4, 8, 16),
        default=8,
        help=(
            "Number of QK/PV shifts scheduled per fast-rotation chunk. "
            "PV emits two value rotations per shift, so 16 reaches the "
            "backend's 32-output fast-rotation limit."
        ),
    )
    parser.add_argument(
        "--linear-hoist-strategy",
        choices=("normal", "ext_normal", "ext_double_hoist"),
        default="normal",
        help=(
            "Hoisted-MAC schedule used by QKV, W_O, gate/up and down. "
            "The default is the measured B32 normal-hoist production "
            "schedule; this switch remains available for matched ablation."
        ),
    )
    parser.add_argument(
        "--rms-coeffs",
        type=Path,
        default=None,
        help="RMSNorm Chebyshev coeffs (default: assets/polynomials/rmsnorm_poly_coeffs.json).",
    )
    parser.add_argument(
        "--silu-coeffs",
        type=Path,
        default=None,
        help="SiLU Chebyshev coeffs (default: assets/polynomials/silu_poly_coeffs.json).",
    )
    parser.add_argument(
        "--softmax-coeffs",
        type=Path,
        default=None,
        help="Softmax Alg2 coeffs (default: assets/polynomials/softmax_poly_coeffs.json).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0 if layer0_preset else 1,
        help=(
            "Warmup repetitions. run_model.py defaults to one lightweight "
            "steady-layer warmup; run_layer0.py defaults to zero."
        ),
    )
    parser.add_argument(
        "--warmup-scope",
        choices=("auto", "steady-layer", "model"),
        default="auto",
        help=(
            "Warm one representative steady-state layer, the complete model, "
            "or choose steady-layer automatically for multi-layer runs."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--expect-token-id",
        type=int,
        help=(
            "Fail after writing the report if the predicted token ID differs. "
            "A fixture's expected_next_token_id is used when this is omitted."
        ),
    )
    parser.add_argument("--pad-side", choices=("left", "right"), default="left")
    parser.add_argument("--no-add-special-tokens", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--diagnostic-decrypt-layers",
        action="store_true",
        help=(
            "Debug only: decrypt each layer boundary and print FHE vs poly "
            "diffs (not hidden-state amplitude)."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--output-hidden-npy",
        type=Path,
        help="Diagnostic: save the final decrypted hidden state as a NumPy array.",
    )
    return parser


def resolve_poly_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Fill RMSNorm/SiLU/Softmax paths with release defaults unless overridden."""

    if args.rms_coeffs is None:
        args.rms_coeffs = DEFAULT_RMS
    if args.silu_coeffs is None:
        args.silu_coeffs = DEFAULT_SILU
    if args.softmax_coeffs is None:
        args.softmax_coeffs = DEFAULT_SOFTMAX
    return args


__all__ = [
    "build_parser",
    "resolve_poly_paths",
    "PROJECT_ROOT",
    "DEFAULT_CONFIG",
    "DEFAULT_RMS",
    "DEFAULT_SILU",
    "DEFAULT_SOFTMAX",
    "DEFAULT_INPUT_IDS",
]
