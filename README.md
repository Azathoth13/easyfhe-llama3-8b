# Odin FHE Llama-3-8B

Encrypted **Llama-3-8B** prefill on GPU CKKS with Odin **feature-major** packing.
Every inter-layer boundary stays encrypted.

| | |
|--|--|
| Golden | Paris128 → token **` Paris`** (`12366`) |
| Reference | **~511 s** · peak **~65 GiB** · **A100 80GB** |
| Layout | Persistent feature-major (`feature_major`) |

License: **GPLv3** — see [`LICENSE`](LICENSE). Third-party notices: [`NOTICE`](NOTICE).

## Quick start

```bash
cd easyfhe-llama3-8b

# EasyFHE: set EASYFHE_SOURCE_DIR or EASYFHE_WHEEL (see docs/SETUP.md)
./scripts/setup_env.sh
source ./scripts/activate_env.sh

python run_layer0.py --audit-only
python run_layer0.py --weights-source synthetic   # encrypted Layer-0 smoke

# Full model (prepared QuaRot weights required)
python run_model.py \
  --model-dir assets/weights/Meta-Llama-3-8B-QuaRot \
  --num-layers 32
```

## Docs

| Doc | Contents |
|-----|----------|
| [docs/SETUP.md](docs/SETUP.md) | Hardware, EasyFHE env, weights |
| [docs/RUN.md](docs/RUN.md) | Layer-0 / Paris128 E2E, troubleshooting |
| [docs/DESIGN.md](docs/DESIGN.md) | Architecture, layout, package map |

## Layout

```text
easyfhe-llama3-8b/
├── run_layer0.py / run_model.py
├── llama3fhe/                 # CKKS runtime
├── assets/
│   ├── feature_major.json     # CKKS context config
│   ├── paris128_example.json  # canonical 128-token fixture
│   ├── polynomials/           # release coeffs (clean names)
│   └── weights/               # user-provided QuaRot checkpoint
├── scripts/                   # setup / activate
└── docs/
```

Not shipped: Meta weights, EasyFHE binary, poly-fit research code, paper/bench trees.
