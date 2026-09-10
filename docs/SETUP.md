# Setup

## Requirements

| | Reference |
|--|-----------|
| GPU | NVIDIA **A100 80GB** (other 80GB-class GPUs may work; try Layer-0 first) |
| GPU memory | Paris128 peak **~65 GiB** measured (plan for **~70 GiB** free) |
| Host RAM | 64GB+ recommended |
| OS / Python | Linux x86_64 · **Python 3.12** |
| EasyFHE | local source checkout or wheel exposing the APIs below |

## 1. EasyFHE environment

This release does **not** redistribute EasyFHE. Point at a compiled checkout or wheel:

```bash
export EASYFHE_SOURCE_DIR=/path/to/EasyFHE   # preferred
# or: export EASYFHE_WHEEL=/path/to/easyfhe-*.whl

./scripts/setup_env.sh
source ./scripts/activate_env.sh
python -c "import easyfhe; print('ok')"
```

The EasyFHE build must provide:

- CKKS context creation (`generate_client_context` / `CKKSContextSpec`);
- `ConstantBundle.plaintext`, `PackedRaw`, `homo_mul_i`, `homo_mul_no_relin`, `homo_relinearize`;
- extended hoisted `hoisted_mac_sum` with reusable baby-step rotations;
- OpenFHE bootstrap (`BootstrapSpec`, `requirements`, `generate`, `bootstrap`);
- `Context.clear_cuda_rotation_cache(keep_rotations=...)`.

Upstream: https://github.com/jizhuoran/EasyFHE (pin may change; use a revision that matches the contract above).

Optional (tokenization; Paris fixture already has `input_ids`):

```bash
pip install transformers tokenizers
```

## 2. QuaRot weights

FHE needs a **full-fused QuaRot** checkpoint. Meta weights are **not** redistributed.

```text
assets/weights/Meta-Llama-3-8B-QuaRot/
├── config.json
├── export_metadata.json   # format llama3_8b_feature_major_quarot_v1, full_fuse=true
└── model.safetensors      # or sharded index + shards
```

See [`assets/weights/README.md`](../assets/weights/README.md). Conversion tooling stays outside this minimal package; produce the checkpoint under Meta / QuaRot licenses before real-weight runs.

`--weights-source synthetic` exercises the full Layer-0 ciphertext graph without model files.

## 3. Release coefficients

Shipped under `assets/polynomials/` with clean names:

| File | Role |
|------|------|
| `rmsnorm_poly_coeffs.json` | RMSNorm Chebyshev |
| `silu_poly_coeffs.json` | SiLU Chebyshev |
| `softmax_poly_coeffs.json` | Softmax Alg2 |

Override with `--rms-coeffs` / `--silu-coeffs` / `--softmax-coeffs` if needed.

Next: [RUN.md](RUN.md).
