# Design

## Stack

| Layer | Role |
|-------|------|
| EasyFHE | GPU CKKS |
| Odin feature-major packing | Persistent FM hidden across layers |
| `llama3fhe/operators/` | Linear / attention / nonlinear / norm |
| `llama3fhe/layouts/` | Feature-major + attention C/Δ carriers |
| `assets/polynomials/` | Softmax / RMSNorm / SiLU release coeffs |

## Prefill (one layer)

```text
FM hidden
  -> RMSNorm
  -> complex-token QKV diagonal linear
  -> Q_C / K_Δ / V_Δ
  -> QK -> Δ-native Softmax -> PV
  -> PairFM -> feature-major W_O -> residual
  -> fused bootstrap+RMSNorm -> feature-major SwiGLU MLP -> residual
  -> FM hidden
```

Each layer is one continuous ciphertext graph. Softmax and residual refreshes use bootstrapping; there is no decrypt/re-encrypt shortcut. The 32-layer runner streams one indexed layer at a time and fuses each inter-layer refresh into the next input RMSNorm.

## Package map

| Path | Role |
|------|------|
| `run_layer0.py` / `run_model.py` | Stable entrypoints |
| `llama3fhe/application/` | CLI, session, streaming orchestration |
| `llama3fhe/layouts/` | Persistent / private packing contracts |
| `llama3fhe/operators/linear/` | GEMV engine + QKV / W_O / MLP |
| `llama3fhe/operators/attention/` | QK / Softmax / PV / output |
| `llama3fhe/operators/nonlinear/` | Polynomial scalar ops |
| `llama3fhe/operators/norm/` | RMSNorm + fused bootstrap refresh |
| `assets/feature_major.json` | CKKS context defaults |
| `assets/paris128_example.json` | Canonical Paris128 fixture |
| `assets/polynomials/*.json` | Release coefficient packs |

## Scope

**In:** Encrypted runtime, release coeffs, Paris fixture, Layer-0 / E2E entrypoints.  
**Out:** Meta / EasyFHE binaries, poly-fit research trees, paper/bench archives, THOR legacy packing.

## Contributing / license

1. Prefer `python run_layer0.py --audit-only` before larger runs.  
2. Keep changes on the FHE path; no poly-fit trees, weights, wheels, or secrets.  
3. English docs; relative paths / env vars.  
4. License: GPLv3 (`LICENSE`).
