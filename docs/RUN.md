# Run

Activate the env first (`source ./scripts/activate_env.sh`).

## Layer-0 checks

```bash
# Import / layout / schedule audit (no GPU FHE context required)
python run_layer0.py --audit-only

# Synthetic encrypted Layer-0 (full shape, no weights)
python run_layer0.py --weights-source synthetic
```

Expect on A100-class GPU: roughly one-layer wall on the order of tens of seconds (schedule-dependent).

## Paris128 end-to-end (32 layers)

Canonical fixture: `assets/paris128_example.json`  
Expected next token: **` Paris`** (`12366`).

```bash
python run_model.py \
  --model-dir assets/weights/Meta-Llama-3-8B-QuaRot \
  --num-layers 32
```

### Reference (A100 80GB)

| Metric | Value |
|--------|------:|
| Encrypted 32-layer prefill | **~511 s** |
| Peak GPU (nvidia-smi) | **~65 GiB** (plan ~70 GiB free) |
| Golden token | **12366** (` Paris`) |

Defaults include a one-layer steady warmup; use `--warmup 0` to disable.  
Override the fixture with `--input-ids-file` or `--prompt` (requires `transformers`).

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Missing EasyFHE | `EASYFHE_SOURCE_DIR` / `EASYFHE_WHEEL` + `./scripts/setup_env.sh` |
| Missing model | Prepare QuaRot checkpoint under `assets/weights/` |
| OOM | Need ~70 GiB free; try Layer-0 first |
| Wrong token | Check `export_metadata.json` (`full_fuse=true`) and fixture IDs |
| Import errors | `source ./scripts/activate_env.sh`; Python 3.12 |

Design notes: [DESIGN.md](DESIGN.md).
