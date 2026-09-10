# Weights

Model files are **not** redistributed. Place a prepared, full-fused QuaRot
checkpoint here, for example:

```text
assets/weights/Meta-Llama-3-8B-QuaRot/
├── config.json
├── export_metadata.json
└── model.safetensors        # or index + shards
```

`export_metadata.json` must declare format `llama3_8b_feature_major_quarot_v1`,
`full_fuse=true`, and ordinary `[out,in]` runtime matrices. Feature-major /
complex diagonal packing is generated online and is not stored in the checkpoint.

Obtain Meta Llama 3 and QuaRot under their licenses. For a weights-free smoke
test:

```bash
python run_layer0.py --weights-source synthetic
```
