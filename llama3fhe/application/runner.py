from __future__ import annotations

"""CLI orchestration for one-layer and streamed full-model execution."""

import argparse
import gc
import sys
import time
from pathlib import Path

import easyfhe
import numpy as np

from .. import Llama3CKKSConfig
from ..approx.chebyshev import chebyshev_ps_mul_depth
from ..approx.layer import poly_transformer_layer_numpy
from ..backend import (
    EasyFHEContext,
    cuda_memory_snapshot,
    plaintext_cache_snapshot,
    release_all,
)
from ..graph import operation_audit
from ..layouts import AttentionPairLayout, FeatureMajorPrefillLayout
from ..model import LayerWeights, Llama3Model, prepare_bootstrap_operator
from ..operators.nonlinear import BootstrapOperator
from .approx_config import ApproximationConfig
from .args import (
    PROJECT_ROOT,
    _fixture_expected_token_id,
    build_parser,
    resolve_poly_paths,
)
from .checkpoint import (
    CheckpointMetadata,
    WeightLoader,
    validate_runtime_checkpoint,
)
from .client import (
    begin_cuda_allocator_sample,
    decrypt_hidden,
    end_cuda_allocator_sample,
    final_logits,
    load_input_ids,
    median,
    nested_timing_sum,
    stage_medians,
    top_k_logits,
    write_json_report,
)
from .inputs import (
    _load_prompt_ids,
    _synthetic_inputs,
)
from .layer_stream import DoubleBufferedLayerStream
from .report import audit_payload, schedule_report
from .session import (
    _create_context,
    _encrypt_hidden,
    _release_transient_device_memory,
)
from .settings import layer_schedule, resolve_settings


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _raise_on_failed_expectation(payload: dict[str, object]) -> None:
    expectation = payload.get("expectation")
    if isinstance(expectation, dict) and expectation.get("passed") is False:
        raise SystemExit(
            "next-token expectation failed: "
            f"expected {expectation.get('expected_token_id')}, "
            f"got {expectation.get('actual_token_id')}"
        )


def _finite_abs_stats(values: np.ndarray) -> tuple[bool, float | None, float | None]:
    finite = bool(np.all(np.isfinite(values)))
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        return finite, None, None
    return (
        finite,
        float(np.max(np.abs(finite_values))),
        float(np.mean(np.abs(finite_values))),
    )


def _diff_stats(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    finite, max_abs, mean_abs = _finite_abs_stats(difference)
    return {"finite": finite, "max_abs": max_abs, "mean_abs": mean_abs}


def _fmt_metric(value: object) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4g}"


def _layer_value_diagnostic(ciphers, *, context, layout) -> dict[str, object]:
    values = decrypt_hidden(ciphers, crypto_context=context, layout=layout)
    finite, max_abs, mean_abs = _finite_abs_stats(values)
    return {
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "values": values,
    }


def _aggregate_layer_stages(
    per_layer: list[dict[str, object]],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in per_layer:
        for name, value in dict(row.get("stage_seconds", {})).items():
            totals[str(name)] = totals.get(str(name), 0.0) + float(value)
    return totals


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run a requested layer prefix through one common encrypted model path."""

    args = resolve_poly_paths(args)
    application_start = time.perf_counter()
    feature_layout = FeatureMajorPrefillLayout()
    attention_layout = AttentionPairLayout()

    config = Llama3CKKSConfig.from_json(args.config)
    if args.depth is not None:
        config.simulator.maxLevelsRemaining = int(args.depth)
    if args.device is not None:
        config.simulator.device = str(args.device)

    # Every schedule decision is resolved and validated up front; the stages
    # below read them and never re-derive one.
    settings = resolve_settings(args, config)
    schedule = layer_schedule(args, settings)
    num_layers = settings.num_layers
    model_scope = settings.model_scope
    warmup_scope = settings.warmup_scope
    input_level = settings.input_level

    approx_config = ApproximationConfig(
        rmsnorm_path=args.rms_coeffs,
        silu_path=args.silu_coeffs,
        softmax_path=args.softmax_coeffs,
    )
    approx_config.validate_layers(num_layers)
    model = Llama3Model(
        config=config,
        num_layers=num_layers,
        feature_layout=feature_layout,
        attention_layout=attention_layout,
    )
    def selected_model_rotations() -> tuple[int, ...]:
        # Key planning must include the deep budget's CtoS/StoC union when
        # the deep main Softmax bootstrap is enabled; the schedule carries
        # every other decision the rotation union depends on.
        return model.required_rotations(
            schedule,
            bootstrap_deep_level_budget=(
                (3, 3) if schedule.softmax_main_deep_bootstrap else None
            ),
        )

    first = approx_config.for_layer(0)
    audit = operation_audit(
        feature_layout,
        attention_layout,
        schedule=schedule,
        input_rmsnorm_degree=len(first.input_norm["coefficients"]) - 1,
        post_attention_rmsnorm_degree=(
            len(first.post_attention_norm["coefficients"]) - 1
        ),
        silu_degree=len(first.silu["coefficients"]) - 1,
        silu_multiplicative_depth=chebyshev_ps_mul_depth(
            first.silu["coefficients"]
        ),
    )
    if bool(args.audit_only):
        return audit_payload(
            args,
            settings,
            schedule,
            rotation_key_count=len(selected_model_rotations()),
            canonical_cipher_count=int(feature_layout.cipher_count),
            degree_summaries={
                "first_layer": approx_config.degree_summary(0),
                "last_layer": approx_config.degree_summary(num_layers - 1),
            },
            audit=audit,
        )

    loader: WeightLoader | None = None
    synthetic_weights: LayerWeights | None = None
    metadata: CheckpointMetadata | None = None
    export_metadata: dict[str, object] | None = None
    tokenizer = None
    raw_token_count: int | None = None
    attention_token_valid_mask = np.ones(
        int(feature_layout.seq_len), dtype=bool
    )
    input_ciphers: tuple[object, ...] = ()
    context: EasyFHEContext | None = None
    bootstrap_operator: BootstrapOperator | None = None
    bootstrap_prepare_seconds = 0.0
    expected_token_id = (
        None if args.expect_token_id is None else int(args.expect_token_id)
    )
    setup_device_memory: dict[str, dict[str, object]] = {}
    result = None

    def record_setup_device_memory(label: str, context) -> None:
        if not bool(args.profile_device_memory):
            return
        snapshot = cuda_memory_snapshot(context.device)
        snapshot["application_elapsed_seconds"] = float(
            time.perf_counter() - application_start
        )
        setup_device_memory[str(label)] = snapshot

    try:
        if str(args.weights_source) == "real":
            if args.model_dir is None:
                raise ValueError("real weights require --model-dir.")
            input_ids_file = args.input_ids_file
            default_input_selected = False
            if input_ids_file is None and args.prompt is None:
                input_ids_file = args.default_input_ids_file
                default_input_selected = input_ids_file is not None
            if input_ids_file is None and args.prompt is None:
                raise ValueError("supply --input-ids-file or --prompt.")
            export_metadata = validate_runtime_checkpoint(args.model_dir)
            metadata = CheckpointMetadata.from_json(args.model_dir / "config.json")
            metadata.validate_llama3_8b(num_layers=num_layers)
            if input_ids_file is not None:
                input_ids = load_input_ids(
                    input_ids_file, seq_len=int(feature_layout.seq_len)
                )
                input_source = _portable_path(input_ids_file)
                prediction_token_index = int(feature_layout.seq_len) - 1
                fixture_expected_token_id = _fixture_expected_token_id(
                    input_ids_file
                )
                if (
                    expected_token_id is None
                    and fixture_expected_token_id is not None
                    and num_layers == 32
                ):
                    expected_token_id = int(fixture_expected_token_id)
            else:
                input_ids, tokenizer, raw_token_count = _load_prompt_ids(
                    args.prompt,
                    model_dir=args.model_dir,
                    metadata=metadata,
                    seq_len=int(feature_layout.seq_len),
                    pad_side=str(args.pad_side),
                    add_special_tokens=not bool(args.no_add_special_tokens),
                )
                input_source = "prompt"
                prediction_token_index = (
                    int(feature_layout.seq_len) - 1
                    if str(args.pad_side) == "left"
                    else min(raw_token_count, int(feature_layout.seq_len)) - 1
                )
                valid_count = min(
                    int(raw_token_count), int(feature_layout.seq_len)
                )
                attention_token_valid_mask.fill(False)
                if valid_count and str(args.pad_side) == "left":
                    attention_token_valid_mask[-valid_count:] = True
                elif valid_count:
                    attention_token_valid_mask[:valid_count] = True
            loader = WeightLoader(args.model_dir, tensor_dtype=np.float32)
            embedding = loader.get("model.embed_tokens.weight", out_dtype=np.float32)
            if embedding.ndim != 2 or embedding.shape[1] != int(feature_layout.hidden_dim):
                raise ValueError(
                    "model.embed_tokens.weight must have shape "
                    f"[vocab, {feature_layout.hidden_dim}], got {embedding.shape}."
                )
            if np.any(input_ids < 0) or np.any(input_ids >= embedding.shape[0]):
                raise ValueError("input IDs contain values outside the checkpoint vocabulary.")
            hidden = np.asarray(embedding[input_ids], dtype=np.float32)
            del embedding
            rope_theta = float(metadata.rope_theta)
            input_report: dict[str, object] = {
                "source": input_source,
                "raw_token_count": raw_token_count,
                "input_ids_head": input_ids[:8].tolist(),
                "input_ids_tail": input_ids[-8:].tolist(),
                "prediction_token_index": int(prediction_token_index),
                "attention_valid_token_count": int(
                    np.count_nonzero(attention_token_valid_mask)
                ),
                "padding_attention_policy": (
                    "causal_valid_keys_with_invalid_query_self_sentinel"
                ),
                "canonical_default_fixture": bool(default_input_selected),
                "expected_next_token_id": expected_token_id,
            }
        else:
            if args.model_dir is not None or args.input_ids_file is not None or args.prompt:
                raise ValueError(
                    "synthetic execution does not consume checkpoint or token inputs."
                )
            hidden, synthetic_weights = _synthetic_inputs(
                args.seed,
                layout=feature_layout,
                attention_layout=attention_layout,
            )
            rope_theta = 500000.0
            prediction_token_index = int(feature_layout.seq_len) - 1
            input_report = {"source": "synthetic", "seed": int(args.seed)}

        prepare_seconds = time.perf_counter() - application_start
        rotation_started = time.perf_counter()
        rotations = selected_model_rotations()
        rotation_plan_seconds = time.perf_counter() - rotation_started
        context = _create_context(config, rotations)
        context.configure_plaintext_cache(
            limit_bytes=int(float(args.plaintext_cache_gb) * (1024**3)),
            max_source_bytes=int(
                float(args.plaintext_cache_max_source_mb) * (1024**2)
            ),
        )
        record_setup_device_memory("context_and_keys", context)
        encrypt_started = time.perf_counter()
        input_ciphers = _encrypt_hidden(
            hidden,
            context=context,
            layout=feature_layout,
            input_level=input_level,
        )
        encrypt_seconds = time.perf_counter() - encrypt_started
        record_setup_device_memory("input_encrypted", context)
        bootstrap_started = time.perf_counter()
        bootstrap_operator = prepare_bootstrap_operator(context, schedule)
        bootstrap_prepare_seconds = time.perf_counter() - bootstrap_started
        record_setup_device_memory("bootstrap_prepared", context)

        def layer_factory(layer_count: int = num_layers) -> DoubleBufferedLayerStream:
            return DoubleBufferedLayerStream(
                loader=loader,
                synthetic_weights=synthetic_weights,
                approx_config=approx_config,
                config=config,
                feature_layout=feature_layout,
                attention_layout=attention_layout,
                num_layers=int(layer_count),
                rope_theta=rope_theta,
            )

        poly_hidden = np.asarray(hidden, dtype=np.float32)
        prev_fhe = np.asarray(hidden, dtype=np.float32)

        def layer_diagnostic(index, ciphers, layer):
            nonlocal poly_hidden, prev_fhe
            decrypted = _layer_value_diagnostic(
                ciphers, context=context, layout=feature_layout
            )
            fhe_values = decrypted.pop("values")
            row: dict[str, object] = {
                "finite": decrypted["finite"],
                "fhe_max_abs": decrypted["max_abs"],
                "fhe_mean_abs": decrypted["mean_abs"],
                "fhe_vs_poly_max_abs": None,
                "fhe_vs_poly_mean_abs": None,
                "fhe_vs_poly_local_max_abs": None,
                "fhe_vs_poly_local_mean_abs": None,
                "silu_gate_max_abs": None,
                "silu_oob_frac": None,
                "silu_fit_interval": None,
                "rmsn1_s_max": None,
                "rmsn1_oob_frac": None,
                "rmsn2_s_max": None,
                "rmsn2_oob_frac": None,
                "poly_error": None,
            }
            try:
                cascade = poly_transformer_layer_numpy(
                    poly_hidden,
                    weights=layer.weights,
                    approximations=layer.approximations,
                    attention_layout=attention_layout,
                    rope_theta=float(layer.rope_theta),
                )
                cascade_diff = _diff_stats(fhe_values, cascade.hidden)
                row["fhe_vs_poly_max_abs"] = cascade_diff["max_abs"]
                row["fhe_vs_poly_mean_abs"] = cascade_diff["mean_abs"]
                row["fhe_vs_poly_finite"] = cascade_diff["finite"]
                poly_hidden = cascade.hidden
            except Exception as exc:
                row["poly_error"] = f"cascade: {exc}"

            try:
                local = poly_transformer_layer_numpy(
                    prev_fhe,
                    weights=layer.weights,
                    approximations=layer.approximations,
                    attention_layout=attention_layout,
                    rope_theta=float(layer.rope_theta),
                )
                local_diff = _diff_stats(fhe_values, local.hidden)
                row["fhe_vs_poly_local_max_abs"] = local_diff["max_abs"]
                row["fhe_vs_poly_local_mean_abs"] = local_diff["mean_abs"]
                row["silu_gate_max_abs"] = local.silu_gate_max_abs
                row["silu_oob_frac"] = local.silu_oob_frac
                row["silu_fit_interval"] = [
                    local.silu_fit_lo,
                    local.silu_fit_hi,
                ]
                row["rmsn1_s_max"] = local.rmsn1_s_max
                row["rmsn1_oob_frac"] = local.rmsn1_oob_frac
                row["rmsn1_fit"] = list(local.rmsn1_fit) if local.rmsn1_fit else None
                row["rmsn2_s_max"] = local.rmsn2_s_max
                row["rmsn2_oob_frac"] = local.rmsn2_oob_frac
                row["rmsn2_fit"] = list(local.rmsn2_fit) if local.rmsn2_fit else None
            except Exception as exc:
                previous = row["poly_error"]
                detail = f"local: {exc}"
                row["poly_error"] = (
                    detail if not previous else f"{previous}; {detail}"
                )

            prev_fhe = np.asarray(fhe_values, dtype=np.float32)
            print(
                "[diagnostic] layer "
                f"{int(index) + 1}/{num_layers} finite={row['finite']} "
                f"fhe_max_abs={_fmt_metric(row['fhe_max_abs'])} "
                f"fhe_mean_abs={_fmt_metric(row['fhe_mean_abs'])} "
                f"fhe_vs_poly_max_abs={_fmt_metric(row['fhe_vs_poly_max_abs'])} "
                f"local_max_abs={_fmt_metric(row['fhe_vs_poly_local_max_abs'])} "
                f"silu_gate={_fmt_metric(row['silu_gate_max_abs'])} "
                f"silu_oob={_fmt_metric(row['silu_oob_frac'])} "
                f"rmsn1_s={_fmt_metric(row['rmsn1_s_max'])} "
                f"rmsn1_oob={_fmt_metric(row['rmsn1_oob_frac'])} "
                f"rmsn2_s={_fmt_metric(row['rmsn2_s_max'])} "
                f"rmsn2_oob={_fmt_metric(row['rmsn2_oob_frac'])}",
                file=sys.stderr,
                flush=True,
            )
            if row["poly_error"]:
                print(
                    f"[diagnostic] layer {int(index) + 1}/{num_layers} "
                    f"poly_error={row['poly_error']}",
                    file=sys.stderr,
                    flush=True,
                )
            return row

        def run_one(
            *,
            evaluation_model=model,
            evaluation_layers: int = num_layers,
            refresh_first_layer: bool | None = None,
            progress_prefix: str = "model",
            diagnostics: bool = True,
        ):
            nonlocal poly_hidden, prev_fhe
            poly_hidden = np.asarray(hidden, dtype=np.float32)
            prev_fhe = np.asarray(hidden, dtype=np.float32)
            baseline_memory = begin_cuda_allocator_sample(context.device)
            sample_result = None
            layer_stream = layer_factory(int(evaluation_layers))
            cache_before = plaintext_cache_snapshot(context)
            try:
                sample_result = evaluation_model.forward(
                    hidden,
                    crypto_context=context,
                    input_ciphers=input_ciphers,
                    layers=layer_stream,
                    bootstrap_operator=bootstrap_operator,
                    schedule=schedule,
                    refresh_first_layer=(
                        bool(args.steady_state_layer)
                        if refresh_first_layer is None
                        else bool(refresh_first_layer)
                    ),
                    attention_token_valid_mask=attention_token_valid_mask,
                    diagnostic_callback=(
                        layer_diagnostic
                        if diagnostics and bool(args.diagnostic_decrypt_layers)
                        else None
                    ),
                    progress_callback=lambda row: print(
                        (
                            f"[{progress_prefix}] layer "
                            f"{int(row['layer']) + 1}/{int(evaluation_layers)} "
                            f"wall={float(row['wall_seconds']):.3f}s "
                            f"levels={list(row['output_levels'])}"
                            + (
                                " mlp_in={after}/{required} "
                                "emergency_bs={emergency}".format(
                                    after=mlp_drop.get("after_min_limbs"),
                                    required=mlp_drop.get(
                                        "required_input_limbs"
                                    ),
                                    emergency=mlp_drop.get(
                                        "overridden_by_emergency_bootstrap"
                                    ),
                                )
                                if isinstance(
                                    (
                                        mlp_drop := dict(
                                            row.get("nested_stage_seconds", {})
                                        ).get("mlp_input_modulus_drop")
                                    ),
                                    dict,
                                )
                                else ""
                            )
                        ),
                        file=sys.stderr,
                        flush=True,
                    ),
                )
                if str(context.device).startswith("cuda"):
                    easyfhe.cuda.synchronize()
                peak_memory = end_cuda_allocator_sample(context.device)
                cache_after = plaintext_cache_snapshot(context)
                stages = _aggregate_layer_stages(sample_result.per_layer)
                encode_seconds = nested_timing_sum(
                    {"layers": sample_result.per_layer},
                    suffixes=("pack_encode", "weight_upload"),
                )
                sample = {
                    "wall_seconds": float(sample_result.wall_seconds),
                    "reusable_pack_encode_seconds": float(encode_seconds),
                    "compute_excluding_prepare_seconds": max(
                        0.0,
                        float(sample_result.wall_seconds)
                        - float(encode_seconds),
                    ),
                    "stage_seconds": stages,
                    "output_levels": list(sample_result.per_layer[-1]["output_levels"]),
                    "cuda_allocator_peak_bytes": peak_memory,
                    "cuda_allocator_incremental_peak_bytes": (
                        None
                        if peak_memory is None or baseline_memory is None
                        else max(0, int(peak_memory - baseline_memory))
                    ),
                    "plaintext_cache": {
                        "before": cache_before,
                        "after": cache_after,
                        "delta": {
                            name: int(cache_after[name])
                            - int(cache_before[name])
                            for name in (
                                "entries",
                                "bytes",
                                "hits",
                                "misses",
                                "skips",
                            )
                        },
                    },
                    "raw_weight_prefetch": layer_stream.stats(),
                }
                if bool(args.profile_operator_breakdown):
                    sample["per_layer"] = list(sample_result.per_layer)
                transferred, sample_result = sample_result, None
                return transferred, sample
            finally:
                layer_stream.close()
                if sample_result is not None:
                    sample_result.release()

        warmup_started = time.perf_counter()
        warmup_layer_count = num_layers
        warmup_model = model
        if warmup_scope == "steady-layer":
            warmup_layer_count = 1
            warmup_model = Llama3Model(
                config=config,
                num_layers=1,
                feature_layout=feature_layout,
                attention_layout=attention_layout,
            )
        for _ in range(int(args.warmup)):
            warmup_result, _ = run_one(
                evaluation_model=warmup_model,
                evaluation_layers=int(warmup_layer_count),
                refresh_first_layer=(
                    True if warmup_scope == "steady-layer" else None
                ),
                progress_prefix="warmup",
                diagnostics=False,
            )
            warmup_result.release()
            if warmup_scope == "model":
                _release_transient_device_memory(context.device)
            else:
                gc.collect()
                if str(context.device).startswith("cuda"):
                    easyfhe.get_device_module(
                        str(context.device)
                    ).synchronize()
        warmup_seconds = (
            0.0
            if int(args.warmup) == 0
            else float(time.perf_counter() - warmup_started)
        )
        samples: list[dict[str, object]] = []
        for sample_index in range(int(args.repetitions)):
            sample_result, sample = run_one()
            samples.append(sample)
            if sample_index + 1 == int(args.repetitions):
                result = sample_result
            else:
                sample_result.release()
                _release_transient_device_memory(context.device)

        decode_started = time.perf_counter()
        decoded = decrypt_hidden(
            result.ciphers, crypto_context=context, layout=feature_layout
        )
        if args.output_hidden_npy is not None:
            args.output_hidden_npy.parent.mkdir(
                parents=True, exist_ok=True
            )
            np.save(args.output_hidden_npy, decoded)
        finite = bool(np.all(np.isfinite(decoded)))
        if not finite:
            raise RuntimeError(f"{num_layers}-layer model produced non-finite hidden states.")
        decode_seconds = time.perf_counter() - decode_started
        peak_values = [
            int(sample["cuda_allocator_peak_bytes"])
            for sample in samples
            if sample["cuda_allocator_peak_bytes"] is not None
        ]
        incremental_peak_values = [
            int(sample["cuda_allocator_incremental_peak_bytes"])
            for sample in samples
            if sample["cuda_allocator_incremental_peak_bytes"] is not None
        ]
        payload: dict[str, object] = {
            "backend": "feature_major",
            "mode": "fhe",
            "model_scope": model_scope,
            "num_layers": num_layers,
            "weights_source": str(args.weights_source),
            "input": input_report,
            "output_shape": list(decoded.shape),
            "finite": finite,
            "max_abs": float(np.max(np.abs(decoded))),
            "mean_abs": float(np.mean(np.abs(decoded))),
            "output_levels": list(samples[-1]["output_levels"]),
            "performance": {
                "warmup": int(args.warmup),
                "warmup_scope": str(warmup_scope),
                "warmup_layers_per_repetition": int(warmup_layer_count),
                "warmup_seconds": float(warmup_seconds),
                "repetitions": int(args.repetitions),
                "input_and_checkpoint_prepare_seconds": float(prepare_seconds),
                "rotation_plan_seconds": float(rotation_plan_seconds),
                "context_keygen_seconds": float(context.keygen_wall_time_s),
                "input_pack_encrypt_seconds": float(encrypt_seconds),
                "application_bootstrap_prepare_seconds": float(
                    bootstrap_prepare_seconds
                ),
                "final_decrypt_seconds": float(decode_seconds),
                "total_seconds": float(time.perf_counter() - application_start),
                "median_wall_seconds": median(sample["wall_seconds"] for sample in samples),
                "median_reusable_pack_encode_seconds": median(
                    sample["reusable_pack_encode_seconds"] for sample in samples
                ),
                "median_compute_excluding_prepare_seconds": median(
                    sample["compute_excluding_prepare_seconds"]
                    for sample in samples
                ),
                "timing_definition": {
                    "wall_seconds": (
                        "model evaluation critical path including plaintext "
                        "pack/encode and weight upload; the bootstrap program "
                        "is prepared by the application, outside this wall"
                    ),
                    "reusable_pack_encode_seconds": (
                        "plaintext pack/encode and weight upload subset"
                    ),
                    "compute_excluding_prepare_seconds": (
                        "wall after subtracting plaintext prepare"
                    ),
                },
                "median_stage_seconds": stage_medians(samples),
                "cuda_allocator": "EasyFHE/PyTorch CUDA allocator",
                "cuda_allocator_peak_bytes": None if not peak_values else max(peak_values),
                "cuda_allocator_incremental_peak_bytes": (
                    None if not incremental_peak_values else max(incremental_peak_values)
                ),
                "device_memory_setup": setup_device_memory,
                "plaintext_cache": plaintext_cache_snapshot(context),
                "samples": samples,
            },
            "runtime": {
                **schedule_report(args, settings, schedule),
                "config": _portable_path(args.config),
                "device": str(config.simulator.device),
                "plaintext_cache_limit_bytes": int(
                    float(args.plaintext_cache_gb) * (1 << 30)
                ),
                "plaintext_cache_max_source_bytes": int(
                    float(args.plaintext_cache_max_source_mb) * (1 << 20)
                ),
                "raw_weight_prefetch": "one_layer_host_double_buffer",
                "logical_bootstrap_calls": int(result.bootstrap_calls),
                "native_bootstrap_calls": int(result.native_bootstrap_calls),
                "requested_rotation_count": len(selected_model_rotations()),
            },
            "approximations": dict(approx_config.paths),
            "per_layer": result.per_layer,
            "audit": audit,
        }
        if loader is not None and metadata is not None:
            logits_started = time.perf_counter()
            logits = final_logits(
                decoded,
                loader=loader,
                metadata=metadata,
                token_index=prediction_token_index,
            )
            top = top_k_logits(logits, int(args.top_k))
            if tokenizer is not None:
                for row in top:
                    row["text"] = tokenizer.decode(
                        [int(row["token_id"])], skip_special_tokens=False
                    )
            payload["next_token"], payload["top_k"] = top[0], top
            if expected_token_id is not None:
                actual_token_id = int(top[0]["token_id"])
                payload["expectation"] = {
                    "expected_token_id": int(expected_token_id),
                    "actual_token_id": actual_token_id,
                    "passed": actual_token_id == expected_token_id,
                }
            payload["performance"]["final_logits_seconds"] = float(
                time.perf_counter() - logits_started
            )
            payload["checkpoint"] = {
                "rope_theta": float(metadata.rope_theta),
                "rms_norm_eps": float(metadata.rms_norm_eps),
                "format": str(export_metadata["format"]),
                "full_fuse": True,
            }
            if num_layers == 1:
                payload["diagnostic_note"] = (
                    "One-layer output is a system diagnostic, not a full-model "
                    "next-token prediction."
                )
        return payload
    finally:
        try:
            if result is not None:
                result.release()
        finally:
            try:
                release_all(input_ciphers)
            finally:
                try:
                    if bootstrap_operator is not None:
                        bootstrap_operator.release()
                finally:
                    try:
                        if context is not None:
                            context.clear_plaintext_cache()
                    finally:
                        if loader is not None:
                            loader.close()


def main(
    argv: list[str] | None = None,
    *,
    layer0_preset: bool = False,
) -> None:
    args = build_parser(layer0_preset=layer0_preset).parse_args(argv)
    payload = run(args)
    write_json_report(payload, args.output_json)
    _raise_on_failed_expectation(payload)


__all__ = ["build_parser", "main", "run"]
