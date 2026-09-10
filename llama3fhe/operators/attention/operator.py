from __future__ import annotations

"""End-to-end attention boundary: QKV carriers in, W_O carriers out."""

import math
import os
import time
from dataclasses import dataclass

import numpy as np

from llama3fhe.backend import release_all, release_if_supported
from llama3fhe.backend import synchronize_device as sync_device
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.layouts.linear import LinearCarrierLayout

from .audit import delta_attention_rotations, delta_softmax_rotations
from .config import AttentionOperatorConfig
from .input import (
    complex_token_attention_rotations,
    prepare_attention_inputs_fhe,
)
from .output import pairfm_output_rotations, prepare_w_o_inputs_fhe
from .pv import (
    fold_probability_delta_halves_fhe,
    fold_value_delta_halves_fhe,
    pv_delta_fhe,
)
from .qk import qk_delta_fhe


def _diagnostic_cipher_range(label: str, ciphers, *, crypto_context) -> None:
    if os.environ.get("LLAMA_FHE_STAGE_DIAGNOSTICS", "") != "1":
        return
    values = np.stack(
        [np.asarray(crypto_context.decrypt(cipher)) for cipher in ciphers]
    )
    print(
        f"[attention-diagnostic] stage={label} "
        f"real_max={float(np.max(np.abs(values.real)))} "
        f"imag_max={float(np.max(np.abs(values.imag)))} "
        f"real_mean={float(np.mean(np.abs(values.real)))} "
        f"imag_mean={float(np.mean(np.abs(values.imag)))} "
        f"levels={sorted({int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers})}",
        flush=True,
    )


@dataclass
class AttentionToWOResult:
    """Own the complex feature-major ciphertexts consumed directly by W_O."""

    w_o_input_ciphers: tuple[object, ...]
    stage_seconds: dict[str, float]
    input_scheduler_stats: dict[str, float]
    output_levels: tuple[int, ...]
    level_trace: dict[str, tuple[int, ...]]

    def release(self) -> None:
        release_all(self.w_o_input_ciphers)


@dataclass
class DeltaAttentionAlg2FHEResult:
    output_ciphers: tuple[object, ...]
    stage_seconds: dict[str, float]
    level_trace: dict[str, tuple[int, ...]]

    def release(self) -> None:
        for cipher in self.output_ciphers:
            release_if_supported(cipher)


def attention_fhe_delta_alg2(
    query_ciphers: tuple[object, ...],
    key_delta_ciphers: tuple[object, ...],
    value_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    softmax_config,
    operator_config: AttentionOperatorConfig | None = None,
    layer_idx: int = 0,
    softmax_bootstrap=None,
    token_valid_mask: np.ndarray | None = None,
) -> DeltaAttentionAlg2FHEResult:
    """Ciphertext-only QK → Delta Alg2 → PV with no repack boundary.

    Every schedule decision is read off ``operator_config`` — the same
    object the surrounding operator validates — so this core cannot state
    defaults of its own.
    """

    from .softmax import softmax_alg2_fhe_delta

    operator_config = (
        AttentionOperatorConfig()
        if operator_config is None
        else operator_config
    )
    pv_prefold_after_softmax = operator_config.pv_prefold_after_softmax
    refresh_softmax_exp_inputs = operator_config.refresh_softmax_exp_inputs
    refresh_softmax_exp_outputs = operator_config.refresh_softmax_exp_outputs

    stage_seconds: dict[str, float] = {}
    scores: tuple[object, ...] = ()
    probabilities: tuple[object, ...] = ()
    folded_probabilities: tuple[object, ...] = ()
    folded_values: tuple[object, ...] = ()
    outputs: tuple[object, ...] = ()
    softmax_result = None
    success = False
    level_trace: dict[str, tuple[int, ...]] = {
        name: tuple(
            sorted(
                {
                    int(crypto_context.level_for_cipher(cipher))
                    for cipher in ciphers
                }
            )
        )
        for name, ciphers in (
            ("query", query_ciphers),
            ("key_delta", key_delta_ciphers),
            ("value_delta", value_delta_ciphers),
        )
    }
    try:
        start = time.perf_counter()
        scores = qk_delta_fhe(
            query_ciphers,
            key_delta_ciphers,
            layout=layout,
            crypto_context=crypto_context,
            operator_config=operator_config,
        )
        sync_device(crypto_context.device)
        stage_seconds["qk"] = time.perf_counter() - start
        level_trace["scores"] = tuple(
            sorted(
                {
                    int(crypto_context.level_for_cipher(cipher))
                    for cipher in scores
                }
            )
        )

        _diagnostic_cipher_range(
            "scores", scores, crypto_context=crypto_context
        )

        start = time.perf_counter()
        softmax_result = softmax_alg2_fhe_delta(
            scores,
            layout=layout,
            layer_idx=int(layer_idx),
            config=softmax_config,
            crypto_context=crypto_context,
            operator_config=operator_config,
            score_scale=1.0 / math.sqrt(float(layout.head_dim)),
            bootstrap_operator=softmax_bootstrap,
            refresh_before_exp=bool(refresh_softmax_exp_inputs),
            refresh_after_exp=bool(refresh_softmax_exp_outputs),
            token_valid_mask=token_valid_mask,
        )
        probabilities = softmax_result.probability_ciphers
        softmax_result.probability_ciphers = ()
        level_trace["probabilities"] = tuple(softmax_result.output_levels)
        sync_device(crypto_context.device)
        stage_seconds["softmax"] = time.perf_counter() - start
        stage_seconds.update(
            {
                f"softmax.{name}": float(seconds)
                for name, seconds in softmax_result.stage_seconds.items()
            }
        )
        _diagnostic_cipher_range(
            "probabilities", probabilities, crypto_context=crypto_context
        )
        for cipher in scores:
            release_if_supported(cipher)
        scores = ()

        if bool(pv_prefold_after_softmax):
            start = time.perf_counter()
            folded_probabilities = fold_probability_delta_halves_fhe(
                probabilities,
                layout=layout,
                crypto_context=crypto_context,
            )
            folded_values = fold_value_delta_halves_fhe(
                value_delta_ciphers,
                layout=layout,
                crypto_context=crypto_context,
            )
            sync_device(crypto_context.device)
            stage_seconds["pv_fold_inputs"] = time.perf_counter() - start

        start = time.perf_counter()
        pv_inputs_pre_folded = bool(pv_prefold_after_softmax)
        outputs = pv_delta_fhe(
            folded_probabilities if bool(pv_prefold_after_softmax) else probabilities,
            folded_values if bool(pv_prefold_after_softmax) else value_delta_ciphers,
            layout=layout,
            crypto_context=crypto_context,
            operator_config=operator_config,
            inputs_pre_folded=pv_inputs_pre_folded,
        )
        sync_device(crypto_context.device)
        stage_seconds["pv"] = time.perf_counter() - start
        _diagnostic_cipher_range(
            "output_c",
            outputs,
            crypto_context=crypto_context,
        )
        for cipher in probabilities:
            release_if_supported(cipher)
        probabilities = ()

        success = True
        return DeltaAttentionAlg2FHEResult(
            output_ciphers=outputs,
            stage_seconds={name: float(seconds) for name, seconds in stage_seconds.items()},
            level_trace={
                **level_trace,
                "output_c": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in outputs
                        }
                    )
                ),
            },
        )
    finally:
        if softmax_result is not None:
            softmax_result.release()
        for cipher in scores + probabilities + folded_probabilities + folded_values:
            release_if_supported(cipher)
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)


def attention_to_w_o_rotations(
    carrier_layout: LinearCarrierLayout,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    operator_config: AttentionOperatorConfig | None = None,
    log_n: int = 16,
) -> tuple[int, ...]:
    """Exact rotation-key union excluding QKV and W_O linear projections."""

    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    rotations = set(
        complex_token_attention_rotations(
            carrier_layout,
            attention_layout,
            sparse_max_baby_offsets=int(
                operator_config.sparse_max_baby_offsets
            ),
            shear_baby_steps=int(operator_config.shear_baby_steps),
            log_n=int(log_n),
        )
    )
    rotations.update(
        delta_attention_rotations(
            attention_layout,
            include_qk_feature_fold=bool(
                operator_config.qk_complex_feature_folding
            ),
            qk_key_features_pre_folded=bool(
                operator_config.qk_key_features_pre_folded
            ),
            qk_query_features_pre_folded=bool(
                operator_config.qk_query_features_pre_folded
            ),
            include_pv_delta_fold=bool(
                operator_config.pv_complex_delta_folding
            ),
            pv_probability_pre_folded=bool(
                operator_config.pv_prefold_after_softmax
            ),
            pv_value_pre_folded=bool(
                operator_config.pv_prefold_after_softmax
            ),
            log_n=int(log_n),
        )
    )
    rotations.update(delta_softmax_rotations(attention_layout))
    rotations.update(
        pairfm_output_rotations(
            feature_layout,
            attention_layout,
            first_baby_steps=int(operator_config.pairfm_first_baby_steps),
            block_baby_steps=int(operator_config.pairfm_block_baby_steps),
            log_n=int(log_n),
        )
    )
    return tuple(sorted(int(value) for value in rotations if int(value)))


def attention_fhe_from_qkv_carriers(
    carrier_ciphers: tuple[object, ...] | list[object],
    *,
    carrier_layout: LinearCarrierLayout,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    softmax_config,
    model_config: Llama3CKKSConfig | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    layer_idx: int = 0,
    softmax_bootstrap=None,
    expected_query: np.ndarray | None = None,
    expected_key: np.ndarray | None = None,
    expected_value: np.ndarray | None = None,
    expected_output_c: np.ndarray | None = None,
    token_valid_mask: np.ndarray | None = None,
    function_prefix: str = "attention",
) -> AttentionToWOResult:
    """Run RoPE/layout, QK, Softmax, PV, and PairFM output conversion.

    ``carrier_ciphers`` remain caller-owned. The returned eight ciphertexts
    are the exact complex token-pair feature-major boundary consumed by W_O.
    """

    carrier_ciphers = tuple(carrier_ciphers)
    model_config = Llama3CKKSConfig() if model_config is None else model_config
    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    expected_inputs = (expected_query, expected_key, expected_value)
    verify_inputs = any(value is not None for value in expected_inputs)
    if verify_inputs and not all(value is not None for value in expected_inputs):
        raise ValueError(
            "expected_query/key/value must either all be supplied or all omitted."
        )

    attention_inputs = None
    attention_core = None
    output = None
    transferred: tuple[object, ...] = ()
    success = False
    stages: dict[str, float] = {}
    try:
        start = time.perf_counter()
        attention_inputs = prepare_attention_inputs_fhe(
            carrier_ciphers,
            carrier_layout=carrier_layout,
            attention_layout=attention_layout,
            crypto_context=crypto_context,
            model_config=model_config,
            operator_config=operator_config,
            verify=bool(verify_inputs),
            expected_query=expected_query,
            expected_key=expected_key,
            expected_value=expected_value,
            function_prefix=f"{function_prefix}.input",
        )
        stages["input"] = time.perf_counter() - start
        stages.update(
            {
                f"input.{name}": float(seconds)
                for name, seconds in attention_inputs.stage_seconds.items()
            }
        )
        _diagnostic_cipher_range(
            "query_c", attention_inputs.query_ciphers,
            crypto_context=crypto_context,
        )
        _diagnostic_cipher_range(
            "key_delta", attention_inputs.key_ciphers,
            crypto_context=crypto_context,
        )
        _diagnostic_cipher_range(
            "value_delta", attention_inputs.value_ciphers,
            crypto_context=crypto_context,
        )

        start = time.perf_counter()
        attention_core = attention_fhe_delta_alg2(
            attention_inputs.query_ciphers,
            attention_inputs.key_ciphers,
            attention_inputs.value_ciphers,
            layout=attention_layout,
            crypto_context=crypto_context,
            softmax_config=softmax_config,
            operator_config=operator_config,
            layer_idx=int(layer_idx),
            softmax_bootstrap=softmax_bootstrap,
            token_valid_mask=token_valid_mask,
        )
        stages["qk_softmax_pv"] = time.perf_counter() - start
        stages.update(
            {
                f"qk_softmax_pv.{name}": float(seconds)
                for name, seconds in attention_core.stage_seconds.items()
            }
        )
        start = time.perf_counter()
        output = prepare_w_o_inputs_fhe(
            attention_core.output_ciphers,
            feature_layout=feature_layout,
            attention_layout=attention_layout,
            crypto_context=crypto_context,
            model_config=model_config,
            operator_config=operator_config,
            bootstrap_operator=softmax_bootstrap,
            verify=expected_output_c is not None,
            expected_output_c=expected_output_c,
            function_prefix=f"{function_prefix}.output",
        )
        stages["output"] = time.perf_counter() - start
        stages.update(
            {
                f"output.{name}": float(seconds)
                for name, seconds in output.stage_seconds.items()
            }
        )
        _diagnostic_cipher_range(
            "pairfm_w_o_input", output.carrier_ciphers,
            crypto_context=crypto_context,
        )

        release_all(output.real_ciphers)
        output.real_ciphers = ()
        transferred = output.carrier_ciphers
        output.carrier_ciphers = ()
        result = AttentionToWOResult(
            w_o_input_ciphers=transferred,
            stage_seconds={str(key): float(value) for key, value in stages.items()},
            input_scheduler_stats={
                str(key): float(value)
                for key, value in attention_inputs.scheduler_stats.items()
            },
            output_levels=tuple(
                sorted(
                    {
                        int(crypto_context.level_for_cipher(cipher))
                        for cipher in transferred
                    }
                )
            ),
            level_trace={
                "linear_carriers": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in carrier_ciphers
                        }
                    )
                ),
                "query": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in attention_inputs.query_ciphers
                        }
                    )
                ),
                "key_delta": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in attention_inputs.key_ciphers
                        }
                    )
                ),
                "value_delta": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in attention_inputs.value_ciphers
                        }
                    )
                ),
                **attention_core.level_trace,
                "w_o_input": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in transferred
                        }
                    )
                ),
            },
        )
        transferred = ()
        success = True
        return result
    finally:
        if output is not None:
            output.release()
        if attention_core is not None:
            attention_core.release()
        if attention_inputs is not None:
            attention_inputs.release()
        if not success:
            release_all(transferred)


__all__ = [
    "AttentionToWOResult",
    "attention_fhe_from_qkv_carriers",
    "attention_to_w_o_rotations",
]
