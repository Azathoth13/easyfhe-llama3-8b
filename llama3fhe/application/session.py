"""Bringing up the CKKS session: context, keys, bootstrap program, encryption.

These are the expensive, once-per-run steps that the per-layer schedule then
takes for granted. They are grouped because they share one invariant: the
rotation-key set generated here must be a superset of every rotation any layer
will ask for, so the key plan and the layer schedule can never be chosen
independently.
"""

from __future__ import annotations

import gc
import time

import easyfhe
import numpy as np
from easyfhe import fhe

from .. import Llama3CKKSConfig
from ..backend import EasyFHEContext, release_all
from ..layouts import FeatureMajorPrefillLayout


def _create_context(
    config: Llama3CKKSConfig,
    rotations: tuple[int, ...],
) -> EasyFHEContext:
    """Create the single explicit EasyFHE context shared by all layers."""

    started = time.perf_counter()
    client, native = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=int(config.simulator.maxLevelsRemaining),
            log_n=int(config.simulator.logN),
            dnum=int(config.simulator.dnum),
            dcrt_bits=int(config.simulator.dcrtBits),
            first_mod=int(config.simulator.firstMod),
            secret_key_dist=str(config.simulator.secretKeyDist),
            scale_mode="fixed",
            rescale_policy=config.simulator.rescale_policy(),
            rotations=rotations,
            auto_load_keys=True,
        ),
        device=str(config.simulator.device),
    )
    return EasyFHEContext(
        tensor=easyfhe,
        fhe=fhe,
        client=client,
        context=native,
        max_slots=int(native.max_slots),
        keygen_wall_time_s=float(time.perf_counter() - started),
    )


def _encrypt_hidden(
    hidden: np.ndarray,
    *,
    context: EasyFHEContext,
    layout: FeatureMajorPrefillLayout,
    input_level: int,
) -> tuple[object, ...]:
    """Pack and encrypt the only clear hidden-state boundary."""

    encrypted: list[object] = []
    try:
        native = context.context
        cur_limbs = int(native.L) - int(input_level)
        for row in layout.pack(hidden, dtype=np.float64):
            encrypted.append(
                context.client.encrypt(
                    np.asarray(row).reshape(-1),
                    device=str(native.device),
                    slots=int(layout.slots),
                    cur_limbs=cur_limbs,
                    scaling_factor=native.scale_at(cur_limbs),
                )
            )
        if str(native.device).startswith("cuda"):
            easyfhe.cuda.synchronize()
        output = tuple(encrypted)
        encrypted = []
        return output
    finally:
        release_all(encrypted)


def _release_transient_device_memory(device: str) -> None:
    """Return dead benchmark temporaries to the device allocator.

    Full-layer warmup creates several multi-GiB encoded-weight tensors.  The
    operators release them, but PyTorch may keep their blocks reserved.  A
    measured repetition can then fail even though no operator cache owns those
    tensors.  This cleanup runs only between complete samples, outside every
    reported operator timer.
    """

    gc.collect()
    if not str(device).startswith("cuda"):
        return
    module = easyfhe.get_device_module(str(device))
    module.synchronize()
    module.empty_cache()