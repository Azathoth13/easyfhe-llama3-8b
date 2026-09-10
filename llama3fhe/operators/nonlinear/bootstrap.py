from __future__ import annotations

"""Explicit EasyFHE bootstrap operator.

Bootstrap is a nonlinear HE operation, not model orchestration.  This module
owns the public EasyFHE bootstrap program and the optional two-pass Meta-BTS
correction.  Callers supply the native CKKS context through the project's
execution session; no private EasyFHE kernels or legacy-wheel patches live
here.
"""

import time
from dataclasses import dataclass

import easyfhe
import easyfhe.bs.openfhe as bs
from easyfhe import fhe

from llama3fhe.backend import release_if_supported


@dataclass(frozen=True)
class BootstrapConfig:
    log_slots: int = 15
    level_budget: tuple[int, int] = (4, 4)
    output_levels: int = 24
    iterations: int = 2
    precision_bits: int = 8
    mode: str = "modraise_first"
    # Rotation keys are model/session constants. Keep their CUDA material
    # resident across bootstraps and transformer layers by default; callers
    # may still opt into host-backed eviction for a constrained device.
    evict_rotation_cache: bool = False
    # Optional second full-slot program with a shallower CtoS/StoC budget.
    # Its program depth is smaller, so it may emit more output levels than
    # the main program (e.g. (3,3) -> 25 vs (4,4) -> 23 at depth 41).  Used
    # by refresh sites whose downstream chain is exactly a couple of levels
    # short (the PairFM paired-input threshold).
    deep_level_budget: tuple[int, int] | None = None
    deep_output_levels: int | None = None
    # Optional sparse program for the RMSN2 rsqrt slim ciphertext only.
    # Same output_levels as the main program; fewer slots because the 128
    # active scales already live in the first 2^slim_log_slots slots.
    slim_log_slots: int | None = None

    def __post_init__(self) -> None:
        if (self.deep_level_budget is None) != (self.deep_output_levels is None):
            raise ValueError(
                "deep_level_budget and deep_output_levels must be set together."
            )
        if self.slim_log_slots is not None:
            slim_log_slots = int(self.slim_log_slots)
            if not 1 <= slim_log_slots < int(self.log_slots):
                raise ValueError(
                    "slim_log_slots must be positive and strictly smaller "
                    "than the main bootstrap log_slots."
                )

    def spec(self) -> bs.BootstrapSpec:
        return bs.BootstrapSpec(
            log_slots=int(self.log_slots),
            level_budget=tuple(int(value) for value in self.level_budget),
            output_levels=int(self.output_levels),
            mode=str(self.mode),
        )

    def deep_spec(self) -> bs.BootstrapSpec | None:
        if self.deep_level_budget is None:
            return None
        return bs.BootstrapSpec(
            log_slots=int(self.log_slots),
            level_budget=tuple(int(v) for v in self.deep_level_budget),
            output_levels=int(self.deep_output_levels),
            mode=str(self.mode),
        )

    def slim_spec(self) -> bs.BootstrapSpec | None:
        if self.slim_log_slots is None:
            return None
        return bs.BootstrapSpec(
            log_slots=int(self.slim_log_slots),
            level_budget=tuple(int(value) for value in self.level_budget),
            output_levels=int(self.output_levels),
            mode=str(self.mode),
        )


@dataclass
class BootstrapOperator:
    """Prepared bootstrap program plus observable execution policy."""

    crypto_context: object
    config: BootstrapConfig
    program: object
    setup_seconds: float
    deep_program: object | None = None
    slim_program: object | None = None
    calls: int = 0
    native_calls: int = 0
    cache_eviction_seconds: float = 0.0

    @classmethod
    def prepare(
        cls,
        crypto_context,
        config: BootstrapConfig,
    ) -> "BootstrapOperator":
        start = time.perf_counter()
        program = None
        deep_program = None
        slim_program = None
        try:
            program = bs.generate(crypto_context.context, config.spec())
            program.constants.set_cache_mode("plain", clear=True)
            deep_spec = config.deep_spec()
            deep_program = (
                None
                if deep_spec is None
                else bs.generate(crypto_context.context, deep_spec)
            )
            if deep_program is not None:
                deep_program.constants.set_cache_mode("plain", clear=True)
            slim_spec = config.slim_spec()
            slim_program = (
                None
                if slim_spec is None
                else bs.generate(crypto_context.context, slim_spec)
            )
            if slim_program is not None:
                slim_program.constants.set_cache_mode("plain", clear=True)
            if str(crypto_context.device).startswith("cuda"):
                easyfhe.cuda.synchronize()
            return cls(
                crypto_context=crypto_context,
                config=config,
                program=program,
                deep_program=deep_program,
                slim_program=slim_program,
                setup_seconds=float(time.perf_counter() - start),
            )
        except Exception:
            try:
                release_if_supported(program)
            finally:
                try:
                    if deep_program is not None:
                        release_if_supported(deep_program)
                finally:
                    if slim_program is not None:
                        release_if_supported(slim_program)
            raise

    @staticmethod
    def required_rotations(
        *,
        log_n: int,
        log_slots: int = 15,
        level_budget: tuple[int, int] = (4, 4),
        deep_level_budget: tuple[int, int] | None = None,
        slim_log_slots: int | None = None,
    ) -> tuple[int, ...]:
        specs = [
            bs.BootstrapSpec(
                log_slots=int(log_slots),
                level_budget=tuple(int(value) for value in level_budget),
                output_levels=0,
            )
        ]
        if deep_level_budget is not None:
            specs.append(
                bs.BootstrapSpec(
                    log_slots=int(log_slots),
                    level_budget=tuple(
                        int(value) for value in deep_level_budget
                    ),
                    output_levels=0,
                )
            )
        if slim_log_slots is not None and int(slim_log_slots) != int(log_slots):
            specs.append(
                bs.BootstrapSpec(
                    log_slots=int(slim_log_slots),
                    level_budget=tuple(int(value) for value in level_budget),
                    output_levels=0,
                )
            )
        return tuple(
            bs.requirements(tuple(specs), log_n=int(log_n)).rotations
        )

    def _evict_unneeded_rotation_material(self) -> None:
        if not bool(self.config.evict_rotation_cache):
            return
        clear = getattr(
            self.crypto_context.context,
            "clear_cuda_rotation_cache",
            None,
        )
        if not callable(clear):
            raise RuntimeError(
                "EasyFHE Context.clear_cuda_rotation_cache is required. "
                "Run against the source checkout documented in "
                "docs/EASYFHE_BACKEND.md."
            )
        start = time.perf_counter()
        clear(
            keep_rotations=self.required_rotations(
                log_n=int(self.crypto_context.context.logN),
                log_slots=int(self.config.log_slots),
                level_budget=self.config.level_budget,
                deep_level_budget=self.config.deep_level_budget,
                slim_log_slots=self.config.slim_log_slots,
            )
        )
        self.cache_eviction_seconds += time.perf_counter() - start

    def refresh(
        self,
        cipher,
        *,
        iterations: int | None = None,
        precision_bits: int | None = None,
        match_two_pass_level: bool = False,
        use_deep_program: bool = False,
        use_slim_program: bool = False,
    ):
        """Return a refreshed ciphertext without consuming ``cipher``."""

        if bool(use_deep_program) and bool(use_slim_program):
            raise ValueError(
                "a refresh can use the deep program or the slim program, "
                "not both."
            )
        self._evict_unneeded_rotation_material()
        program = self.program
        if bool(use_deep_program):
            if self.deep_program is None:
                raise ValueError(
                    "deep bootstrap requested without deep_level_budget in "
                    "BootstrapConfig."
                )
            program = self.deep_program
        elif bool(use_slim_program):
            if self.slim_program is None:
                raise ValueError(
                    "slim bootstrap requested without slim_log_slots in "
                    "BootstrapConfig."
                )
            program = self.slim_program
        native_passes = [0]
        selected_iterations = (
            int(self.config.iterations)
            if iterations is None
            else int(iterations)
        )
        original_slots = int(cipher.slots)
        program_slots = int(program.spec.slots)
        bootstrap_input = cipher
        if original_slots > program_slots:
            if not bool(use_slim_program):
                raise ValueError(
                    "bootstrap input slots="
                    f"{original_slots} exceeds program slots={program_slots}."
                )
            # RMSN2 slim scales occupy the first 128 slots of a 2^15 cipher.
            # The sparse program only accepts 2^slim_log_slots slots, so
            # retarget the view; extract/broadcast still read those 128 values.
            bootstrap_input = cipher.cipher_like(
                cipher.cv, slots=program_slots
            )
        result = _bootstrap_with_optional_correction(
            bootstrap_input,
            crypto_context=self.crypto_context.context,
            program=program,
            iterations=selected_iterations,
            precision_bits=(
                int(self.config.precision_bits)
                if precision_bits is None
                else int(precision_bits)
            ),
            native_passes=native_passes,
        )
        if int(result.slots) != original_slots:
            result = result.cipher_like(result.cv, slots=original_slots)
        if bool(match_two_pass_level) and selected_iterations == 1:
            target_limbs = int(result.state.cur_limbs) - 1
            if target_limbs <= 1:
                release_if_supported(result)
                raise ValueError(
                    "matching the two-pass bootstrap level leaves too few limbs."
                )
            target_state = result.state.replace(
                cur_limbs=target_limbs,
                scale_degree=1,
                scaling_factor=None,
            )
            try:
                matched = fhe.align_to(
                    result, target_state, self.crypto_context.context
                )
            except Exception:
                release_if_supported(result)
                raise
            if matched is not result:
                release_if_supported(result)
            result = matched
        if str(self.crypto_context.device).startswith("cuda"):
            easyfhe.cuda.synchronize()
        self.calls += 1
        self.native_calls += int(native_passes[0])
        return result

    def release(self) -> None:
        try:
            release_if_supported(self.program)
        finally:
            try:
                if self.deep_program is not None:
                    release_if_supported(self.deep_program)
            finally:
                try:
                    if self.slim_program is not None:
                        release_if_supported(self.slim_program)
                finally:
                    self.program = None
                    self.deep_program = None
                    self.slim_program = None


def _multiply_integer(cipher, value: int, *, crypto_context):
    encoded = fhe.encode_scalar(
        int(value),
        cur_limbs=int(cipher.state.cur_limbs),
        scale_degree=0,
        mode="integer",
        context=crypto_context,
    )
    return fhe.homo_mul_scalar(cipher, encoded, crypto_context)


def _multiply_double_rescale(cipher, value: float, *, crypto_context):
    encoded = fhe.encode_scalar(
        float(value),
        cur_limbs=int(cipher.state.cur_limbs),
        scale_degree=1,
        scaling_factor=cipher.state.scaling_factor,
        context=crypto_context,
    )
    return fhe.homo_mul_scalar_rescale(cipher, encoded, crypto_context)


def _bootstrap_with_optional_correction(
    cipher,
    *,
    crypto_context,
    program,
    iterations: int,
    precision_bits: int,
    native_passes: list[int] | None = None,
):
    """Apply one bootstrap or the existing two-pass Meta-BTS correction."""

    iterations = int(iterations)
    if iterations not in (1, 2):
        raise ValueError("bootstrap iterations must be one or two.")

    def once(value):
        if native_passes is not None:
            native_passes[0] += 1
        return bs.bootstrap(value, crypto_context, program)

    if iterations == 1:
        return once(cipher)

    initial_limbs = int(cipher.state.cur_limbs)
    original = cipher.deep_copy()
    first = once(cipher)
    first_limbs = int(first.state.cur_limbs)
    if first_limbs <= initial_limbs:
        release_if_supported(original)
        return first

    try:
        factor = 1 << int(precision_bits)
        if int(precision_bits) <= 0:
            first_down = fhe.align_to(
                first, original.state, crypto_context
            )
            original_aligned = fhe.align_to(
                original, first_down.state, crypto_context
            )
            error_input = fhe.homo_sub(
                first_down, original_aligned, crypto_context
            )
            error = once(error_input)
            first_aligned = fhe.align_to(
                first, error.state, crypto_context
            )
            error_aligned = fhe.align_to(
                error, first_aligned.state, crypto_context
            )
            return fhe.homo_sub(first_aligned, error_aligned, crypto_context)

        first_scaled = _multiply_integer(
            first, factor, crypto_context=crypto_context
        )
        first_down = fhe.align_to(
            first_scaled, original.state, crypto_context
        )
        original_scaled = _multiply_integer(
            original, factor, crypto_context=crypto_context
        )
        original_scaled = fhe.align_to(
            original_scaled, first_down.state, crypto_context
        )
        error_input = fhe.homo_sub(
            first_down, original_scaled, crypto_context
        )
        error = once(error_input)
        first_aligned = fhe.align_to(
            first_scaled, error.state, crypto_context
        )
        error_aligned = fhe.align_to(
            error, first_aligned.state, crypto_context
        )
        corrected = fhe.homo_sub(
            first_aligned, error_aligned, crypto_context
        )
        return _multiply_double_rescale(
            corrected,
            1.0 / float(factor),
            crypto_context=crypto_context,
        )
    except ValueError:
        return first
    finally:
        release_if_supported(original)


__all__ = ["BootstrapConfig", "BootstrapOperator"]
