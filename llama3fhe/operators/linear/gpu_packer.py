"""GPU diagonal weight packers for the feature-major linears.

Two Triton packers serve the production path: the token-lane BSGS packer
(square diagonals) and its structured variant (rectangular Q/K/V and MLP
matrices packed through a square physical geometry). Everything runs on the
EasyFHE tensor bridge — ``easyfhe`` is installed as ``torch`` for Triton.
"""

from __future__ import annotations

import sys
import types

import easyfhe as torch

if not hasattr(torch, "version"):
    torch.version = types.SimpleNamespace(hip=None)
elif not hasattr(torch.version, "hip"):
    torch.version.hip = None
sys.modules.setdefault("torch", torch)
import triton
import triton.language as tl

SLOTS = 2**15


def synchronize() -> None:
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()


def _enable_triton_easyfhe_bridge() -> None:
    """Let Triton use EasyFHE's torch-compatible CUDA runtime."""
    import triton.backends.driver as triton_driver

    if getattr(triton_driver.GPUDriver, "_easyfhe_bridge", False):
        return

    def init(self):
        self.get_device_capability = torch.cuda.get_device_capability
        self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device

    triton_driver.GPUDriver.__init__ = init
    triton_driver.GPUDriver._easyfhe_bridge = True


def pack_token_lane_bsgs_weight_chunk_triton_gpu(
    weight,
    *,
    slots: int,
    dimension: int,
    token_lanes: int,
    baby_steps: int,
    giant_base: int,
    giant_count: int,
    device: str = "cuda",
    block_size: int = 256,
):
    """Pack one feature-major TokenLane BSGS chunk on the GPU.

    The output exactly matches ``new_packing.qkv._build_chunk_bundle``:

    ``row(g,b)[slot] = W[(feature(slot)-g*B) % D,
                          (feature(slot)+b) % D]``.

    ``weight`` stays resident on the GPU, avoiding a full square weight upload
    and a CPU ``rows x slots`` temporary for every plaintext chunk.
    """

    _enable_triton_easyfhe_bridge()
    slots = int(slots)
    dimension = int(dimension)
    token_lanes = int(token_lanes)
    baby_steps = int(baby_steps)
    giant_base = int(giant_base)
    giant_count = int(giant_count)
    if slots <= 0 or dimension <= 0 or token_lanes <= 0:
        raise ValueError("slots/dimension/token_lanes must be positive.")
    if dimension * token_lanes != slots:
        raise ValueError(
            "TokenLane GPU packing requires dimension*token_lanes == slots, "
            f"got {dimension}*{token_lanes} != {slots}."
        )
    if baby_steps <= 0 or giant_base < 0 or giant_count <= 0:
        raise ValueError(
            "baby_steps/giant_count must be positive and giant_base non-negative."
        )
    if not torch.is_tensor(weight):
        raise TypeError(
            f"TokenLane GPU packer expects a device tensor, got {type(weight)}."
        )
    if not bool(getattr(weight, "is_cuda", False)):
        raise ValueError("TokenLane GPU packer requires a CUDA-resident weight tensor.")
    if tuple(int(value) for value in weight.shape) != (dimension, dimension):
        raise ValueError(
            f"weight must have shape {(dimension, dimension)}, got "
            f"{tuple(int(value) for value in weight.shape)}."
        )

    complex_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "complex32", None),
            getattr(torch, "complex64", None),
            getattr(torch, "complex128", None),
        )
        if dtype is not None
    )
    weight_is_complex = weight.dtype in complex_dtypes
    if weight_is_complex:
        if weight.dtype != torch.complex128:
            weight = weight.to(dtype=torch.complex128)
        weight_storage = weight.view(torch.float64)
    else:
        if weight.dtype not in (torch.float32, torch.float64):
            weight = weight.to(dtype=torch.float64)
        weight_storage = weight

    row_count = giant_count * baby_steps
    packed = torch.empty(
        (row_count, slots), dtype=torch.complex128, device=device
    )
    packed_storage = packed.view(torch.float64)
    grid = (row_count, triton.cdiv(slots, block_size))
    _pack_token_lane_bsgs_weight_chunk_triton_kernel[grid](
        weight_storage,
        packed_storage,
        slots,
        dimension,
        token_lanes,
        baby_steps,
        giant_base,
        WEIGHT_COMPLEX=bool(weight_is_complex),
        WEIGHT_FP32=bool(
            not weight_is_complex and weight.dtype == torch.float32
        ),
        BLOCK=block_size,
    )
    return packed


@triton.jit
def _pack_token_lane_bsgs_weight_chunk_triton_kernel(
    weight,
    packed_real,
    SLOTS: tl.constexpr,
    DIMENSION: tl.constexpr,
    TOKEN_LANES: tl.constexpr,
    BABY_STEPS: tl.constexpr,
    GIANT_BASE: tl.constexpr,
    WEIGHT_COMPLEX: tl.constexpr,
    WEIGHT_FP32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_index = tl.program_id(0)
    slots = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot_mask = slots < SLOTS

    giant_rel = row_index // BABY_STEPS
    baby = row_index - giant_rel * BABY_STEPS
    giant = GIANT_BASE + giant_rel
    feature = slots // TOKEN_LANES
    output_feature = (feature - giant * BABY_STEPS) % DIMENSION
    output_feature = tl.where(
        output_feature < 0, output_feature + DIMENSION, output_feature
    )
    input_feature = (feature + baby) % DIMENSION
    weight_index = output_feature * DIMENSION + input_feature

    if WEIGHT_COMPLEX:
        real = tl.load(weight + 2 * weight_index, mask=slot_mask, other=0.0)
        imag = tl.load(
            weight + 2 * weight_index + 1, mask=slot_mask, other=0.0
        )
    else:
        loaded = tl.load(weight + weight_index, mask=slot_mask, other=0.0)
        real = loaded.to(tl.float64) if WEIGHT_FP32 else loaded
        imag = tl.zeros((BLOCK,), tl.float64)

    output_index = (row_index * SLOTS + slots) * 2
    tl.store(packed_real + output_index, real, mask=slot_mask)
    tl.store(packed_real + output_index + 1, imag, mask=slot_mask)


def pack_structured_token_lane_bsgs_weight_chunk_triton_gpu(
    sources,
    metadata,
    *,
    slots: int,
    dimension: int,
    token_lanes: int,
    baby_steps: int,
    giant_base: int,
    giant_count: int,
    device: str = "cuda",
    block_size: int = 256,
):
    """Pack BSGS diagonals directly from logical real weight sources.

    ``metadata`` contains device vectors describing the physical output row
    and input-column mapping.  The map is documented by
    :class:`StructuredLinearWeight`; this kernel applies it only at entries
    selected by the diagonal schedule, so no dense physical square matrix is
    ever constructed.
    """

    _enable_triton_easyfhe_bridge()
    slots = int(slots)
    dimension = int(dimension)
    token_lanes = int(token_lanes)
    baby_steps = int(baby_steps)
    giant_base = int(giant_base)
    giant_count = int(giant_count)
    if slots <= 0 or dimension <= 0 or token_lanes <= 0:
        raise ValueError("slots/dimension/token_lanes must be positive.")
    if dimension * token_lanes != slots:
        raise ValueError(
            "structured TokenLane packing requires dimension*token_lanes "
            f"== slots, got {dimension}*{token_lanes} != {slots}."
        )
    if baby_steps <= 0 or giant_base < 0 or giant_count <= 0:
        raise ValueError(
            "baby_steps/giant_count must be positive and giant_base non-negative."
        )
    sources = tuple(sources)
    if not 1 <= len(sources) <= 3:
        raise ValueError("structured GPU packer requires one to three sources.")
    for index, source in enumerate(sources):
        if not torch.is_tensor(source) or not bool(
            getattr(source, "is_cuda", False)
        ):
            raise ValueError(
                f"structured source {index} must be a CUDA tensor."
            )
        if source.ndim != 2:
            raise ValueError(
                f"structured source {index} must be rank two, got {source.shape}."
            )
    source_dtype = sources[0].dtype
    if any(source.dtype != source_dtype for source in sources):
        raise TypeError("structured GPU sources must use one common dtype.")
    if source_dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "structured GPU sources must use float32 or float64, got "
            f"{source_dtype}."
        )

    required_metadata = (
        "real_source",
        "real_row",
        "real_scale",
        "imag_source",
        "imag_row",
        "imag_scale",
        "column",
        "column_scale",
    )
    missing = [name for name in required_metadata if name not in metadata]
    if missing:
        raise ValueError(f"structured GPU metadata is missing {missing}.")
    for name in required_metadata:
        value = metadata[name]
        if not torch.is_tensor(value) or not bool(
            getattr(value, "is_cuda", False)
        ):
            raise ValueError(f"structured metadata {name!r} must be a CUDA tensor.")
        if tuple(int(item) for item in value.shape) != (dimension,):
            raise ValueError(
                f"structured metadata {name!r} must have shape "
                f"{(dimension,)}, got {tuple(int(item) for item in value.shape)}."
            )

    padded_sources = sources + (sources[0],) * (3 - len(sources))
    strides = [
        (int(source.stride(0)), int(source.stride(1)))
        for source in padded_sources
    ]
    row_count = giant_count * baby_steps
    packed = torch.empty(
        (row_count, slots), dtype=torch.complex128, device=device
    )
    packed_storage = packed.view(torch.float64)
    grid = (row_count, triton.cdiv(slots, block_size))
    _pack_structured_token_lane_bsgs_weight_chunk_triton_kernel[grid](
        padded_sources[0],
        padded_sources[1],
        padded_sources[2],
        metadata["real_source"],
        metadata["real_row"],
        metadata["real_scale"],
        metadata["imag_source"],
        metadata["imag_row"],
        metadata["imag_scale"],
        metadata["column"],
        metadata["column_scale"],
        packed_storage,
        slots,
        dimension,
        token_lanes,
        baby_steps,
        giant_base,
        strides[0][0],
        strides[0][1],
        strides[1][0],
        strides[1][1],
        strides[2][0],
        strides[2][1],
        SOURCE_COUNT=len(sources),
        SOURCE_FP32=bool(source_dtype == torch.float32),
        BLOCK=block_size,
    )
    return packed


@triton.jit
def _pack_structured_token_lane_bsgs_weight_chunk_triton_kernel(
    source0,
    source1,
    source2,
    real_source_map,
    real_row_map,
    real_scale_map,
    imag_source_map,
    imag_row_map,
    imag_scale_map,
    column_map,
    column_scale_map,
    packed_real,
    SLOTS: tl.constexpr,
    DIMENSION: tl.constexpr,
    TOKEN_LANES: tl.constexpr,
    BABY_STEPS: tl.constexpr,
    GIANT_BASE: tl.constexpr,
    SOURCE0_ROW_STRIDE: tl.constexpr,
    SOURCE0_COLUMN_STRIDE: tl.constexpr,
    SOURCE1_ROW_STRIDE: tl.constexpr,
    SOURCE1_COLUMN_STRIDE: tl.constexpr,
    SOURCE2_ROW_STRIDE: tl.constexpr,
    SOURCE2_COLUMN_STRIDE: tl.constexpr,
    SOURCE_COUNT: tl.constexpr,
    SOURCE_FP32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_index = tl.program_id(0)
    slots = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot_mask = slots < SLOTS

    giant_rel = row_index // BABY_STEPS
    baby = row_index - giant_rel * BABY_STEPS
    giant = GIANT_BASE + giant_rel
    feature = slots // TOKEN_LANES
    output_feature = (feature - giant * BABY_STEPS) % DIMENSION
    output_feature = tl.where(
        output_feature < 0, output_feature + DIMENSION, output_feature
    )
    input_feature = (feature + baby) % DIMENSION

    real_source = tl.load(real_source_map + output_feature, mask=slot_mask)
    real_row = tl.load(real_row_map + output_feature, mask=slot_mask)
    real_scale = tl.load(real_scale_map + output_feature, mask=slot_mask)
    imag_source = tl.load(imag_source_map + output_feature, mask=slot_mask)
    imag_row = tl.load(imag_row_map + output_feature, mask=slot_mask)
    imag_scale = tl.load(imag_scale_map + output_feature, mask=slot_mask)
    column = tl.load(column_map + input_feature, mask=slot_mask)
    column_scale = tl.load(column_scale_map + input_feature, mask=slot_mask)
    column_valid = column >= 0

    real = tl.zeros((BLOCK,), tl.float64)
    imag = tl.zeros((BLOCK,), tl.float64)

    real0 = tl.load(
        source0 + real_row * SOURCE0_ROW_STRIDE + column * SOURCE0_COLUMN_STRIDE,
        mask=slot_mask & column_valid & (real_source == 0),
        other=0.0,
    )
    imag0 = tl.load(
        source0 + imag_row * SOURCE0_ROW_STRIDE + column * SOURCE0_COLUMN_STRIDE,
        mask=slot_mask & column_valid & (imag_source == 0),
        other=0.0,
    )
    real += real0.to(tl.float64) if SOURCE_FP32 else real0
    imag += imag0.to(tl.float64) if SOURCE_FP32 else imag0
    if SOURCE_COUNT > 1:
        real1 = tl.load(
            source1
            + real_row * SOURCE1_ROW_STRIDE
            + column * SOURCE1_COLUMN_STRIDE,
            mask=slot_mask & column_valid & (real_source == 1),
            other=0.0,
        )
        imag1 = tl.load(
            source1
            + imag_row * SOURCE1_ROW_STRIDE
            + column * SOURCE1_COLUMN_STRIDE,
            mask=slot_mask & column_valid & (imag_source == 1),
            other=0.0,
        )
        real += real1.to(tl.float64) if SOURCE_FP32 else real1
        imag += imag1.to(tl.float64) if SOURCE_FP32 else imag1
    if SOURCE_COUNT > 2:
        real2 = tl.load(
            source2
            + real_row * SOURCE2_ROW_STRIDE
            + column * SOURCE2_COLUMN_STRIDE,
            mask=slot_mask & column_valid & (real_source == 2),
            other=0.0,
        )
        imag2 = tl.load(
            source2
            + imag_row * SOURCE2_ROW_STRIDE
            + column * SOURCE2_COLUMN_STRIDE,
            mask=slot_mask & column_valid & (imag_source == 2),
            other=0.0,
        )
        real += real2.to(tl.float64) if SOURCE_FP32 else real2
        imag += imag2.to(tl.float64) if SOURCE_FP32 else imag2

    real *= real_scale * column_scale
    imag *= imag_scale * column_scale
    output_index = (row_index * SLOTS + slots) * 2
    tl.store(packed_real + output_index, real, mask=slot_mask)
    tl.store(packed_real + output_index + 1, imag, mask=slot_mask)
