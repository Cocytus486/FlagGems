import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

try:
    # `.language` carries the tle.dsa surface (uni_sram row DMA) used by the 2D
    # diff path on the KL3 cluster pipeline.
    import triton.experimental.tle.language as tle_xpu  # noqa: F401
except ImportError:  # triton without the XPU tile-language extension
    tle_xpu = None

logger = logging.getLogger(__name__)

# The generic diff uses @libtuner (key=["M","N"]) which re-autotunes every
# distinct (M, N) shape -> large compile + IR explosion. Worse, its 2D kernel
# addresses a strided tile whose runtime row stride defeats XPU contiguity
# analysis -> discrete access on every 2D/3D shape.
#
# Fix (no libtuner, fixed BLOCK): drive one program per (row, chunk) with a
# pre-offset base pointer so each program does a purely contiguous 1D block-DMA.
# 1D inputs keep the fast flat-DMA path.
BLOCK = 1024
# fp16/fp32 widen the subtraction to fp32 before storing, so a wider BLOCK is
# numerically safe. bf16 must keep BLOCK=1024: at 8192 the BF16_RNE store path
# is folded into a native bf16 subtraction that truncates instead of rounding.
# int8/uint8/int16 likewise keep 1024 (the narrow store is the same fragile
# path).
_BLOCK_FP = 8192


def _pick_block(dtype):
    return _BLOCK_FP if dtype in (torch.float16, torch.float32) else BLOCK

# Narrow dtypes need an explicit accumulation width. XPU fuses a narrow
# load/sub/store chain into a native narrow subtraction: for bf16/fp16 that
# rounds toward zero while ATen rounds to nearest-even, and for int8/uint8/int16
# the backend cannot select the narrow vector op
# (`LLVM ERROR: Cannot select: v32i16 = sub`). Widening explicitly per dtype
# removes the dependency on optimizer behaviour.
_FP32_ACC_DTYPES = (torch.float16, torch.bfloat16)
_INT32_ACC_DTYPES = (torch.int8, torch.uint8, torch.int16)


def _acc_flags(dtype):
    return (
        dtype in _FP32_ACC_DTYPES,
        dtype in _INT32_ACC_DTYPES,
        dtype is torch.bfloat16,
    )


@triton.jit
def _diff_sub(a, b, FP32_ACC: tl.constexpr, INT_ACC: tl.constexpr):
    if FP32_ACC:
        d = b.to(tl.float32) - a.to(tl.float32)
    elif INT_ACC:
        d = b.to(tl.int32) - a.to(tl.int32)
    else:
        d = b - a
    return d


@triton.jit
def _diff_sub2(a, b, c, FP32_ACC: tl.constexpr, INT_ACC: tl.constexpr, BF16_RNE: tl.constexpr):
    # second-order diff (c - b) - (b - a). torch.diff(n=2) is two steps, and for
    # narrow floats each first-order diff is rounded to the storage dtype before
    # the second subtraction; reproduce that intermediate rounding so the fused
    # result stays bit-close to the reference. exact/int dtypes need none.
    d1a = _diff_sub(a, b, FP32_ACC, INT_ACC)
    d1b = _diff_sub(b, c, FP32_ACC, INT_ACC)
    if FP32_ACC:
        if BF16_RNE:
            d1a = _to_bf16_rne(d1a)
            d1b = _to_bf16_rne(d1b)
        else:
            d1a = d1a.to(tl.float16)
            d1b = d1b.to(tl.float16)
    return _diff_sub(d1a, d1b, FP32_ACC, INT_ACC)


@triton.jit
def _to_bf16_rne(d):
    # The widened sub above is only honoured when the narrowing store cannot be
    # folded back into a native bf16 subtraction; when it is folded, XPU keeps
    # the extra mantissa bits truncated toward zero (1 ulp per step, 2 ulp for
    # n=2 -> fails the bf16 rtol on cancellation-heavy elements). Rounding to
    # nearest-even on the fp32 bit pattern makes the rounding explicit, so the
    # result no longer depends on that fusion.
    # -65536 is 0xFFFF0000 as a signed int32 (keeps the bf16 bits).
    u = d.to(tl.int32, bitcast=True)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u & -65536).to(tl.float32, bitcast=True).to(tl.bfloat16)


@triton.jit
def _diff_store(out_ptr, offs, d, mask, BF16_RNE: tl.constexpr):
    if BF16_RNE:
        tl.store(out_ptr + offs, _to_bf16_rne(d), mask)
    else:
        tl.store(out_ptr + offs, d.to(out_ptr.dtype.element_ty), mask)


@libentry()
@triton.jit
def diff_kernel_1d(
    in_ptr,
    out_ptr,
    N_OUT,
    BLOCK: tl.constexpr,
    FP32_ACC: tl.constexpr = False,
    INT_ACC: tl.constexpr = False,
    BF16_RNE: tl.constexpr = False,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_OUT
    a = tl.load(in_ptr + offs, mask)
    b = tl.load(in_ptr + offs + 1, mask)
    d = _diff_sub(a, b, FP32_ACC, INT_ACC)
    _diff_store(out_ptr, offs, d, mask, BF16_RNE)


@libentry()
@triton.jit
def diff_kernel_1d_n2(
    in_ptr,
    out_ptr,
    N_OUT,
    BLOCK: tl.constexpr,
    FP32_ACC: tl.constexpr = False,
    INT_ACC: tl.constexpr = False,
    BF16_RNE: tl.constexpr = False,
):
    # Fused second-order diff in one launch: d2[i] = x[i+2] - 2*x[i+1] + x[i].
    # For small tensors the ping-pong path pays two launches + two syncs, which
    # dominates; a single kernel reads three shifted views and writes once.
    pid = tle.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_OUT
    a = tl.load(in_ptr + offs, mask)
    b = tl.load(in_ptr + offs + 1, mask)
    c = tl.load(in_ptr + offs + 2, mask)
    d = _diff_sub2(a, b, c, FP32_ACC, INT_ACC, BF16_RNE)
    _diff_store(out_ptr, offs, d, mask, BF16_RNE)


@libentry()
@triton.jit
def diff_kernel_2d(
    in_ptr,
    out_ptr,
    N_OUT,
    M_STRIDE_IN,
    M_STRIDE_OUT,
    BLOCK: tl.constexpr,
    FP32_ACC: tl.constexpr = False,
    INT_ACC: tl.constexpr = False,
    BF16_RNE: tl.constexpr = False,
):
    pid_m = tle.program_id(0)
    pid_c = tle.program_id(1)
    row_in = in_ptr + pid_m * M_STRIDE_IN
    row_out = out_ptr + pid_m * M_STRIDE_OUT
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_OUT
    a = tl.load(row_in + offs, mask)
    b = tl.load(row_in + offs + 1, mask)
    d = _diff_sub(a, b, FP32_ACC, INT_ACC)
    _diff_store(row_out, offs, d, mask, BF16_RNE)


@libentry()
@triton.jit
def diff_kernel_2d_n2(
    in_ptr,
    out_ptr,
    N_OUT,
    M_STRIDE_IN,
    M_STRIDE_OUT,
    BLOCK: tl.constexpr,
    FP32_ACC: tl.constexpr = False,
    INT_ACC: tl.constexpr = False,
    BF16_RNE: tl.constexpr = False,
):
    # Fused second-order diff for the small-tensor Triton path (same rationale as
    # diff_kernel_1d_n2): one launch instead of the ping-pong's two + two syncs.
    pid_m = tle.program_id(0)
    pid_c = tle.program_id(1)
    row_in = in_ptr + pid_m * M_STRIDE_IN
    row_out = out_ptr + pid_m * M_STRIDE_OUT
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_OUT
    a = tl.load(row_in + offs, mask)
    b = tl.load(row_in + offs + 1, mask)
    c = tl.load(row_in + offs + 2, mask)
    d = _diff_sub2(a, b, c, FP32_ACC, INT_ACC, BF16_RNE)
    _diff_store(row_out, offs, d, mask, BF16_RNE)


# dtypes the tle.dsa row path can express. fp16/bf16 buffers are widened to f32
# by TritonConvertType (the SDNN DMA does the 16-bit -> f32 conversion on the
# fly), so the LM subtraction is f32 and round-to-nearest. int32/int64/uint8
# have no supported uni_sram buffer, and int8/int16 would need the (broken
# under SDNN) widen-to-int32 to stay overflow-safe -- those keep the Triton
# kernel.
_TL_BUF_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
# On-chip budget for the two-buffer diff tile. tle_copy's single-buffer row
# kernel uses a 32KB tile; diff needs two buffers (a and a+1), so halve it. The
# row (what one DMA descriptor moves) keeps tle_copy's 4KB.
_DIFF_TILE_BYTES = 16384
_DIFF_ROW_BYTES = 4096
# Below this many elements the dsa path's fixed overhead (two uni_sram DMAs +
# to_tensor) loses to the simple per-row Triton kernel; above it the 2D tile's
# full-row DMA wins.
_TLE_MIN_NUMEL = 1_000_000


def _tle_dma_available():
    if tle_xpu is None:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


def _diff_tile(rows, cols, element_size):
    cols_block = min(
        triton.next_power_of_2(cols), max(_DIFF_ROW_BYTES // element_size, 1)
    )
    rows_block = min(
        triton.next_power_of_2(rows),
        max(_DIFF_TILE_BYTES // (cols_block * element_size), 1),
    )
    return rows_block, cols_block


@triton.jit(
    # None of these change the code, only the data it moves, so specializing on
    # them buys nothing and costs a kernel per shape/pointer class.
    do_not_specialize=["src_ptr", "dst_ptr", "rows", "cols", "s_row", "t_row", "row_blocks"]
)
def _tle_diff_row_kernel(
    src_ptr,
    dst_ptr,
    rows,
    cols,
    s_row,
    t_row,
    row_blocks,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BUF_DTYPE: tl.constexpr,
):
    # One program owns an [ROWS, COLS] tile of the output. diff reads
    # in[m, 0:N-1] -- never the element at index N -- so the +1 shift of the
    # second copy stays in bounds on every tile and `sizes` alone makes the tail
    # exact (no zero-fill, no cross-row overhang).
    pid = tl.program_id(0)
    row0 = (pid % row_blocks) * ROWS
    col0 = tl.program_id(1) * COLS

    r = row0 + tl.arange(0, ROWS)
    c = col0 + tl.arange(0, COLS)
    row_tail = tl.minimum(rows - row0, ROWS)
    col_tail = tl.minimum(cols - col0, COLS)

    src = src_ptr + r[:, None] * s_row + c[None, :]
    buf_a = tle_xpu.dsa.alloc([ROWS, COLS], BUF_DTYPE, tle_xpu.dsa.UNI_SRAM)
    buf_b = tle_xpu.dsa.alloc([ROWS, COLS], BUF_DTYPE, tle_xpu.dsa.UNI_SRAM)
    tle_xpu.dsa.copy(src, buf_a, sizes=[row_tail, col_tail])
    tle_xpu.dsa.copy(src + 1, buf_b, sizes=[row_tail, col_tail])

    a = tle_xpu.dsa.to_tensor(buf_a)
    b = tle_xpu.dsa.to_tensor(buf_b)
    d = b - a

    dst = dst_ptr + r[:, None] * t_row + c[None, :]
    tle_xpu.dsa.copy(d, dst, sizes=[row_tail, col_tail])


def diff(input, n=1, dim=-1, prepend=None, append=None) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DIFF")

    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)

    if n <= 0:
        return input

    shape = list(input.shape)
    dim = dim % input.ndim
    reduce_len = shape[dim]

    if n >= reduce_len:
        empty_tensor = torch.tensor([], dtype=input.dtype, device=input.device)
        return torch.reshape(empty_tensor, shape[:dim] + [0] + shape[(dim + 1) :])

    input = dim_compress(input, dim)
    N = reduce_len
    M = input.numel() // N

    is_1d = len(shape) == 1
    fp32_acc, int_acc, bf16_rne = _acc_flags(input.dtype)
    block = _pick_block(input.dtype)
    buf_dtype = (
        _TL_BUF_DTYPE.get(input.dtype)
        if (not is_1d and _tle_dma_available() and input.numel() >= _TLE_MIN_NUMEL)
        else None
    )

    def _launch(src, dst, in_stride_m, out_stride_m, n_bound):
        n_out = n_bound - 1
        with torch_device_fn.device(src.device):
            if is_1d:
                grid = (triton.cdiv(n_out, block),)
                diff_kernel_1d[grid](
                    src,
                    dst,
                    n_out,
                    BLOCK=block,
                    FP32_ACC=fp32_acc,
                    INT_ACC=int_acc,
                    BF16_RNE=bf16_rne,
                )
            else:
                if buf_dtype is not None:
                    rows_block, cols_block = _diff_tile(M, n_out, src.element_size())
                    row_blocks = triton.cdiv(M, rows_block)
                    grid = (row_blocks, triton.cdiv(n_out, cols_block))
                    _tle_diff_row_kernel[grid](
                        src,
                        dst,
                        M,
                        n_out,
                        in_stride_m,
                        out_stride_m,
                        row_blocks,
                        ROWS=rows_block,
                        COLS=cols_block,
                        BUF_DTYPE=buf_dtype,
                        is_sdnn=True,
                        num_stages=2,
                    )
                else:
                    grid = (M, triton.cdiv(n_out, block))
                    diff_kernel_2d[grid](
                        src,
                        dst,
                        n_out,
                        in_stride_m,
                        out_stride_m,
                        BLOCK=block,
                        FP32_ACC=fp32_acc,
                        INT_ACC=int_acc,
                        BF16_RNE=bf16_rne,
                    )

    out_shape = list(input.shape)
    out_shape[-1] = N - n
    output = torch.empty(out_shape, device=input.device, dtype=input.dtype)

    if n == 1:
        _launch(input, output, N, N - 1, N)
        return torch.moveaxis(output, -1, dim)

    if n == 2 and buf_dtype is None:
        # Fused second-order diff, one launch (see diff_kernel_1d_n2 / _2d_n2):
        # for small tensors the ping-pong below pays two launches + two syncs,
        # which dwarfs the actual work. N_OUT = N - 2 here.
        n_out = N - 2
        with torch_device_fn.device(input.device):
            if is_1d:
                grid = (triton.cdiv(n_out, block),)
                diff_kernel_1d_n2[grid](
                    input,
                    output,
                    n_out,
                    BLOCK=block,
                    FP32_ACC=fp32_acc,
                    INT_ACC=int_acc,
                    BF16_RNE=bf16_rne,
                )
            else:
                grid = (M, triton.cdiv(n_out, block))
                diff_kernel_2d_n2[grid](
                    input,
                    output,
                    n_out,
                    N,
                    N - 2,
                    BLOCK=block,
                    FP32_ACC=fp32_acc,
                    INT_ACC=int_acc,
                    BF16_RNE=bf16_rne,
                )
        return torch.moveaxis(output, -1, dim)

    # n >= 2: ping-pong between two scratch buffers, writing the last iteration
    # directly into `output` (size N-n). No explicit synchronize: the launches
    # queue on one stream, so the read-after-write on the scratch buffers is
    # already ordered; a host sync would only add per-iteration latency.
    scratch_a_shape = list(input.shape)
    scratch_a_shape[-1] = N - 1
    scratch_a = torch.empty(scratch_a_shape, device=input.device, dtype=input.dtype)
    if n >= 3:
        scratch_b_shape = list(input.shape)
        scratch_b_shape[-1] = N - 2
        scratch_b = torch.empty(scratch_b_shape, device=input.device, dtype=input.dtype)

    _launch(input, scratch_a, N, N - 1, N)
    src, src_stride = scratch_a, N - 1

    for k in range(1, n):
        if k == n - 1:
            dst, dst_stride = output, N - n
        elif k % 2 == 1:
            dst, dst_stride = scratch_b, N - 2
        else:
            dst, dst_stride = scratch_a, N - 1
        _launch(src, dst, src_stride, dst_stride, N - k)
        src, src_stride = dst, dst_stride

    return torch.moveaxis(output, -1, dim)
