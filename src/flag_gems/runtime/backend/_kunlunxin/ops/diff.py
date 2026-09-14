import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# The generic diff uses @libtuner (key=["M","N"] with 45 configs on kunlunxin)
# so every distinct (M, N) shape re-autotunes all configs -> huge compile +
# IR explosion (13.6M-line dump). Worse, its diff_kernel_2d addresses a 2D
# strided tile `M_offsets[:,None]*M_STRIDE + offs` whose runtime row stride
# defeats XPU contiguity analysis -> fully discrete access (0.003-0.03x torch
# on every 2D/3D shape).
#
# Fix (no libtuner, fixed BLOCK): drive one program per (row, chunk) with a
# pre-offset base pointer so each program does a purely contiguous 1D block-DMA
# `out[row, j:j+BLOCK] = in[row, j+1:...] - in[row, j:...]`. A fixed BLOCK=8192
# beats an N-adaptive block on XPU (large tiles stay well utilized; smaller
# tiles regress small-N cases). 1D inputs keep the fast flat-DMA path.
BLOCK = 1024
# fp16/fp32 widen the subtraction to fp32 before storing, so a wide BLOCK is
# numerically safe and measurably faster (0.250x -> 0.356x dtype-equal-weight,
# fp32 0.259x -> 0.498x). bf16 must keep BLOCK=1024: at 8192 the BF16_RNE
# store path is folded into a native bf16 subtraction that truncates instead of
# rounding, so ~84% of elements drift past tolerance (test_diff dtype2, 11/65
# fail). int8/uint8/int16 likewise keep 1024 (validated there; no RNE involved
# but the narrow store is the same fragile path).
_BLOCK_FP = 8192


def _pick_block(dtype):
    return _BLOCK_FP if dtype in (torch.float16, torch.float32) else BLOCK


# Narrow dtypes need an explicit accumulation width. XPU fuses a narrow
# load/sub/store chain into a native narrow subtraction: for bf16/fp16 that
# rounds toward zero while ATen rounds to nearest-even (1 ulp per step, 2 ulp
# for n=2 -> over the 1e-4 tolerance, seen 2026-09-03 / 09-08), and for
# int8/uint8/int16 the backend cannot even select the narrow vector op
# (`LLVM ERROR: Cannot select: v32i16 = sub`, seen 2026-09-07). Widening
# explicitly per dtype removes the dependency on optimizer behaviour instead
# of relying on tricks (an earlier "multiply by a runtime 1.0" barrier was
# folded away by the optimizer, which is how both failures resurfaced).
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
def _to_bf16_rne(d):
    # The widened sub above is only honoured when the narrowing store cannot be
    # folded back into a native bf16 subtraction; when it is folded, XPU keeps
    # the extra mantissa bits truncated toward zero (1 ulp per step, 2 ulp for
    # n=2 -> fails the bf16 rtol on cancellation-heavy elements, seen
    # 2026-09-09). Rounding to nearest-even on the fp32 bit pattern makes the
    # rounding explicit, so the result no longer depends on that fusion.
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

    # n >= 2: ping-pong between two scratch buffers, writing the last iteration
    # directly into `output` (size N-n).
    scratch_a_shape = list(input.shape)
    scratch_a_shape[-1] = N - 1
    scratch_a = torch.empty(scratch_a_shape, device=input.device, dtype=input.dtype)
    if n >= 3:
        scratch_b_shape = list(input.shape)
        scratch_b_shape[-1] = N - 2
        scratch_b = torch.empty(scratch_b_shape, device=input.device, dtype=input.dtype)

    _launch(input, scratch_a, N, N - 1, N)
    torch_device_fn.synchronize()
    src, src_stride = scratch_a, N - 1

    for k in range(1, n):
        if k == n - 1:
            dst, dst_stride = output, N - n
        elif k % 2 == 1:
            dst, dst_stride = scratch_b, N - 2
        else:
            dst, dst_stride = scratch_a, N - 1
        _launch(src, dst, src_stride, dst_stride, N - k)
        torch_device_fn.synchronize()
        src, src_stride = dst, dst_stride

    return torch.moveaxis(output, -1, dim)
