import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _assert_async_kernel(x_ptr, MSG: tl.constexpr):
    val = tl.load(x_ptr)
    tl.device_assert(val != 0, MSG)


def _assert_async(tensor: torch.Tensor, msg: str = "Assertion failed"):
    logger.debug("GEMS_KUNLUNXIN ASSERT_ASYNC")
    if tensor.numel() != 1:
        raise RuntimeError(
            f"Boolean value of Tensor with shape {list(tensor.shape)} is ambiguous"
        )
    with torch_device_fn.device(tensor.device):
        _assert_async_kernel[(1,)](tensor, MSG=msg)
