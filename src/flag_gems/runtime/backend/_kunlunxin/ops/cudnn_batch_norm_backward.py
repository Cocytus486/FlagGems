# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems.utils import libentry

from .batch_norm import batch_norm_backward

logger = logging.getLogger(__name__)


# NOTE (kunlunxin / XPU): the generic cudnn_batch_norm_backward kernel tiles the
# [N, C, S] layout with a 2D [BLOCK_M, BLOCK_N] block, which the XPU compiler cannot
# lower for the benchmark shapes (`triton_xpu.convert_layout` shape mismatch /
# `tt.addptr` type mismatch in ConvertTritonXPUToLLVM). Reuse the contiguous
# per-(n, c)-slice 1D backward path from batch_norm.py instead; the only extra work is
# turning the saved variance into the inverse std that path expects.
@libentry()
@triton.jit
def _save_invstd_kernel(
    save_var, save_invstd, epsilon, n_elements, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    variance = tl.load(save_var + offsets, mask=mask).to(tl.float32)
    tl.store(save_invstd + offsets, tl.rsqrt(variance + epsilon), mask=mask)


def cudnn_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean: Tensor = None,
    running_var: Tensor = None,
    save_mean: Tensor = None,
    save_var: Tensor = None,
    epsilon: float = 1e-5,
    reserveSpace: Tensor = None,
):
    """CUDNN batch-norm backward using saved training statistics on XPU."""
    logger.debug("GEMS_KUNLUNXIN CUDNN_BATCH_NORM_BACKWARD")
    if save_mean is None or save_var is None:
        raise ValueError(
            "cudnn_batch_norm_backward requires saved training mean and variance"
        )

    save_var = save_var.contiguous()
    save_invstd = torch.empty_like(save_var, dtype=torch.float32)
    block = 256
    _save_invstd_kernel[(triton.cdiv(save_var.numel(), block),)](
        save_var, save_invstd, epsilon, save_var.numel(), BLOCK=block
    )
    return batch_norm_backward(
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train=True,
        eps=epsilon,
        output_mask=(True, True, True),
    )
