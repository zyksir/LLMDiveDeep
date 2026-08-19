"""Backend wrapper for the custom B10 NVLS multimem kernels."""

from __future__ import annotations

import torch

from ..context import Ctx
from ..kernels.b10_multimem import MultimemARKernel


class B10MultimemBackend:
    def __init__(self, ctx: Ctx, blocks: int = 16) -> None:
        self.ctx = ctx
        self._kernel = MultimemARKernel(ctx, blocks)

    def supports(self, op: str, x: torch.Tensor) -> bool:
        if x.dtype != self.ctx.dtype or x.numel() > self.ctx.max_numel:
            return False
        if op == "all_reduce":
            return x.numel() * x.element_size() % (self.ctx.world * 16) == 0
        if op == "allreduce_norm":
            return x.ndim == 2 and x.shape[0] <= self._kernel._MAX_ROWS
        return False

    def symm_input(self, shape, offset: int = 0) -> torch.Tensor:
        return self._kernel.symm_input(shape, offset)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self._kernel.all_reduce(x)

    def all_reduce_lamport(self, x: torch.Tensor) -> torch.Tensor:
        return self._kernel.all_reduce_lamport(x)

    def allreduce_norm(self, x, gamma, eps, residual):
        return self._kernel.allreduce_norm(x, gamma, eps, residual)
