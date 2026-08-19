"""``torch_symm`` backend: torch symmetric-memory collectives.

Variants map 1:1 onto torch.ops.symm_mem kernels — if the symm-mem
library loads, ALL of them are available:

    torch_symm:multimem   multimem_all_reduce_ / multimem_all_gather_out
                          (NVSwitch multicast)
    torch_symm:1shot      one_shot_all_reduce
    torch_symm:2shot      two_shot_all_reduce_

The backend owns the two ``max_numel`` symm staging buffers; operands
are copied in when they are not already symm-resident (producers can
write into :meth:`symm_input` to skip that copy).

ALIASING contract: ops that reduce in place return VIEWS of the
staging buffer, valid until the next symm collective on this instance.
"""

from __future__ import annotations

import math

import torch
import torch.distributed._symmetric_memory as symm_mem_mod

from ..context import Ctx


def available() -> bool:
    return hasattr(torch.ops.symm_mem, "multimem_all_reduce_")


class TorchSymmBackend:
    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.gname = ctx.group.group_name
        self._symm_in = symm_mem_mod.empty(
            ctx.max_numel, dtype=ctx.dtype, device=ctx.device)
        self._symm_out = symm_mem_mod.empty(
            ctx.max_numel, dtype=ctx.dtype, device=ctx.device)
        symm_mem_mod.rendezvous(self._symm_in, self.gname)
        symm_mem_mod.rendezvous(self._symm_out, self.gname)

    # ------------------------------------------------------------ staging

    def symm_input(self, shape, dtype=None, offset: int = 0) -> torch.Tensor:
        """A symm-resident view to produce into (skips the stage-in
        copy of the next symm collective). ``offset`` (elements) lets
        two live operands share the buffer disjointly — e.g. a second
        AR's producer writing above the first AR's staging region."""
        n = int(math.prod(shape))
        assert offset + n <= self.ctx.max_numel, "symm staging overflow"
        t = self._symm_in[offset : offset + n]
        return t.view(*shape) if dtype is None else \
            t.view(*shape).to(dtype)

    def staged(self, x: torch.Tensor) -> torch.Tensor:
        base = self._symm_in.data_ptr()
        span = self.ctx.max_numel * self._symm_in.element_size()
        if x.is_contiguous() and base <= x.data_ptr() < base + span:
            return x  # already symm-resident (any offset view)
        v = self._symm_in[: x.numel()].view_as(x)
        v.copy_(x)
        return v

    def output(self, shape) -> torch.Tensor:
        n = math.prod(shape)
        assert n <= self.ctx.max_numel, "symm output staging overflow"
        return self._symm_out[:n].view(*shape)

    # ---------------------------------------------------------------- ops

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        src = self.staged(x)
        out = self._symm_out[: x.numel() * self.ctx.world] \
            .view(self.ctx.world * x.shape[0], x.shape[1])
        torch.ops.symm_mem.multimem_all_gather_out(src, self.gname, out)
        return out

    def all_reduce(self, x: torch.Tensor, variant: str) -> torch.Tensor:
        v = self.staged(x)
        if variant == "multimem":
            return torch.ops.symm_mem.multimem_all_reduce_(
                v, "sum", self.gname)
        if variant == "1shot":
            return torch.ops.symm_mem.one_shot_all_reduce(
                v, "sum", self.gname)
        if variant == "2shot":
            return torch.ops.symm_mem.two_shot_all_reduce_(
                v, "sum", self.gname)
        raise ValueError(f"unknown torch_symm all_reduce variant {variant!r}")
