"""Backend wrapper for the custom B10 copy-engine collectives."""

from __future__ import annotations

import torch

from ..context import Ctx
from ..kernels.b10_copy_engine import LowContentionComm


class B10CopyEngineBackend:
    def __init__(self, ctx: Ctx,
                 shared_buffer: torch.Tensor | None = None) -> None:
        self.ctx = ctx
        # one shared IPC pool for both movers (and, when provided, the
        # torch_symm output staging) — saves 2x max_numel of symm HBM
        self._shared_buffer = shared_buffer
        self._movers: dict[str, LowContentionComm] = {}

    def ready(self) -> bool:
        return bool(self._movers)

    def supports(self, op: str, x: torch.Tensor, kind: str) -> bool:
        if x.dtype != self.ctx.dtype:
            return False
        if op == "all_gather":
            return x.numel() * self.ctx.world <= self.ctx.max_numel
        if op in ("reduce_scatter", "all_to_all"):
            return x.numel() <= self.ctx.max_numel
        if op == "all_reduce":
            return (
                x.numel() <= self.ctx.max_numel
                and (kind != "sm" or x.shape[0] % self.ctx.world == 0)
            )
        return False

    def build(self) -> None:
        self.mover("dma")
        self.mover("sm")

    def mover(self, kind: str) -> LowContentionComm:
        mover = self._movers.get(kind)
        if mover is None:
            mover = LowContentionComm(
                self.ctx.group,
                self.ctx.max_numel,
                dtype=self.ctx.dtype,
                device=self.ctx.device,
                mover=kind,
                buffer=self._shared_buffer,
            )
            self._movers[kind] = mover
        return mover

    def all_gather(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        mover = self.mover(kind)
        out = mover.all_gather(x, "push")
        mover.wait()
        return out

    def reduce_scatter(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        return self.mover(kind).reduce_scatter(x, "push")

    def all_reduce(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        return self.mover(kind).all_reduce(x, "push")

    def all_to_all(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        mover = self.mover(kind)
        out = mover.all_to_all(x, "push")
        mover.wait()
        return out
