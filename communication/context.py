"""Shared per-instance state handed to every backend.

``Ctx`` owns what more than one backend needs: the process group and
its geometry, the dtype/device, the zero-residual cache (fused AR+norm
kernels require a residual operand; a zero one recovers the pure
post-reduce norm), and the CUDA-graph cache (capture-once/replay with
returned tensors living in the graph pool).
"""

from __future__ import annotations

from typing import Callable

import torch


class Ctx:
    def __init__(self, group, world: int, rank: int,
                 device: torch.device, dtype: torch.dtype,
                 max_numel: int) -> None:
        self.group = group
        self.world = world
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.max_numel = max_numel
        self._zero_res: dict[int, torch.Tensor] = {}
        self._graphs: dict[tuple, torch.cuda.CUDAGraph] = {}

    def zero_residual(self, rows: int, dim: int) -> torch.Tensor:
        """[rows, dim] zeros, cached per dim and grown as needed."""
        z = self._zero_res.get(dim)
        if z is None or z.shape[0] < rows:
            z = torch.zeros(rows, dim, device=self.device,
                            dtype=self.dtype)
            self._zero_res[dim] = z
        return z[:rows]

    def graphed(self, key: tuple, fn: Callable):
        """Replay ``fn`` as a cached CUDA graph (capture on first use,
        keyed on operand pointers — stable layer buffers replay, fresh
        pointers capture a new graph) and return fn's return value:
        tensors returned from the captured invocation live in the
        graph's pool, so the SAME tensors are valid after every
        replay. Inside an OUTER capture the body runs eagerly instead
        (graph.replay() cannot nest)."""
        if torch.cuda.is_current_stream_capturing():
            return fn()
        ent = self._graphs.get(key)
        if ent is None:
            fn()  # warmup outside capture (cuBLAS / multimem need it)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                ret = fn()
            ent = (g, ret)
            self._graphs[key] = ent
        ent[0].replay()
        return ent[1]
