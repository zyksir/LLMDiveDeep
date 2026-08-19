"""NCCL collectives over torch symmetric-memory staging buffers.

This is the explicit NCCLSym baseline: NCCL owns the collective, while
inputs and outputs use symmetric allocations that NCCL can register and
promote to NVLS when the node's IMEX/NVLink fabric is provisioned.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def all_gather(group, symm, x: torch.Tensor) -> torch.Tensor:
    src = symm.staged(x)
    out = symm.output((dist.get_world_size(group) * x.shape[0], x.shape[1]))
    dist.all_gather_into_tensor(out, src, group=group)
    return out


def reduce_scatter(group, symm, x: torch.Tensor) -> torch.Tensor:
    src = symm.staged(x)
    out = symm.output((x.shape[0] // dist.get_world_size(group), x.shape[1]))
    dist.reduce_scatter_tensor(out, src, group=group)
    return out


def all_reduce(group, symm, x: torch.Tensor) -> torch.Tensor:
    out = symm.staged(x)
    dist.all_reduce(out, group=group)
    return out


def all_to_all(group, symm, x: torch.Tensor) -> torch.Tensor:
    src = symm.staged(x)
    out = symm.output(x.shape)
    dist.all_to_all_single(out, src, group=group)
    return out
