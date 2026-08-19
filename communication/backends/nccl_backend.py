"""``nccl`` backend: plain torch.distributed compositions.

Capacity-unbounded — every op here allocates its own output and never
touches shared staging, so it is the fallback the dispatcher's
capacity guard resolves to for oversize operands.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def all_gather(group, x: torch.Tensor) -> torch.Tensor:
    world = dist.get_world_size(group)
    out = torch.empty(world * x.shape[0], x.shape[1],
                      device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(out, x, group=group)
    return out


def reduce_scatter(group, x: torch.Tensor) -> torch.Tensor:
    world = dist.get_world_size(group)
    out = torch.empty(x.shape[0] // world, x.shape[1],
                      device=x.device, dtype=x.dtype)
    dist.reduce_scatter_tensor(out, x, group=group)
    return out


def all_reduce(group, x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    dist.all_reduce(out, group=group)
    return out


def all_to_all(group, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    return out
