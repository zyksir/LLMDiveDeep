"""``torch_low_contention`` backend: torch's low-contention symm-mem
collectives (``_low_contention_all_gather`` / ``_reduce_scatter``).

They run on symm-mem's PRIVATE backend stream; ``wait_tensor`` blocks
the calling stream on it AND pops the c10d work registry (leaked works
would otherwise accumulate per call). Capacity-unbounded (they manage
their own workspace).
"""

from __future__ import annotations

import torch


def all_gather(group, x: torch.Tensor) -> torch.Tensor:
    out = torch.ops.symm_mem._low_contention_all_gather(
        x, group.group_name)
    torch.ops._c10d_functional.wait_tensor(out)
    return out


def reduce_scatter(group, x: torch.Tensor) -> torch.Tensor:
    out = torch.ops.symm_mem._low_contention_reduce_scatter(
        x, "sum", group.group_name)
    torch.ops._c10d_functional.wait_tensor(out)
    return out
