"""Vendored sglang K3 fused GEMM+communication ops (see package docstring).

Two kernel families from ``jit/csrc/kimi_k3/comm/``:

* ``gemm_ar`` (``gemm_ar.cuh``) — row-parallel ``allreduce(x @ w.T)`` in
  ONE kernel: each rank computes its local [M, K] @ [7168, K]^T partial
  and pushes finished tiles into a peer-mapped P2P region; a tile-local
  reduce writes the fully reduced [M, 7168] on every rank. N is fixed to
  7168 (K3 hidden), M in [1, 512]; the JIT module is per-K but the comm
  region is K-independent (``region_bytes`` depends only on world size),
  so one :func:`ensure_gemm_ar` serves every K. CUDA-graph compatible
  after init (device-resident epoch tickets). All TP ranks must call
  with the same M in lockstep.

* ``gemm_ag`` (``gemm_ag.cuh``) — column-parallel up_proj GEMV +
  multicast all-gather + fused add3 for small decode M (<= 12, TP8):
  ``out = x @ weight.T + b (+ c)`` where ``weight`` is the FULL
  replicated [7168, 3584] up_proj (each rank reads only its row slice).
  Rides the CustomAllReduceV2 push plane, so
  ``comm.get_sgl_ar_state()`` / ``all_reduce.register_comm`` must have
  run first.
"""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist

from ..sgl_copied_kernels.ops.kimi_k3 import gemm_ar as _gemm_ar
from ..sgl_copied_kernels.ops.kimi_k3.gemm_ag import (
    K as GEMM_AG_K,
    MAX_TOKENS as GEMM_AG_MAX_TOKENS,
    N as GEMM_AG_N,
    gemm_ag_up_proj,
)

__all__ = [
    "GEMM_AG_K",
    "GEMM_AG_MAX_TOKENS",
    "GEMM_AG_N",
    "GEMM_AR_MAX_TOKENS",
    "GEMM_AR_N",
    "ensure_gemm_ar",
    "gemm_ar_fits",
    "gemm_ag_up_proj",
    "o_proj_gemm_ar",
]

GEMM_AR_N = _gemm_ar.N  # 7168, the compile-time output width
GEMM_AR_MAX_TOKENS = _gemm_ar.MAX_TOKENS  # 512 (kMMax)

o_proj_gemm_ar = _gemm_ar.o_proj_gemm_ar
gemm_ar_fits = _gemm_ar.fits


def ensure_gemm_ar(k: int, group: Optional[dist.ProcessGroup] = None) -> None:
    """Idempotent gemm_ar setup; COLLECTIVE on the first call per
    process (P2P region rendezvous + JIT build of the [k] module — run
    it from every rank together, outside CUDA-graph capture).

    ``group`` must be non-NCCL (the rendezvous runs CPU-side
    collectives); ``None`` reuses the adapter-owned gloo group over
    WORLD, matching how :func:`~.comm.build_sgl_ar_state` resolves its
    group. Later calls with a DIFFERENT ``k`` are fine: the comm region
    is K-independent and each K's JIT module binds to it lazily.
    """
    if _gemm_ar.initialized():
        return
    from .comm import _resolve_group

    g = _resolve_group(group)
    _gemm_ar.init(
        world_size=dist.get_world_size(g),
        rank=dist.get_rank(g),
        group=g,
        k=int(k),
    )
