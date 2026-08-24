"""``trt`` backend: TRT-LLM's custom allreduce op.

``torch.ops.trtllm.allreduce`` forced to the exact MIN_LATENCY strategy,
which dispatches its Lamport oneshot/twoshot family and falls back to
NCCL when needed.

Construction is COLLECTIVE (IPC workspace rendezvous) and assumes the
whole world is the TP group (TRT-LLM's Mapping) — the dispatcher only
builds it for whole-world groups. TRT-LLM ships no other custom DENSE
collectives worth wiring: its allgather / reducescatter / alltoall
python ops are plain NCCL wrappers, and the rest of its comm zoo is
topology-gated (MNNVL, UB) or specialized
(MoEAllReduce, helix A2A, nvfp4_gemm_allreduce which needs
NVFP4-quantized operands).
"""

from __future__ import annotations

import importlib.util
import os

import torch

from ..context import Ctx

# Measured 2026-08-22: early trigger (=0) REGRESSED the full layer
# ~10-20 us/iter — the deferred lamport cleanup lands on later ARs and
# no torch-launched follower exploits the early window. Keep =1 until
# the followers are PDL-launched kernels.
_TRIGGER_AT_END = os.environ.get("K3_TRT_AR_TRIGGER_AT_END", "1") == "1"


def available() -> bool:
    return importlib.util.find_spec("tensorrt_llm") is not None


class TrtBackend:
    def __init__(self, ctx: Ctx) -> None:
        from tensorrt_llm._torch.distributed.ops import (
            get_allreduce_workspace,
        )
        from tensorrt_llm.functional import (
            AllReduceFusionOp,
            AllReduceStrategy,
        )
        from tensorrt_llm.mapping import Mapping
        self.ctx = ctx
        self.mapping = Mapping(world_size=ctx.world, tp_size=ctx.world,
                               rank=ctx.rank, gpus_per_node=ctx.world)
        self.tp_group = list(self.mapping.tp_group)
        self.workspace = get_allreduce_workspace(self.mapping)
        self._op = AllReduceFusionOp
        self._strategy = AllReduceStrategy

    def _call(self, x, residual, gamma, eps: float) -> list:
        fusion = (self._op.RESIDUAL_RMS_NORM if residual is not None
                  else self._op.NONE)
        # trigger_completion_at_end=False = the PDL early trigger the
        # stock fused tail uses: the AR signals launch-completion while
        # its lamport spin runs, so the NEXT PDL-launched kernel on this
        # stream overlaps the wait (flashinfer-backend kernels already
        # launch_with_pdl). K3_TRT_AR_TRIGGER_AT_END=1 restores the
        # conservative behaviour.
        return torch.ops.trtllm.allreduce(
            x, residual, gamma, None, None, self.workspace,
            self.tp_group, int(self._strategy.MIN_LATENCY),
            int(fusion), eps, _TRIGGER_AT_END)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self._call(x, None, None, 1e-5)[0]

    def allreduce_norm(self, x, gamma, eps, residual):
        """Returns (norm_out, residual_out). ``residual`` None supplies
        the cached zero residual (the fused kernel requires one)."""
        if residual is None:
            residual = self.ctx.zero_residual(x.shape[0], x.shape[1])
        outs = self._call(x, residual, gamma, eps)
        return outs[0], outs[1]
