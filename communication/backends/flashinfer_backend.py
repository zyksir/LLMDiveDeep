"""``flashinfer`` backend: FlashInfer's Lamport allreduce-fusion
kernels (the TRT-LLM oneshot/twoshot family, PDL launches).

Variants:

    flashinfer:1shot   use_oneshot=True  — single-kernel Lamport AR
    flashinfer:2shot   use_oneshot=False — two-shot (RS + AG) variant

The same kernel serves every op that wraps an AR: plain kAllReduce for
``all_reduce`` / ``gemm_allreduce``, and the fused
kARResidualRMSNorm (AR + residual add + RMSNorm in ONE launch) for
``allreduce_norm`` / ``allreduce_norm_gemm`` (a zero residual recovers
the pure post-reduce norm).

``FlashInferBackend`` owns one shared IPC workspace sized at
construction. ``FlashInferAllReduce`` (moved from kimi_k3_layer/
comm.py) wraps a DEDICATED workspace — callers may wire two so
the fused norm-AR and the shared-expert AR can run CONCURRENTLY.
"""

from __future__ import annotations

import importlib.util

import torch
import torch.distributed as dist

from ..context import Ctx


def available() -> bool:
    return importlib.util.find_spec("flashinfer") is not None


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float,
            out: torch.Tensor | None = None) -> torch.Tensor:
    """FlashInfer's rmsnorm kernel, torch fallback when unavailable."""
    if available():
        from flashinfer.norm import rmsnorm as fi_rmsnorm
        return fi_rmsnorm(x.contiguous(), gamma, eps, out=out)
    res = torch.nn.functional.rms_norm(x, (x.shape[-1],), gamma, eps)
    if out is not None:
        out.copy_(res)
        return out
    return res


class FlashInferBackend:
    def __init__(self, ctx: Ctx, max_tokens: int,
                 max_hidden: int) -> None:
        from flashinfer.comm import (
            trtllm_create_ipc_workspace_for_all_reduce_fusion,
        )
        self.ctx = ctx
        self.max_tokens = max_tokens
        self.max_hidden = max_hidden
        (
            self._ipc,
            self.workspace,
            self._memory_handles,
            self._metadata,
        ) = trtllm_create_ipc_workspace_for_all_reduce_fusion(
            tp_rank=ctx.rank,
            tp_size=ctx.world,
            max_token_num=max_tokens,
            hidden_dim=max_hidden,
            use_fp32_lamport=False,
            group=ctx.group,
            create_metadata=True,
            use_symm_dev_mem=True,
        )

    def fits(self, tokens: int, hidden: int) -> bool:
        return tokens <= self.max_tokens and hidden <= self.max_hidden

    def _fusion(self, x: torch.Tensor, *, pattern: str, oneshot: bool,
                workspace=None, allreduce_out=None, residual_in=None,
                residual_out=None, norm_out=None, gamma=None,
                eps: float | None = None) -> None:
        from flashinfer.comm import (
            AllReduceFusionPattern,
            trtllm_allreduce_fusion,
        )
        code = (AllReduceFusionPattern.kAllReduce if pattern == "ar"
                else AllReduceFusionPattern.kARResidualRMSNorm)
        trtllm_allreduce_fusion(
            allreduce_in=x, world_size=self.ctx.world,
            world_rank=self.ctx.rank,
            token_num=x.shape[0], hidden_dim=x.shape[1],
            workspace_ptrs=(self.workspace if workspace is None
                            else workspace),
            launch_with_pdl=True, trigger_completion_at_end=True,
            fp32_acc=False, pattern_code=code, use_oneshot=oneshot,
            allreduce_out=allreduce_out, residual_in=residual_in,
            residual_out=residual_out, norm_out=norm_out,
            quant_out=None, scale_out=None, rms_gamma=gamma,
            rms_eps=eps, scale_factor=None, layout_code=None,
        )

    def all_reduce(self, x: torch.Tensor, oneshot: bool,
                   out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(x)
        self._fusion(x, pattern="ar", oneshot=oneshot,
                     allreduce_out=out)
        return out

    def allreduce_norm(self, x, gamma, eps, residual,
                       oneshot: bool, workspace=None):
        """One fused AR + residual + RMSNorm launch; the kernel writes
        freshly allocated outputs (no staging, no copies). ``residual``
        None supplies the cached zero residual; ``workspace`` overrides
        the shared one (bounded callers pass their dedicated one).
        Returns (norm, new_residual)."""
        if residual is None:
            residual = self.ctx.zero_residual(x.shape[0], x.shape[1])
        norm = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        self._fusion(x, pattern="fused", oneshot=oneshot,
                     workspace=workspace, residual_in=residual,
                     residual_out=residual_out, norm_out=norm,
                     gamma=gamma, eps=eps)
        return norm, residual_out

    def destroy(self) -> None:
        from flashinfer.comm import (
            trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
        )
        try:
            trtllm_destroy_ipc_workspace_for_all_reduce_fusion(self._ipc)
        except Exception:  # noqa: BLE001 - teardown best effort
            pass


class FlashInferAllReduce:
    """One-shot Lamport allreduce on a DEDICATED IPC workspace.

    This is the same kernel family TRT-LLM ships as the mnnvl
    ``oneshotAllreduceFusionKernel``. Concurrent reductions use TWO of
    these so the fused norm-AR and the shared-expert AR can run
    concurrently (they cannot share one workspace)."""

    def __init__(
        self,
        rank: int,
        world: int,
        *,
        max_tokens: int = 64,
        max_hidden: int = 10752,
        pdl: bool = True,
        group=None,
    ) -> None:
        from flashinfer.comm import (
            trtllm_create_ipc_workspace_for_all_reduce_fusion,
        )

        self.rank = rank
        self.world = world
        self.pdl = pdl
        (
            self._ipc,
            self._workspace,
            self._memory_handles,
            self._metadata,
        ) = trtllm_create_ipc_workspace_for_all_reduce_fusion(
            tp_rank=rank,
            tp_size=world,
            max_token_num=max_tokens,
            hidden_dim=max_hidden,
            use_fp32_lamport=False,
            group=group if group is not None else dist.group.WORLD,
            create_metadata=True,
            use_symm_dev_mem=True,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from flashinfer.comm import (
            AllReduceFusionPattern,
            trtllm_allreduce_fusion,
        )

        out = torch.empty_like(x)
        trtllm_allreduce_fusion(
            allreduce_in=x,
            world_size=self.world,
            world_rank=self.rank,
            token_num=x.shape[0],
            hidden_dim=x.shape[1],
            workspace_ptrs=self._workspace,
            launch_with_pdl=self.pdl,
            trigger_completion_at_end=True,
            fp32_acc=False,
            pattern_code=AllReduceFusionPattern.kAllReduce,
            use_oneshot=True,
            allreduce_out=out,
            residual_in=None,
            residual_out=None,
            norm_out=None,
            quant_out=None,
            scale_out=None,
            rms_gamma=None,
            rms_eps=None,
            scale_factor=None,
            layout_code=None,
        )
        return out

    def norm_reduce(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        residual: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Fused oneshot allreduce + residual add + RMSNorm (one kernel).

        Returns rmsnorm(allreduce(x) + residual) * gamma. Pass a zero
        residual to get a pure post-reduce norm."""
        from flashinfer.comm import (
            AllReduceFusionPattern,
            trtllm_allreduce_fusion,
        )

        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        trtllm_allreduce_fusion(
            allreduce_in=x,
            world_size=self.world,
            world_rank=self.rank,
            token_num=x.shape[0],
            hidden_dim=x.shape[1],
            workspace_ptrs=self._workspace,
            launch_with_pdl=self.pdl,
            trigger_completion_at_end=True,
            fp32_acc=False,
            pattern_code=AllReduceFusionPattern.kARResidualRMSNorm,
            use_oneshot=True,
            allreduce_out=None,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=None,
            scale_out=None,
            rms_gamma=gamma,
            rms_eps=eps,
            scale_factor=None,
            layout_code=None,
        )
        return norm_out

    def destroy(self) -> None:
        from flashinfer.comm import (
            trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
        )

        try:
            trtllm_destroy_ipc_workspace_for_all_reduce_fusion(self._ipc)
        except Exception:  # noqa: BLE001 - teardown best effort
            pass
