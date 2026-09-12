"""RMSNorm entry point used by the Kimi-K3 MoE layer."""

from __future__ import annotations

import torch


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply the same fused RMSNorm kernel as TRT-LLM's module."""
    from flashinfer.norm import rmsnorm as flashinfer_rmsnorm

    return flashinfer_rmsnorm(x.contiguous(), weight, eps)
