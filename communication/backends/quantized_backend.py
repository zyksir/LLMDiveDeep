"""NVIDIA NVLink INT8/FP8 two-shot AllReduce backend.

Kernel provenance: vLLM PR #39783 at c77c662a, based on Meta Kraken.
"""

import torch

from ..context import Ctx
from ..kernels.quantized_allreduce import (
    DEFAULT_GROUP_SIZE,
    QuantizedAllReduceState,
    two_shot_quantized_allreduce,
)


class QuantizedBackend:
    def __init__(self, ctx: Ctx, max_numel: int) -> None:
        self.ctx = ctx
        self.max_numel = max_numel
        self._state: QuantizedAllReduceState | None = None
        self._output: torch.Tensor | None = None

    def _ensure_buffers(self) -> None:
        if self._state is not None:
            return
        device = torch.device("cuda", self.ctx.rank)
        self._state = QuantizedAllReduceState(
            self.max_numel,
            DEFAULT_GROUP_SIZE,
            device,
            self.ctx.group,
        )
        self._output = torch.empty(
            self.max_numel,
            dtype=torch.bfloat16,
            device=device,
        )

    def supports(self, x: torch.Tensor) -> bool:
        return (
            x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.numel() % 8 == 0
            and 8192 <= x.numel() <= self.max_numel
        )

    def all_reduce(self, x: torch.Tensor, *, use_fp8: bool) -> torch.Tensor:
        if not self.supports(x):
            raise ValueError(
                "quantized AllReduce requires contiguous BF16, numel "
                f"divisible by 8, and 8192 <= numel <= {self.max_numel}"
            )
        self._ensure_buffers()
        assert self._state is not None and self._output is not None
        return two_shot_quantized_allreduce(
            x,
            self._output,
            self._state,
            use_fp8=use_fp8,
        )
