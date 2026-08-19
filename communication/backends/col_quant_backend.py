"""Backend wrapper for column all-gather and fused quantization."""

from __future__ import annotations

import torch

from ..kernels.col_quant import (
    TunedAllGather,
    build_tuned_all_gather,
    nccl_all_gather_cols as _nccl_all_gather_cols,
)


def nccl_all_gather_cols(x: torch.Tensor) -> torch.Tensor:
    return _nccl_all_gather_cols(x)


def build_col_quant_all_gather(
    rank: int,
    world: int,
    max_rows: int,
    max_output_columns: int,
) -> TunedAllGather:
    return build_tuned_all_gather(
        rank,
        world,
        max_rows,
        max_output_columns,
    )
