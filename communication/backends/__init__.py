"""Backend implementations behind ``communication.Collectives``.

One module per backend; ``collective.py`` only dispatches:

    nccl_backend             plain torch.distributed (unbounded fallback)
    nccl_symm_backend        NCCL over symmetric-memory staging
    torch_symm_backend       torch symmetric-memory (multimem / 1shot /
                             2shot) + the graphed fused schedules
    torch_lc_backend         torch's low-contention symm-mem ops
    b10_copy_engine_backend  our LowContentionComm movers (dma / sm)
    flashinfer_backend       FlashInfer Lamport AR (+ fused AR+norm)
    trt_backend              TRT-LLM custom allreduce (+ fused AR+norm)
    sglang_backend           sglang's vendored Kimi-K3 fused MNNVL
                             kernels: push/pull AR (+ fused RMSNorm),
                             GEMM+AR; MoE-finalize AR and GEMM+AG as
                             separate entry points (SM100/103 only)
    col_quant_backend        column AG with fused MXFP8 quant
    b10_multimem_backend     custom NVLS multimem collectives
"""

from . import (  # noqa: F401
    b10_copy_engine_backend,
    b10_multimem_backend,
    col_quant_backend,
    flashinfer_backend,
    nccl_backend,
    nccl_symm_backend,
    sglang_backend,
    torch_lc_backend,
    torch_symm_backend,
    trt_backend,
)

from ..context import Ctx  # noqa: F401
