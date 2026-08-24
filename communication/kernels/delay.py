"""Rank-skew injection kernel for serving-realistic layer benches.

In serving, each MoE block follows attention whose per-rank duration
varies (dummy-weight KDA/MLA variance, host jitter), so ranks reach the
MoE sync points skewed; a path with 3 sync points pays that skew up to
3x per layer, a single-sync path pays it once. Single-layer benches
have no attention and therefore no skew — this kernel injects it:
a busy-wait of (ns_base + ns_per_rank * rank) before each iteration,
inside the CUDA graph.
"""
from __future__ import annotations

import torch

_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__global__ void delay_kernel(int64_t ns) {
    if (ns <= 0) return;
    int64_t start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    now = start;
    while (now - start < ns)
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
}

void run_delay(int64_t ns) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    delay_kernel<<<1, 1, 0, stream>>>(ns);
}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline
        _MODULE = load_inline(
            name="k3_delay",
            cpp_sources=("#include <torch/extension.h>\n"
                         "void run_delay(int64_t);"),
            cuda_sources=_SRC, extra_cuda_cflags=["-O3"],
            with_cuda=True, functions=["run_delay"])
    return _MODULE


def delay_ns(ns: int) -> None:
    """Enqueue a busy-wait of `ns` nanoseconds on the current stream."""
    _module().run_delay(int(ns))
