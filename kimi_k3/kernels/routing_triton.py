from __future__ import annotations

import torch
import triton
import triton.language as tl

NUM_EXPERTS = 896
TOP_K = 16
ROUTED_SCALING = 1.0
_BLOCK = 1024
_NUM_WARPS = 4


@triton.jit
def _route_kernel(
    logits_ptr,
    bias_ptr,
    ids_ptr,
    scales_ptr,
    logits_stride,
    E: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < E
    logits = tl.load(
        logits_ptr + row * logits_stride + offs, mask=mask, other=0.0
    )
    scores = tl.sigmoid(logits.to(tl.float32))
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    biased = tl.where(mask, scores + bias, -float("inf"))

    bits = biased.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    key32 = tl.where(bits >> 31 != 0, (~bits) & 0xFFFFFFFF, bits | 0x80000000)
    keys = (key32 << 10) | (BLOCK - 1 - offs).to(tl.int64)

    lanes = tl.arange(0, K)
    picks = tl.zeros((K,), dtype=tl.int32)
    for k in tl.static_range(K):
        best = tl.max(keys, axis=0)
        pick = (BLOCK - 1 - (best & (BLOCK - 1))).to(tl.int32)
        picks = tl.where(lanes == k, pick, picks)
        keys = tl.where(offs == pick, -1 << 62, keys)

    picked_scores = tl.sum(
        tl.where(offs[None, :] == picks[:, None], scores[None, :], 0.0), axis=1
    )
    score_sum = 0.0
    for k in tl.static_range(K):
        score_sum += tl.sum(tl.where(lanes == k, picked_scores, 0.0), axis=0)

    scales = picked_scores * (SCALE / score_sum)
    tl.store(ids_ptr + row * K + lanes, picks)
    tl.store(scales_ptr + row * K + lanes, scales)


def route_triton(
    logits: torch.Tensor,
    gate_bias: torch.Tensor,
    *,
    fmt: str = "trtllm_gen",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route with the shipped Kimi-K3 Triton policy."""
    assert fmt in ("fused", "trtllm_gen"), fmt
    batch = logits.shape[0]
    scale_dtype = torch.float32 if fmt == "fused" else torch.bfloat16
    ids = torch.empty(batch, TOP_K, device=logits.device, dtype=torch.int32)
    scales = torch.empty(batch, TOP_K, device=logits.device, dtype=scale_dtype)
    _route_kernel[(batch,)](
        logits,
        gate_bias,
        ids,
        scales,
        logits.stride(0),
        E=NUM_EXPERTS,
        K=TOP_K,
        SCALE=ROUTED_SCALING,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return ids, scales


def bench(
    *,
    batch: int = 16,
    fmt: str = "trtllm_gen",
    iters: int = 100,
    repeats: int = 5,
) -> dict[str, float]:
    """Check correctness and graph latency against PyTorch routing."""
    from ._bench import check_regression, graph_time_us, routing_reference

    torch.manual_seed(0)
    logits = torch.randn(
        batch, NUM_EXPERTS + 32, device="cuda", dtype=torch.bfloat16
    )[:, :NUM_EXPERTS]
    bias = torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.float32)

    expected = routing_reference(
        logits,
        bias,
        top_k=TOP_K,
        scale=ROUTED_SCALING,
        fmt=fmt,
    )
    actual = route_triton(logits, bias, fmt=fmt)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-3, atol=2e-3)

    reference_us = graph_time_us(
        lambda: routing_reference(
            logits,
            bias,
            top_k=TOP_K,
            scale=ROUTED_SCALING,
            fmt=fmt,
        ),
        iters=iters,
        repeats=repeats,
    )
    kernel_us = graph_time_us(
        lambda: route_triton(logits, bias, fmt=fmt),
        iters=iters,
        repeats=repeats,
    )
    check_regression(reference_us, kernel_us)
    return {"reference_us": reference_us, "kernel_us": kernel_us}


route_for_kimi_k3 = route_triton

__all__ = ["bench", "route_for_kimi_k3", "route_triton"]
