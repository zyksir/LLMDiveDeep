"""Attention using PyTorch matmul/softmax and an educational online softmax.

Contract: Q [B,Q,Hq,D], K/V [B,K,Hkv,D], Hq % Hkv == 0, Q <= K.
Queries are the final Q positions of the KV sequence (bottom-right causal).
No projections, RoPE, cache writes, output norm, dropout or backward benchmark.
"""

import torch


def validate_qkv(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("expected BSHD Q/K/V, with matching K and V shapes")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K/V batch and head dimensions must agree")
    if min(*q.shape, *k.shape) < 1 or q.shape[1] > k.shape[1]:
        raise ValueError("positive dimensions and Q <= K are required")
    if q.shape[2] % k.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    if not (q.dtype == k.dtype == v.dtype and q.device == k.device == v.device):
        raise ValueError("Q/K/V dtype and device must match")


def attention_mask(q_len, kv_len, *, device, causal=True, window=None):
    """[Q,K] boolean mask: True means KEEP. W includes the current token."""
    if not 0 < q_len <= kv_len:
        raise ValueError("require 0 < Q <= K")
    if window is not None and (window < 1 or not causal):
        raise ValueError("window must be positive and requires causal=True")
    pos = torch.arange(q_len, device=device)[:, None] + kv_len - q_len
    keys = torch.arange(kv_len, device=device)[None, :]
    keep = (
        keys <= pos
        if causal
        else torch.ones((q_len, kv_len), device=device, dtype=torch.bool)
    )
    if window is not None:
        keep = keep & (keys >= pos - window + 1)
    return keep


def expanded_bhsd(q, k, v):
    """Readable GQA expansion; optimized kernels should share KV instead."""
    repeat = q.shape[2] // k.shape[2]
    return (
        q.transpose(1, 2).float(),
        k.repeat_interleave(repeat, dim=2).transpose(1, 2).float(),
        v.repeat_interleave(repeat, dim=2).transpose(1, 2).float(),
    )


def matmul_attention(q, k, v, *, causal=True, window=None, mask=None, scale=None):
    """Three visible operations: QK^T, row softmax, PV. Accumulates in FP32.

    This deliberately materializes [B,H,Q,K]; it is the math oracle, not a
    FlashAttention kernel. The optional mask lets a harness prepare masks once.
    """
    validate_qkv(q, k, v)
    qh, kh, vh = expanded_bhsd(q, k, v)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if mask is None:
        mask = attention_mask(
            q.shape[1], k.shape[1], device=q.device, causal=causal, window=window
        )
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, vh)
    return output.transpose(1, 2).to(q.dtype)


def online_attention(q, k, v, *, causal=True, window=None, tile=128, scale=None):
    """Tile the exact online-softmax recurrence in ordinary PyTorch.

    No [Q,K] score matrix is retained. Python loops, separate torch launches,
    GQA expansion and FP32 matmuls make this a teaching reference, NOT FA2/3/4.
    With a window, entire out-of-window key tiles are omitted before matmul.
    """
    validate_qkv(q, k, v)
    if tile < 1 or (window is not None and (window < 1 or not causal)):
        raise ValueError("positive tile/window required; window must be causal")
    qh, kh, vh = expanded_bhsd(q, k, v)
    b, h, nq, d = qh.shape
    nk = kh.shape[2]
    scale = d**-0.5 if scale is None else scale
    pieces = []
    for begin in range(0, nq, tile):
        end = min(begin + tile, nq)
        query = qh[:, :, begin:end]
        absolute = torch.arange(begin, end, device=q.device) + nk - nq
        m = torch.full((b, h, end - begin, 1), -torch.inf, device=q.device)
        denominator = torch.zeros_like(m)
        numerator = torch.zeros((b, h, end - begin, d), device=q.device)
        lower = max(0, begin + nk - nq - window + 1) if window else 0
        upper = end + nk - nq if causal else nk
        for kb in range(lower, upper, tile):
            ke = min(kb + tile, upper)
            key_pos = torch.arange(kb, ke, device=q.device)
            keep = (
                (key_pos[None] <= absolute[:, None])
                if causal
                else torch.ones(
                    (end - begin, ke - kb), dtype=torch.bool, device=q.device
                )
            )
            if window:
                keep = keep & (key_pos[None] >= absolute[:, None] - window + 1)
            scores = (query @ kh[:, :, kb:ke].transpose(-2, -1)) * scale
            scores = scores.masked_fill(~keep, -torch.inf)
            new_m = torch.maximum(m, scores.amax(-1, keepdim=True))
            # Some rows have not reached their first valid tile yet.
            safe_m = torch.where(torch.isfinite(new_m), new_m, 0.0)
            alpha = torch.exp(m - safe_m)
            p = torch.exp(scores - safe_m)
            numerator = alpha * numerator + p @ vh[:, :, kb:ke]
            denominator = alpha * denominator + p.sum(-1, keepdim=True)
            m = new_m
        pieces.append(
            numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
        )
    return torch.cat(pieces, dim=2).transpose(1, 2).to(q.dtype)
