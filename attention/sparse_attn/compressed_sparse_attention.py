"""Readable CSA operators, not a DeepSeek layer or a production cache manager.

All attention inputs are already projected/normalized/position-encoded latent
vectors. Compression below stops BEFORE RMSNorm/RoPE/quantization. See
compressed_sparse_attention.md for these deliberately separate boundaries.
"""

import torch


def gated_compress(values, logits, ratio, *, overlap=False, ape=None):
    """Channel-wise gated pooling of complete groups, in FP32.

    Input [B,T,D] (non-overlap) or [B,T,2D] (V4 overlap).
    Overlap entry j combines previous group's first half and current group's
    second half. The unavailable previous group at j=0 is masked, not averaged.
    `ape`, if supplied, has shape [ratio,input_channels]. No projections here.
    """
    if values.ndim != 3 or values.shape != logits.shape or ratio < 1:
        raise ValueError("expected equal [B,T,C] values/logits and positive ratio")
    b, t, c = values.shape
    if overlap and c % 2:
        raise ValueError("overlap requires an even channel dimension")
    n, d = t // ratio, c // (2 if overlap else 1)
    if n == 0:
        return values.new_empty(b, 0, d, dtype=torch.float32)
    x = values[:, : n * ratio].float().reshape(b, n, ratio, c)
    g = logits[:, : n * ratio].float().reshape(b, n, ratio, c)
    if ape is not None:
        if ape.shape != (ratio, c):
            raise ValueError("APE must have shape [ratio,input_channels]")
        g = g + ape.float()
    if overlap:
        prev_x = torch.cat((torch.zeros_like(x[:, :1, :, :d]), x[:, :-1, :, :d]), dim=1)
        prev_g = torch.cat(
            (torch.full_like(g[:, :1, :, :d], -torch.inf), g[:, :-1, :, :d]), dim=1
        )
        x = torch.cat((prev_x, x[..., d:]), dim=2)
        g = torch.cat((prev_g, g[..., d:]), dim=2)
    return (torch.softmax(g, dim=2) * x).sum(dim=2)


def indexer_topk(query, keys, weights, positions, *, ratio, topk):
    """Unquantized learned-indexer score and causal top-k reference.

    query [B,Q,Hi,Di], keys [B,N,Di], weights [B,Q,Hi] already include
    the model's scale. positions [Q] are absolute zero-based token positions.
    Result [B,Q,min(topk,N)] has compressed-cache IDs or -1 for unavailable
    groups. Weights may be signed. Ties need not match another top-k kernel.
    """
    if ratio < 1 or topk < 1 or keys.shape[1] < 1:
        raise ValueError("ratio, topk, and compressed key count must be positive")
    scores = torch.einsum("bqhd,bnd->bqhn", query.float(), keys.float()).relu()
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2)
    end = (torch.arange(keys.shape[1], device=keys.device) + 1) * ratio - 1
    scores = scores.masked_fill(end[None, :] > positions[:, None], -torch.inf)
    selected_scores, ids = scores.topk(min(topk, keys.shape[1]), dim=-1)
    return ids.masked_fill(~torch.isfinite(selected_scores), -1).to(torch.int32)


def combined_selection(raw_length, positions, global_ids, *, window, ratio):
    """Build [local | global] IDs in conceptual [raw_cache | compressed_cache].

    global_ids [B,Q,K] is shared across main attention heads. Causality is
    enforced AGAIN after selection; an unfinished group must never be visible.
    Padding is -1; local entries precede compressed entries. No deduplication
    across representations: a pooled summary and a raw token are different KV.
    """
    if window < 1 or ratio < 1 or raw_length < 1:
        raise ValueError("positive window, ratio, and raw length required")
    if positions.ndim != 1 or positions.numel() != global_ids.shape[1]:
        raise ValueError("positions must have one absolute position per query")
    local = (
        positions[:, None] - window + 1 + torch.arange(window, device=positions.device)
    )
    local = local.masked_fill((local < 0) | (local >= raw_length), -1)
    local = local.unsqueeze(0).expand(global_ids.shape[0], -1, -1)
    visible = (global_ids >= 0) & (
        (global_ids + 1) * ratio - 1 <= positions[None, :, None]
    )
    visible &= global_ids < raw_length // ratio
    global_part = torch.where(visible, global_ids + raw_length, -1)
    return torch.cat((local, global_part), dim=-1).to(torch.int32)


def gather_attention(q, cache, indices, *, sink=None, scale=None):
    """Sparse MQA with shared K=V: q [B,Q,H,D], cache [B,N,D].

    indices [B,Q,L], -1 padding. A finite per-head sink logit has zero value
    and participates in the SAME softmax as local/global tokens. The gathered
    temporary [B,Q,L,D] is intentional: this is a readable, slow reference.
    Prepared backend adapters validate indices before timing this function.
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    selected = cache[batch, indices.clamp_min(0).long()].float()
    scores = torch.einsum("bqhd,bqld->bqhl", q.float(), selected) * scale
    scores = scores.masked_fill(indices[:, :, None, :] < 0, -torch.inf)
    if sink is not None:
        sink_scores = sink.float()[None, None, :, None].expand(*scores.shape[:-1], 1)
        scores = torch.cat((scores, sink_scores), dim=-1)
    probs = torch.softmax(scores, dim=-1)
    # No visible entries and no sink: choose output zero, not NaN.
    probs = torch.nan_to_num(probs)[..., : indices.shape[-1]]
    return torch.einsum("bqhl,bqld->bqhd", probs, selected).to(q.dtype)


def selection_mask(indices, cache_length):
    """Independent dense oracle mask, built outside the core's timed region."""
    ids = torch.arange(cache_length, device=indices.device)
    mask = torch.zeros(
        *indices.shape[:2], cache_length, dtype=torch.bool, device=indices.device
    )
    # Avoid an enormous [B,Q,L,N] equality tensor in this teaching oracle.
    for slot in range(indices.shape[-1]):
        mask |= indices[..., slot, None] == ids
    return mask


def dense_selected_attention(q, cache, mask, *, sink=None, scale=None):
    """Full QK^T with the same selected support; independent gather oracle."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.einsum("bqhd,bnd->bqhn", q.float(), cache.float()) * scale
    scores = scores.masked_fill(~mask[:, :, None, :], -torch.inf)
    if sink is not None:
        scores = torch.cat(
            (scores, sink.float()[None, None, :, None].expand(*scores.shape[:-1], 1)),
            dim=-1,
        )
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1))[..., : cache.shape[1]]
    return torch.einsum("bqhn,bnd->bqhd", probs, cache.float()).to(q.dtype)
