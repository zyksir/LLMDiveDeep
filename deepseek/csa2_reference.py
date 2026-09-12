"""Inspectable CSA2 operators on projected, floating-point tensors.

No weights, quantization, RoPE, paged allocator, MoE or CUDA kernels are
hidden here. Supply Q/K after their respective normalization/RoPE and
dequantization. Compression operates BEFORE normalization/RoPE.
See research.md for the exact boundary and sglang_alignment.md for parity.
"""

from dataclasses import dataclass

import torch

from .config import CANDIDATE_SOURCE, layer_spec


def gated_compress(values: torch.Tensor, gates: torch.Tensor | None, ratio: int):
    """[B,T,D] -> [B,floor(T/r),D], channel-wise softmax over each group.

    Returns pre-RMSNorm latents. An incomplete suffix is deliberately omitted.
    Ratio one has no gate and is the identity at this operator boundary.
    """
    if values.ndim != 3 or ratio < 1:
        raise ValueError("expected values [B,T,D] and positive ratio")
    if ratio == 1:
        return values
    if gates is None or gates.shape != values.shape:
        raise ValueError("ratio > 1 requires gates with the same shape as values")
    b, t, d = values.shape
    n = t // ratio
    v = values[:, : n * ratio].float().reshape(b, n, ratio, d)
    g = gates[:, : n * ratio].float().reshape(b, n, ratio, d)
    return (v * g.softmax(dim=2)).sum(dim=2).to(values.dtype)


class StreamingCompressor:
    """Carries an unfinished group across prefill chunks and decode steps.

    Inputs are already projected values/gates. Construct one per request
    batch and KV owner; do not reuse it for an unrelated prefix.
    """

    def __init__(self, ratio: int):
        if ratio < 1:
            raise ValueError("ratio must be positive")
        self.ratio = ratio
        self.tail_values = None
        self.tail_gates = None
        self.tokens_seen = 0

    def append(self, values, gates=None):
        if values.ndim != 3:
            raise ValueError("expected [B,T,D]")
        if self.ratio > 1 and (gates is None or gates.shape != values.shape):
            raise ValueError("ratio > 1 requires matching gates")
        self.tokens_seen += values.shape[1]
        if self.ratio == 1:
            return values
        if self.tail_values is not None:
            values = torch.cat((self.tail_values, values), dim=1)
            gates = torch.cat((self.tail_gates, gates), dim=1)
        output = gated_compress(values, gates, self.ratio)
        end = output.shape[1] * self.ratio
        self.tail_values = values[:, end:].clone()
        self.tail_gates = gates[:, end:].clone()
        return output


def index_scores(q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor):
    """[B,Q,H_i,D_i], [B,N,D_i], [B,Q,H_i] -> [B,Q,N].

    weights already includes 1/sqrt(D_i * H_i), as in DeepSeek/SGLang.
    ReLU precedes the (possibly signed) head weights. There is no softmax.
    """
    dots = torch.einsum("bqhd,bnd->bqhn", q.float(), keys.float())
    return (dots.relu() * weights.float().unsqueeze(-1)).sum(dim=2)


def causal_index_scores(q, keys, weights, positions, ratio):
    """positions [Q] are absolute zero-based query token positions."""
    if ratio < 1 or positions.ndim != 1 or positions.numel() != q.shape[1]:
        raise ValueError("expected positive ratio and one absolute position per query")
    scores = index_scores(q, keys, weights)
    visible = (positions + 1) // ratio
    ids = torch.arange(keys.shape[1], device=q.device)
    return scores.masked_fill(ids >= visible[:, None], -torch.inf)


def select_topk(scores, k):
    """Fixed-width int32 logical positions, position-sorted, padded with -1.

    Stable score ties prefer smaller positions for reproducible teaching.
    Production top-k kernels need not use the same tie-breaking rule.
    """
    if k < 1:
        raise ValueError("k must be positive")
    n = scores.shape[-1]
    take = min(k, n)
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :take]
    valid = torch.isfinite(scores.gather(-1, ids))
    ids = torch.where(valid, ids, n).sort(dim=-1).values
    ids = torch.where(ids < n, ids, -1).to(torch.int32)
    return torch.nn.functional.pad(ids, (0, k - take), value=-1)


def select_candidates(scores, visible_lengths, topk_blocks, block_size):
    """Block-max pool with newest reachable block pinned; returns [B,Q,C].

    C <= topk_blocks * block_size. Future/padded positions are -1.
    'Block' here is an indexing group, not an allocator page or compressor.
    """
    if topk_blocks < 1 or block_size < 1:
        raise ValueError("candidate dimensions must be positive")
    n = scores.shape[-1]
    if n == 0:
        return torch.empty(
            *scores.shape[:-1], 0, dtype=torch.int32, device=scores.device
        )
    padded = torch.nn.functional.pad(scores, (0, -n % block_size), value=-torch.inf)
    blocks = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    last = (visible_lengths - 1) // block_size
    block_ids = torch.arange(blocks.shape[-1], device=scores.device)
    # Use a finite maximum so select_topk's validity test accepts the pinned block.
    blocks = blocks.masked_fill(block_ids == last, torch.finfo(blocks.dtype).max)
    chosen = select_topk(blocks, min(topk_blocks, blocks.shape[-1])).long()
    slots = torch.arange(block_size, device=scores.device)
    ids = (chosen[..., None] * block_size + slots).flatten(-2)
    valid = (chosen[..., None] >= 0).expand(*chosen.shape, block_size).flatten(-2)
    valid = valid & (ids < n) & (ids < visible_lengths)
    ids = torch.where(valid, ids, n).sort(dim=-1).values
    return torch.where(ids < n, ids, -1).to(torch.int32)


def reindex_candidates(q, keys, weights, candidate_ids, k):
    """Gather BEFORE scoring: the score tensor has width C, not context N.

    This intentionally differs from the official minimal model's dense-score
    then mask implementation, while implementing the report's bounded work.
    """
    b, queries, _, _ = q.shape
    if candidate_ids.shape[:2] != (b, queries):
        raise ValueError("candidates must have the same batch/query axes as q")
    if candidate_ids.numel() and (
        (candidate_ids < -1).any() or (candidate_ids >= keys.shape[1]).any()
    ):
        raise ValueError("invalid candidate index")
    if keys.shape[1] == 0 or candidate_ids.shape[-1] == 0:
        return torch.full((b, queries, k), -1, dtype=torch.int32, device=q.device)
    batch = torch.arange(b, device=q.device)[:, None, None]
    selected = keys[batch, candidate_ids.long().clamp_min(0)]
    dots = torch.einsum("bqhd,bqcd->bqhc", q.float(), selected.float())
    scores = (dots.relu() * weights.float().unsqueeze(-1)).sum(dim=2)
    scores = scores.masked_fill(candidate_ids < 0, -torch.inf)
    local = select_topk(scores, k).long()
    ids = candidate_ids.gather(-1, local.clamp_min(0))
    ids = torch.where(local >= 0, ids, -1)
    n = keys.shape[1]
    ids = torch.where(ids >= 0, ids, n).sort(dim=-1).values
    return torch.where(ids < n, ids, -1).to(torch.int32)


def window_indices(positions, window, batch_size):
    """Logical history indices for a contiguous cache, including the current token."""
    if window < 1:
        raise ValueError("window must be positive")
    ids = (
        positions[:, None] - window + 1 + torch.arange(window, device=positions.device)
    )
    return ids.clamp_min(-1).int().unsqueeze(0).expand(batch_size, -1, -1)


def joint_sparse_attention(q, local_kv, main_kv, local_ids, main_ids, sink):
    """One softmax over local and main entries plus a zero-value sink.

    q [B,Q,H,D]; KV [B,N,D] uses the SAME latent for K and V;
    ids [B,Q,K] use -1 for padding; sink [H] is an unscaled logit.
    Inputs are normalized, post-RoPE and dequantized. Output precedes inverse
    RoPE and the grouped output projection. No cross-branch deduplication.
    """
    b, queries, heads, d = q.shape
    if sink.shape != (heads,):
        raise ValueError("sink must have shape [H]")
    batch = torch.arange(b, device=q.device)[:, None, None]
    selected, valid = [], []
    for cache, ids in ((local_kv, local_ids), (main_kv, main_ids)):
        if ids.shape[:2] != (b, queries):
            raise ValueError("indices must have shape [B,Q,K]")
        if ((ids < -1) | (ids >= cache.shape[1])).any():
            raise ValueError("cache index outside [-1, N)")
        if cache.shape[1] == 0:
            entries = q.new_zeros(b, queries, ids.shape[-1], d)
        else:
            entries = cache[batch, ids.long().clamp_min(0)]
        selected.append(entries)
        valid.append(ids >= 0)
    kv = torch.cat(selected, dim=2).float()
    mask = torch.cat(valid, dim=2)
    logits = torch.einsum("bqhd,bqkd->bqhk", q.float(), kv) * d**-0.5
    logits = logits.masked_fill(~mask[:, :, None, :], -torch.inf)
    null = sink.float().view(1, 1, heads, 1).expand(b, queries, heads, 1)
    probabilities = torch.cat((logits, null), dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bqhk,bqkd->bqhd", probabilities, kv).to(q.dtype)


@dataclass
class RoutedKV:
    main_kv: torch.Tensor
    indices: torch.Tensor
    kv_owner: int
    index_owner: int


class CSA2Router:
    """Explicit per-forward shared state for the released 40-layer schedule.

    The caller owns persistent KV caches. This object owns only routing state
    for a single batch of queries, and requires layers in order. New queries
    must use a new router: reuse is across LAYERS, never across decode times.
    """

    def __init__(self, positions, topk=512, candidate_blocks=2048, block_size=8):
        if topk < 1 or candidate_blocks < 1 or block_size < 1:
            raise ValueError("selection sizes must be positive")
        self.positions = positions
        self.topk = topk
        self.candidate_blocks = candidate_blocks
        self.block_size = block_size
        self.next_layer = 0
        self.main_kv = self.index_k = self.indices = self.candidates = None

    def route(self, layer_id, *, q=None, weights=None, main_kv=None, index_k=None):
        if layer_id != self.next_layer:
            raise ValueError(
                f"expected layer {self.next_layer}; construct a new router for new queries"
            )
        spec = layer_spec(layer_id)
        if spec.mode == "SWA":
            self.next_layer += 1
            return None
        if spec.mode == "Full":
            if (
                main_kv is None
                or index_k is None
                or main_kv.shape[:2] != index_k.shape[:2]
            ):
                raise ValueError(
                    "Full requires corresponding main KV and index K caches"
                )
            self.main_kv, self.index_k = main_kv, index_k
        elif main_kv is not None or index_k is not None:
            raise ValueError("only a Full layer may replace the shared caches")
        if spec.mode != "Reuse":
            if q is None or weights is None:
                raise ValueError(
                    "Full/Reindex requires current layer indexer Q and weights"
                )
            if spec.mode == "Reindex":
                self.indices = reindex_candidates(
                    q, self.index_k, weights, self.candidates, self.topk
                )
            else:
                scores = causal_index_scores(
                    q, self.index_k, weights, self.positions, spec.ratio
                )
                self.indices = select_topk(scores, self.topk)
                if layer_id == CANDIDATE_SOURCE:
                    visible = ((self.positions + 1) // spec.ratio)[:, None]
                    self.candidates = select_candidates(
                        scores, visible, self.candidate_blocks, self.block_size
                    )
        self.next_layer += 1
        return RoutedKV(self.main_kv, self.indices, spec.kv_owner, spec.index_owner)
