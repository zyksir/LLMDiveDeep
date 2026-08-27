"""Standalone DeepSeek-V4 attention layers: CSA and HCA.

Paper (arXiv 2606.19348 §2.3) and HuggingFace `modeling_deepseek_v4.py`.
This is the *layer* contract, not the indexer-only stub in `DSA.py`.

Each decoder block is one of:

* **CSA** (`compressed_sparse_attention`, m=4): overlapping softmax-gated
  compressor + lightning indexer that keeps `index_topk` compressed entries
  per query. Core attention softmaxes over [SWA window | selected compressed].
* **HCA** (`heavily_compressed_attention`, m'=128): non-overlapping compressor,
  **no indexer**. Core attention softmaxes over [SWA window | every compressed
  entry whose source window has closed].
* **SWA**: sliding-window branch only (the first two Flash layers).
* **dense**: same Q/KV/o_proj backbone, full causal MQA (cost baseline).

Shared backbone (paper §2.3.3): shared K=V MQA, partial interleaved RoPE on
the trailing `rope_head_dim` of each head, per-head attention sink, grouped
low-rank output projection. mHC / MoE are out of scope — this is one
attention block.

Prefill takes `[B, S, hidden]`. Decode takes one new token plus a synthetic
history cache (SWA window + already-compressed entries) so long-context cost
can be measured without replaying a 1 M-token prefill.

No KV-cache object, no paged layout, no production kernels. Eager math, so
the compressor / indexer / softmax boundary stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LayerKind = Literal["csa", "hca", "swa", "dense"]


# ---------------------------------------------------------------------------
# Specs (DeepSeek-V4-Flash / Pro production shapes, plus a tiny bench spec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dsv4Spec:
    """Shape-only configuration for one V4 attention block."""

    name: str
    hidden_size: int
    num_heads: int
    head_dim: int
    q_lora_rank: int
    rope_head_dim: int
    o_groups: int
    o_lora_rank: int
    sliding_window: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    compress_rate_csa: int = 4
    compress_rate_hca: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0

    def __post_init__(self) -> None:
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        if self.rope_head_dim % 2:
            raise ValueError("rope_head_dim must be even")
        if self.rope_head_dim > self.head_dim:
            raise ValueError("rope_head_dim cannot exceed head_dim")
        if (self.num_heads * self.head_dim) % self.o_groups:
            raise ValueError("o_groups must divide num_heads * head_dim")
        if self.index_head_dim % 2:
            raise ValueError("index_head_dim must be even")
        if self.index_head_dim < self.rope_head_dim:
            raise ValueError("index_head_dim must be >= rope_head_dim")


SPECS: Dict[str, Dsv4Spec] = {
    # Production DeepSeek-V4-Flash (paper §4.2.1). CSA top-k = 512.
    "flash": Dsv4Spec(
        name="V4-Flash",
        hidden_size=4096,
        num_heads=64,
        head_dim=512,
        q_lora_rank=1024,
        rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        sliding_window=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
    ),
    # Production DeepSeek-V4-Pro. CSA top-k = 1024; twice the Q heads.
    "pro": Dsv4Spec(
        name="V4-Pro",
        hidden_size=7168,
        num_heads=128,
        head_dim=512,
        q_lora_rank=1536,
        rope_head_dim=64,
        o_groups=16,
        o_lora_rank=1024,
        sliding_window=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
    ),
    # Ratio-preserving shrink so prefill sweeps fit in one GPU. Compression
    # rates, SWA, and indexer MQA layout match Flash; width does not.
    "tiny": Dsv4Spec(
        name="V4-tiny",
        hidden_size=1024,
        num_heads=8,
        head_dim=128,
        q_lora_rank=256,
        rope_head_dim=32,
        o_groups=4,
        o_lora_rank=256,
        sliding_window=128,
        index_n_heads=16,
        index_head_dim=64,
        index_topk=64,
    ),
}


# ---------------------------------------------------------------------------
# Small ops (RMSNorm, grouped o_proj, interleaved partial RoPE)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype)


class GroupedLinear(nn.Module):
    """Block-diagonal linear used by V4's grouped output projection."""

    def __init__(
        self,
        in_features_per_group: int,
        out_features_per_group: int,
        n_groups: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.weight = nn.Parameter(
            torch.empty(
                n_groups * out_features_per_group,
                in_features_per_group,
                dtype=dtype,
            )
        )
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_groups, in_per_group] → [..., n_groups, out_per_group]
        leading = x.shape[:-2]
        hidden_dim = x.shape[-1]
        weight = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        grouped = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        out = torch.bmm(grouped, weight).transpose(0, 1)
        return out.reshape(*leading, self.n_groups, -1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved RoPE on the trailing `rope_head_dim` channels of `x`.

    `x` is `[B, S, D]` or `[B, S, H, D]`. `cos`/`sin` are `[B, S, rope/2]`
    (one θ per pair) and broadcast over the head axis.
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"apply_partial_rope expects 3D or 4D x, got {x.ndim}D")
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    if x.ndim == 4:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    rope_dim = cos.shape[-1]
    if rope_dim > x.shape[-1]:
        raise ValueError(f"rope dim {rope_dim} exceeds last dim {x.shape[-1]}")
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (rope.float() * cos + _rotate_half(rope).float() * sin).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


def rope_cos_sin(
    positions: torch.Tensor, rope_dim: int, theta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`positions` is [B, S] int. Returns cos/sin of shape [B, S, rope_dim/2]."""
    half = rope_dim // 2
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, half, device=positions.device, dtype=torch.float32)
            / half
        )
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq
    return freqs.cos(), freqs.sin()


def softmax_pool(values: torch.Tensor, logits: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax-gated convex combination along `dim`, softmax in fp32."""
    weights = logits.softmax(dim=dim, dtype=torch.float32).to(values.dtype)
    return (values * weights).sum(dim=dim)


def sliding_window_keys(kv: torch.Tensor, window: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal sliding-window gather.

    kv: [B, S, D] → keys [B, S, window, D] and valid [B, S, window].
    Query t sees source tokens [t-window+1, t], left-padded with invalid slots.
    """
    batch, seq_len, dim = kv.shape
    if window < 1:
        raise ValueError("sliding_window must be >= 1")
    padded = F.pad(kv, (0, 0, window - 1, 0))
    keys = padded.unfold(1, window, 1).permute(0, 1, 3, 2).contiguous()
    token_index = torch.arange(seq_len, device=kv.device)
    source = token_index.unsqueeze(1) - (window - 1) + torch.arange(
        window, device=kv.device
    )
    valid = source >= 0
    return keys, valid.unsqueeze(0).expand(batch, -1, -1)


# ---------------------------------------------------------------------------
# Compressors
# ---------------------------------------------------------------------------


class HCACompressor(nn.Module):
    """Non-overlapping softmax-gated pool, one entry per `m'` source tokens.

    Paper eqs. 20–23. Window `w` covers tokens `[w m', (w+1) m')` and becomes
    visible to query `t` once `t >= (w+1)*m' - 1`, i.e. `w < (t+1)//m'`.
    """

    def __init__(self, spec: Dsv4Spec, dtype: torch.dtype):
        super().__init__()
        self.spec = spec
        self.rate = spec.compress_rate_hca
        self.kv_proj = nn.Linear(spec.hidden_size, spec.head_dim, bias=False, dtype=dtype)
        self.gate_proj = nn.Linear(spec.hidden_size, spec.head_dim, bias=False, dtype=dtype)
        self.position_bias = nn.Parameter(
            torch.zeros(self.rate, spec.head_dim, dtype=dtype)
        )
        self.kv_norm = RMSNorm(spec.head_dim, spec.rms_norm_eps, dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        usable = (seq_len // self.rate) * self.rate
        if usable == 0:
            return hidden_states.new_zeros(batch, 0, self.spec.head_dim)
        kv = self.kv_proj(hidden_states[:, :usable])
        gate = self.gate_proj(hidden_states[:, :usable])
        n_windows = usable // self.rate
        kv = kv.view(batch, n_windows, self.rate, -1)
        gate = gate.view(batch, n_windows, self.rate, -1) + self.position_bias
        compressed = self.kv_norm(softmax_pool(kv, gate, dim=2))
        positions = (
            torch.arange(n_windows, device=hidden_states.device) * self.rate
        ).unsqueeze(0).expand(batch, -1)
        cos, sin = rope_cos_sin(
            positions, self.spec.rope_head_dim, self.spec.compress_rope_theta
        )
        return apply_partial_rope(compressed, cos, sin)

    def causal_valid(self, seq_len: int, n_windows: int, device: torch.device) -> torch.Tensor:
        """[S, T] bool: query t may see compressed entry w."""
        query = torch.arange(seq_len, device=device)
        entry = torch.arange(n_windows, device=device)
        return entry.unsqueeze(0) < ((query + 1) // self.rate).unsqueeze(1)


class CSACompressor(nn.Module):
    """Overlapping Ca/Cb softmax-gated pool, one entry per `m` source tokens.

    Each token emits two series packed in `2 * head_dim`: Ca (contribution to
    the *next* window) and Cb (contribution to the *current* window). Entry
    `w` pools window `w-1`'s Ca with window `w`'s Cb over `2m` slots. Window 0
    has no previous Ca, so those slots stay zero / `-inf` (softmax weight 0).
    """

    def __init__(self, spec: Dsv4Spec, dtype: torch.dtype, dim: Optional[int] = None):
        super().__init__()
        self.spec = spec
        self.rate = spec.compress_rate_csa
        self.dim = spec.head_dim if dim is None else dim
        self.kv_proj = nn.Linear(spec.hidden_size, 2 * self.dim, bias=False, dtype=dtype)
        self.gate_proj = nn.Linear(spec.hidden_size, 2 * self.dim, bias=False, dtype=dtype)
        self.position_bias = nn.Parameter(
            torch.zeros(self.rate, 2 * self.dim, dtype=dtype)
        )
        self.kv_norm = RMSNorm(self.dim, spec.rms_norm_eps, dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        usable = (seq_len // self.rate) * self.rate
        if usable == 0:
            return hidden_states.new_zeros(batch, 0, self.dim)
        kv = self.kv_proj(hidden_states[:, :usable])
        gate = self.gate_proj(hidden_states[:, :usable])
        n_windows = usable // self.rate
        kv = kv.view(batch, n_windows, self.rate, -1)
        gate = gate.view(batch, n_windows, self.rate, -1) + self.position_bias
        overlapped_kv, overlapped_gate = self._overlap(kv, gate)
        compressed = self.kv_norm(softmax_pool(overlapped_kv, overlapped_gate, dim=2))
        positions = (
            torch.arange(n_windows, device=hidden_states.device) * self.rate
        ).unsqueeze(0).expand(batch, -1)
        if self.dim < self.spec.rope_head_dim:
            raise ValueError(
                f"compressor dim {self.dim} < rope_head_dim {self.spec.rope_head_dim}"
            )
        cos, sin = rope_cos_sin(
            positions, self.spec.rope_head_dim, self.spec.compress_rope_theta
        )
        return apply_partial_rope(compressed, cos, sin)

    def _overlap(
        self, kv: torch.Tensor, gate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, n_windows, ratio, _ = kv.shape
        dim = self.dim
        overlapped_kv = kv.new_zeros(batch, n_windows, 2 * ratio, dim)
        overlapped_gate = gate.new_full(
            (batch, n_windows, 2 * ratio, dim), float("-inf")
        )
        overlapped_kv[:, :, ratio:] = kv[..., dim:]
        overlapped_gate[:, :, ratio:] = gate[..., dim:]
        if n_windows > 1:
            overlapped_kv[:, 1:, :ratio] = kv[:, :-1, :, :dim]
            overlapped_gate[:, 1:, :ratio] = gate[:, :-1, :, :dim]
        return overlapped_kv, overlapped_gate

    def causal_valid(self, seq_len: int, n_windows: int, device: torch.device) -> torch.Tensor:
        query = torch.arange(seq_len, device=device)
        entry = torch.arange(n_windows, device=device)
        return entry.unsqueeze(0) < ((query + 1) // self.rate).unsqueeze(1)


class LightningIndexer(nn.Module):
    """CSA lightning indexer: compress at `index_head_dim`, ReLU-gated MQA, top-k.

    Score for query t vs compressed entry s (paper eqs. 13–17):

        I_{t,s} = sum_h w_{t,h} * ReLU(<q_{t,h}, k_s^{IComp}>)
    """

    def __init__(self, spec: Dsv4Spec, dtype: torch.dtype):
        super().__init__()
        self.spec = spec
        self.compressor = CSACompressor(spec, dtype, dim=spec.index_head_dim)
        self.q_b_proj = nn.Linear(
            spec.q_lora_rank,
            spec.index_n_heads * spec.index_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.weights_proj = nn.Linear(
            spec.hidden_size, spec.index_n_heads, bias=False, dtype=dtype
        )
        self.softmax_scale = spec.index_head_dim ** -0.5
        self.weights_scale = spec.index_n_heads ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        compressed_kv: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (topk_indices [B, S, k], compressed_kv [B, T, D_idx])."""
        batch, seq_len, _ = hidden_states.shape
        if compressed_kv is None:
            compressed_kv = self.compressor(hidden_states)
        n_windows = compressed_kv.shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=hidden_states.device
            ).unsqueeze(0).expand(batch, -1)
        cos, sin = rope_cos_sin(
            position_ids, self.spec.rope_head_dim, self.spec.compress_rope_theta
        )
        query = self.q_b_proj(q_lora).view(
            batch, seq_len, self.spec.index_n_heads, self.spec.index_head_dim
        )
        query = apply_partial_rope(query, cos, sin)
        scores = torch.einsum("bshd,btd->bsht", query.float(), compressed_kv.float())
        scores = F.relu(scores) * self.softmax_scale
        gates = self.weights_proj(hidden_states).float() * self.weights_scale
        logits = (scores * gates.unsqueeze(-1)).sum(dim=2)
        if n_windows == 0:
            empty = logits.new_zeros(batch, seq_len, 0, dtype=torch.long)
            return empty, compressed_kv
        causal_threshold = (position_ids + 1) // self.spec.compress_rate_csa
        entry = torch.arange(n_windows, device=hidden_states.device)
        future = entry.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
        logits = logits.masked_fill(future, float("-inf"))
        top_k = min(self.spec.index_topk, n_windows)
        indices = logits.topk(top_k, dim=-1).indices
        invalid = indices >= causal_threshold.unsqueeze(-1)
        indices = torch.where(invalid, torch.full_like(indices, -1), indices)
        return indices, compressed_kv


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------


@dataclass
class Dsv4Output:
    hidden_states: torch.Tensor
    topk_indices: Optional[torch.Tensor]
    n_compressed: int
    n_core_keys: int


@dataclass
class DecodeCache:
    """Already-compressed history for a T=1 decode step at context length S."""

    seq_len: int
    swa_kv: torch.Tensor  # [B, W, head_dim]
    compressed_kv: torch.Tensor  # [B, T, head_dim]
    indexer_kv: Optional[torch.Tensor]  # [B, T, index_head_dim], CSA only


class Dsv4AttentionLayer(nn.Module):
    """One V4 attention block. `kind` selects CSA / HCA / SWA / dense."""

    def __init__(
        self,
        spec: Dsv4Spec,
        kind: LayerKind,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if kind not in ("csa", "hca", "swa", "dense"):
            raise ValueError(f"unknown layer kind {kind!r}")
        self.spec = spec
        self.kind = kind
        self.dtype = dtype
        self.q_a_proj = nn.Linear(spec.hidden_size, spec.q_lora_rank, bias=False, dtype=dtype)
        self.q_a_norm = RMSNorm(spec.q_lora_rank, spec.rms_norm_eps, dtype)
        self.q_b_proj = nn.Linear(
            spec.q_lora_rank, spec.num_heads * spec.head_dim, bias=False, dtype=dtype
        )
        self.q_b_norm = UnweightedRMSNorm(spec.rms_norm_eps)
        self.kv_proj = nn.Linear(spec.hidden_size, spec.head_dim, bias=False, dtype=dtype)
        self.kv_norm = RMSNorm(spec.head_dim, spec.rms_norm_eps, dtype)
        self.o_a_proj = GroupedLinear(
            spec.num_heads * spec.head_dim // spec.o_groups,
            spec.o_lora_rank,
            spec.o_groups,
            dtype,
        )
        self.o_b_proj = nn.Linear(
            spec.o_groups * spec.o_lora_rank, spec.hidden_size, bias=False, dtype=dtype
        )
        self.sinks = nn.Parameter(torch.zeros(spec.num_heads, dtype=dtype))
        self.scaling = spec.head_dim ** -0.5
        self.csa_compressor = CSACompressor(spec, dtype) if kind == "csa" else None
        self.hca_compressor = HCACompressor(spec, dtype) if kind == "hca" else None
        self.indexer = LightningIndexer(spec, dtype) if kind == "csa" else None

    def _project_qkv(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, theta: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        spec = self.spec
        q_lora = self.q_a_norm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(q_lora).view(batch, seq_len, spec.num_heads, spec.head_dim)
        query = self.q_b_norm(query)
        kv = self.kv_norm(self.kv_proj(hidden_states))
        cos, sin = rope_cos_sin(positions, spec.rope_head_dim, theta)
        query = apply_partial_rope(query, cos, sin)
        kv = apply_partial_rope(kv, cos, sin)
        return q_lora, query, kv

    def _grouped_out(
        self, context: torch.Tensor, positions: torch.Tensor, theta: float
    ) -> torch.Tensor:
        # Undo V-side RoPE on the trailing slice (K=V, paper §2.3.3).
        spec = self.spec
        batch, seq_len, heads, dim = context.shape
        cos, sin = rope_cos_sin(positions, spec.rope_head_dim, theta)
        context = apply_partial_rope(context, cos, -sin)
        grouped = context.reshape(batch, seq_len, spec.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped)

    def _attend(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Softmax over a per-query key set, plus a per-head sink.

        query: [B, S, H, D]
        keys:  [B, S, K, D]  (MQA: one KV, broadcast across heads)
        valid: [B, S, K] bool
        """
        scores = torch.einsum("bshd,bskd->bhsk", query.float(), keys.float())
        scores = scores * self.scaling
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        sink = self.sinks.float().view(1, -1, 1, 1).expand(
            query.shape[0], query.shape[2], query.shape[1], 1
        )
        logits = torch.cat([scores, sink], dim=-1)
        logits = logits - logits.amax(dim=-1, keepdim=True)
        probs = logits.softmax(dim=-1)[..., :-1].to(keys.dtype)
        return torch.einsum("bhsk,bskd->bshd", probs, keys)

    def _core_from_branches(
        self,
        query: torch.Tensor,
        swa_keys: torch.Tensor,
        swa_valid: torch.Tensor,
        comp_keys: Optional[torch.Tensor],
        comp_valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        if comp_keys is None:
            context = self._attend(query, swa_keys, swa_valid)
            return context, swa_keys.shape[2]
        keys = torch.cat([swa_keys, comp_keys], dim=2)
        valid = torch.cat([swa_valid, comp_valid], dim=2)
        return self._attend(query, keys, valid), keys.shape[2]

    def _gather_compressed(
        self, compressed: torch.Tensor, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather per-query compressed rows. `indices` is [B, S, k], -1 = pad."""
        valid = indices >= 0
        if compressed.shape[1] == 0:
            batch, seq_len, top_k = indices.shape
            empty = compressed.new_zeros(batch, seq_len, top_k, compressed.shape[-1])
            return empty, valid
        safe = indices.clamp_min(0)
        gathered = compressed.unsqueeze(1).expand(-1, indices.shape[1], -1, -1)
        index = safe.unsqueeze(-1).expand(*safe.shape, compressed.shape[-1])
        gathered = gathered.gather(2, index)
        return gathered, valid

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> Dsv4Output:
        """Prefill: `hidden_states` is [B, S, hidden]."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be [B, S, hidden]")
        batch, seq_len, hidden = hidden_states.shape
        if hidden != self.spec.hidden_size:
            raise ValueError(
                f"hidden_size {hidden} != spec {self.spec.hidden_size}"
            )
        device = hidden_states.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        theta = (
            self.spec.rope_theta
            if self.kind == "swa"
            else self.spec.compress_rope_theta
        )
        q_lora, query, kv = self._project_qkv(hidden_states, positions, theta)

        if self.kind == "dense":
            key_mat = kv.unsqueeze(1).expand(-1, seq_len, -1, -1)
            causal = torch.arange(seq_len, device=device).unsqueeze(0) <= torch.arange(
                seq_len, device=device
            ).unsqueeze(1)
            valid = causal.unsqueeze(0).expand(batch, -1, -1)
            context = self._attend(query, key_mat, valid)
            out = self._grouped_out(context, positions, theta)
            return Dsv4Output(out, None, 0, seq_len)

        swa_keys, swa_valid = sliding_window_keys(kv, self.spec.sliding_window)
        if self.kind == "swa":
            context, n_keys = self._core_from_branches(
                query, swa_keys, swa_valid, None, None
            )
            out = self._grouped_out(context, positions, theta)
            return Dsv4Output(out, None, 0, n_keys)

        if self.kind == "hca":
            compressed = self.hca_compressor(hidden_states)
            n_comp = compressed.shape[1]
            comp_keys = compressed.unsqueeze(1).expand(-1, seq_len, -1, -1)
            comp_valid = self.hca_compressor.causal_valid(
                seq_len, n_comp, device
            ).unsqueeze(0).expand(batch, -1, -1)
            context, n_keys = self._core_from_branches(
                query, swa_keys, swa_valid, comp_keys, comp_valid
            )
            out = self._grouped_out(context, positions, theta)
            return Dsv4Output(out, None, n_comp, n_keys)

        if self.kind != "csa":
            raise ValueError(f"unhandled kind {self.kind!r}")

        compressed = self.csa_compressor(hidden_states)
        topk, _ = self.indexer(hidden_states, q_lora, position_ids=positions)
        gathered, gather_valid = self._gather_compressed(compressed, topk)
        context, n_keys = self._core_from_branches(
            query, swa_keys, swa_valid, gathered, gather_valid
        )
        out = self._grouped_out(context, positions, theta)
        return Dsv4Output(out, topk, compressed.shape[1], n_keys)

    def build_decode_cache(
        self,
        seq_len: int,
        batch: int = 1,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ) -> DecodeCache:
        """Synthetic history at context `seq_len` (already compressed).

        Allocates the tensors a decode step would read: the last `sliding_window`
        raw KV rows plus `floor(S / m)` compressed rows. Does not run prefill.
        """
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if device is None:
            device = next(self.parameters()).device
        spec = self.spec
        generator = torch.Generator(device=device).manual_seed(seed)
        window = min(spec.sliding_window, seq_len)
        swa_kv = torch.randn(
            batch, window, spec.head_dim, generator=generator, device=device, dtype=self.dtype
        )
        if self.kind == "hca":
            n_comp = seq_len // spec.compress_rate_hca
        elif self.kind == "csa":
            n_comp = seq_len // spec.compress_rate_csa
        else:
            n_comp = 0
        compressed = torch.randn(
            batch, n_comp, spec.head_dim, generator=generator, device=device, dtype=self.dtype
        ) if n_comp else torch.zeros(
            batch, 0, spec.head_dim, device=device, dtype=self.dtype
        )
        indexer_kv = None
        if self.kind == "csa" and n_comp:
            indexer_kv = torch.randn(
                batch,
                n_comp,
                spec.index_head_dim,
                generator=generator,
                device=device,
                dtype=self.dtype,
            )
        return DecodeCache(seq_len, swa_kv, compressed, indexer_kv)

    @torch.no_grad()
    def decode(self, hidden_token: torch.Tensor, cache: DecodeCache) -> Dsv4Output:
        """One new token `[B, 1, hidden]` against `cache` (context length S)."""
        if hidden_token.ndim != 3 or hidden_token.shape[1] != 1:
            raise ValueError("hidden_token must be [B, 1, hidden]")
        batch = hidden_token.shape[0]
        device = hidden_token.device
        spec = self.spec
        position = torch.full((batch, 1), cache.seq_len, device=device, dtype=torch.long)
        theta = spec.rope_theta if self.kind == "swa" else spec.compress_rope_theta
        q_lora, query, kv = self._project_qkv(hidden_token, position, theta)
        swa_keys = torch.cat([cache.swa_kv, kv], dim=1)[:, -spec.sliding_window :, :]
        swa_keys = swa_keys.unsqueeze(1)
        swa_valid = torch.ones(
            batch, 1, swa_keys.shape[2], dtype=torch.bool, device=device
        )

        if self.kind == "dense":
            raise ValueError("dense decode needs the full KV history; use prefill")

        if self.kind == "swa":
            context, n_keys = self._core_from_branches(
                query, swa_keys, swa_valid, None, None
            )
            out = self._grouped_out(context, position, theta)
            return Dsv4Output(out, None, 0, n_keys)

        if self.kind == "hca":
            comp_keys = cache.compressed_kv.unsqueeze(1)
            n_comp = cache.compressed_kv.shape[1]
            visible = (cache.seq_len + 1) // spec.compress_rate_hca
            valid = torch.arange(n_comp, device=device) < visible
            comp_valid = valid.view(1, 1, n_comp).expand(batch, 1, -1)
            context, n_keys = self._core_from_branches(
                query, swa_keys, swa_valid, comp_keys, comp_valid
            )
            out = self._grouped_out(context, position, theta)
            return Dsv4Output(out, None, n_comp, n_keys)

        if self.kind != "csa":
            raise ValueError(f"unhandled kind {self.kind!r}")

        indexer_kv = cache.indexer_kv
        if indexer_kv is None:
            raise ValueError("CSA decode cache is missing indexer_kv")
        topk, _ = self.indexer(
            hidden_token, q_lora, compressed_kv=indexer_kv, position_ids=position
        )
        gathered, gather_valid = self._gather_compressed(cache.compressed_kv, topk)
        context, n_keys = self._core_from_branches(
            query, swa_keys, swa_valid, gathered, gather_valid
        )
        out = self._grouped_out(context, position, theta)
        return Dsv4Output(out, topk, cache.compressed_kv.shape[1], n_keys)


def build_layer(
    kind: LayerKind,
    spec: Dsv4Spec | str = "flash",
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> Dsv4AttentionLayer:
    if isinstance(spec, str):
        if spec not in SPECS:
            raise ValueError(f"unknown spec {spec!r}, expected one of {list(SPECS)}")
        spec = SPECS[spec]
    layer = Dsv4AttentionLayer(spec, kind, dtype=dtype)
    if device is not None:
        layer = layer.to(device)
    return layer.eval()
