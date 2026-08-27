"""DeepSeek-style sparse attention (DSA) implementation, isolated from the
benchmark drivers.

This file holds *everything that reproduces the attention math* — readable
paper-form and weight-absorbed MLA, the end-to-end DSA path, the
lightning-indexer hot path, the FP8 per-block quantizer, the real `nn.Module`
indexers (DSv3.2 / GLM-5 / DSv4-C4), and thin wrappers that register the FA4
dense and block-sparse Triton backends. The two
`bench_*_sparse_attention.py` files import from here and contain only the
sweep / table-printing logic, so the "what is DSA" code and the "how do we
time it" code no longer live in one confusing file.

Scope note (language-model attention first):
    The block-sparse FA kernel used as the sparse-attention step below is
    currently sourced from an external block-sparse kernel package (a
    `sparse_attn/` module) -- the same generic Triton block-sparse kernel
    WanVideo's DiT uses for *video*
    sparse attention). We reuse it here purely as the LM sparse-attention
    step. The video-specific sparse-attention story (VSA / WanVideo) is out
    of scope for now and can be investigated later.

Layout convention: q, k, v are [B, S, H, D] bf16 (BSHD). Backends that need
BHSD transpose internally so all call sites share one `run` signature.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hyperparameters (DeepSeek V3.2 / GLM-5 defaults)
# ---------------------------------------------------------------------------

# DSA indexer shape.
INDEXER_N_HEADS = 64
INDEXER_HEAD_DIM = 128
INDEXER_TOPK = 2048
INDEXER_BLOCK_SIZE = 128  # FP8 quant block size; also matches deep_gemm.

# Sparse-FA block size (matches the block-sparse kernel and the DSA paper tiling).
SPARSE_BLOCK = 64


# ---------------------------------------------------------------------------
# Environment self-check
# ---------------------------------------------------------------------------

def env_report() -> Dict[str, str]:
    """Return a dict describing which optional deps this environment has.

    Every entry is user-visible so anyone running a bench can see what is
    going to be exercised versus what will fall back to a torch reference.
    """
    report: Dict[str, str] = {}
    try:
        import torch as _t  # noqa: PLC0415
        report["torch"] = f"{_t.__version__} (cuda={_t.version.cuda})"
        report["cuda_available"] = str(_t.cuda.is_available())
        if _t.cuda.is_available():
            report["device"] = _t.cuda.get_device_name(0)
    except ImportError:
        report["torch"] = "MISSING -- see requirements.txt"
    try:
        import triton as _tt  # noqa: PLC0415
        report["triton"] = _tt.__version__
    except ImportError:
        report["triton"] = "MISSING (needed by block-sparse FA)"
    try:
        import deep_gemm as _dg  # noqa: PLC0415
        report["deep_gemm"] = "present" if hasattr(_dg, "fp8_mqa_logits") else \
            "present but no fp8_mqa_logits (indexer will use FP32 fallback)"
    except ImportError:
        report["deep_gemm"] = "MISSING (indexer will use FP32 fallback, ~50x slower)"
    try:
        from flash_attn.cute import flash_attn_func  # noqa: F401, PLC0415
        report["flash_attn.cute"] = "present"
    except ImportError:
        report["flash_attn.cute"] = "MISSING (dense baseline will fall back to sgl FA4)"
    return report


def try_import_deep_gemm():
    """Return the deep_gemm module iff it exposes fp8_mqa_logits, else None."""
    try:
        import deep_gemm  # noqa: PLC0415
        if hasattr(deep_gemm, "fp8_mqa_logits"):
            return deep_gemm
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# FP8 per-block quantize (shared by the indexer sim and the nn.Module indexers)
# ---------------------------------------------------------------------------

def fp8_quant_per_block(
    x_bf16: torch.Tensor, block_size: int = INDEXER_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize bf16 -> FP8 e4m3 with one FP32 scale per block.

    Mirrors sglang's `act_quant` for the DSA indexer. Returns (q_fp8, scale).

    Shape convention: x is [T, H, D] or [T, D]. We quantize along the last dim
    in chunks of `block_size`. The returned scale is **squeezed**: with the
    DSA default head_dim = block_size = 128 there is one block per row, so the
    scale collapses to [T, H] or [T], matching what `deep_gemm.fp8_mqa_logits`
    expects for its kv-scale argument.
    """
    orig_shape = x_bf16.shape
    last_dim = orig_shape[-1]
    assert last_dim % block_size == 0, f"last dim {last_dim} not multiple of block {block_size}"
    num_blocks = last_dim // block_size
    x = x_bf16.reshape(-1, last_dim).to(torch.float32)
    blocks = x.view(-1, num_blocks, block_size)
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = absmax / fp8_max
    q = (blocks / scale).to(torch.float8_e4m3fn)
    q = q.view(*orig_shape)
    scale = scale.squeeze(-1).to(torch.float32)  # last dim collapses
    if num_blocks == 1:
        # No per-token chunking -- one scale per row. Reshape to orig leading dims.
        scale = scale.view(*orig_shape[:-1])
    else:
        scale = scale.view(*orig_shape[:-1], num_blocks)
    return q, scale


# ---------------------------------------------------------------------------
# Dense FA4 backends (dense baseline for the speedup columns)
# ---------------------------------------------------------------------------

def register_fa4() -> Optional[Callable]:
    """flash_attn FA4 (CuTe). Returns None if unavailable."""
    try:
        from flash_attn.cute import flash_attn_func
    except ImportError:
        return None

    def run(q, k, v, seq_len, causal, softmax_scale):
        return flash_attn_func(
            q=q, k=k, v=v, softmax_scale=softmax_scale, causal=causal,
        )[0]
    return run


def register_sgl_fa4() -> Optional[Callable]:
    """sglang FA4 path (flash_attention_v4). Same kernel family as FA4 cute --
    kept only as a fallback for environments without `flash_attn.cute`."""
    try:
        from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func
    except ImportError:
        return None

    def run(q, k, v, seq_len, causal, softmax_scale):
        return flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=None, cu_seqlens_k=None,
            max_seqlen_q=seq_len, max_seqlen_k=seq_len,
            softmax_scale=softmax_scale,
            causal=causal,
            return_softmax_lse=False,
        )
    return run


def register_dense_fn() -> Optional[Callable]:
    """The dense baseline used for every speedup column: FA4 cute, falling
    back to sglang's FA4. cuDNN SDPA is intentionally *not* used here -- its
    BSHD->BHSD transpose would be timed inside the attention call, which is an
    unfair baseline."""
    return register_fa4() or register_sgl_fa4()


# ---------------------------------------------------------------------------
# Block-sparse FA backend (the sparse-attention step, external kernel package)
# ---------------------------------------------------------------------------

def resolve_sparse_kernel_path() -> Optional[str]:
    """Locate the directory holding the block-sparse FA kernel package.

    The block-sparse Triton kernel used as the sparse-attention step (PART
    B/D/E) is not part of upstream sglang; it lives in an external package that
    exposes a `sparse_attn/` module. Point `SPARSE_ATTN_KERNELS_DIR` at the
    directory that contains that `sparse_attn/` package. Returns None when the
    env var is unset or does not point at such a directory -- the block-sparse
    parts then skip cleanly.
    """
    env = os.environ.get("SPARSE_ATTN_KERNELS_DIR")
    if env and os.path.isdir(os.path.join(env, "sparse_attn")):
        return env
    return None


def register_sparse_fa() -> Optional[Tuple[Callable, Callable]]:
    """OpenAI-style block-sparse FA with prefetched LUT.

    Returns a (build_lut, run) pair. build_lut is called per (S, B, H,
    sparsity) once, then `run(q,k,v,lut_idx,lut_num,sm_scale)` is the hot path.

    NOTE: this kernel comes from an external block-sparse kernel package (the
    generic block-sparse Triton kernel WanVideo also uses for video sparse
    attention). We reuse it as the LM sparse-attention step; the video-specific
    usage is out of scope here and can be investigated later.
    """
    kernel_path = resolve_sparse_kernel_path()
    if kernel_path is None:
        return None
    if kernel_path not in sys.path:
        sys.path.insert(0, kernel_path)
    try:
        from sparse_attn.video_sparse_kernel import (
            triton_block_sparse_attn_fwd_openai_prefetch,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  sparse FA import failed: {e}")
        return None

    def build_lut(q: torch.Tensor, S: int, sparsity: float) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B, _, H, _ = q.shape
        Nq = Nkv = S // SPARSE_BLOCK
        topk = max(1, int(round(sparsity * Nkv)))
        # Random per-query top-k LUT. Equivalent to "indexer already ran";
        # the *cost* of the indexer is measured separately in the indexer part.
        scores = torch.rand(B, H, Nq, Nkv, device=q.device)
        idx = scores.topk(topk, dim=-1).indices.to(torch.int32).contiguous()
        q2k_num = torch.full((B, H, Nq), topk, dtype=torch.int32, device=q.device)
        return idx, q2k_num, topk

    def run(q, k, v, idx, num, sm_scale):
        out, _ = triton_block_sparse_attn_fwd_openai_prefetch(
            q, k, v, idx, num, sm_scale=sm_scale,
        )
        return out

    return build_lut, run


# ===========================================================================
# Lightning indexer -- kernel-level simulator (used by the kernel bench)
# ===========================================================================

class IndexerSimulator:
    """Reproduce the lightning indexer's *hot path* on synthetic tensors.

    All shapes match `sglang/.../attention/dsa/dsa_indexer.py`'s `Indexer`:

        Q index : [T, n_heads=64, head_dim=128] FP8 (per-block UE8M0)
        K index : [T, head_dim=128]             FP8  (MQA -- 1 K head)
        weights : [T, n_heads=64]               FP32 per-head gate
        score   : [T, T]                        FP32 logits, causal-masked
        out     : [T, topk=2048]                int32 top-k indices

    We do NOT include the original projections (wq_b / wk / weights_proj /
    rotary / layernorm) in the "logits" hot path since those scale linearly
    with token count and are not the bottleneck. We *do* time them separately
    so we can report "indexer proj cost" vs "indexer logits + topk cost".
    """

    def __init__(
        self,
        S: int,
        n_heads: int = INDEXER_N_HEADS,
        head_dim: int = INDEXER_HEAD_DIM,
        topk: int = INDEXER_TOPK,
        device: torch.device = torch.device("cuda"),
        hidden_size: int = 7168,  # DSv3.2 / GLM-5
        q_lora_rank: int = 1536,
    ):
        self.S = S
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.topk = topk
        self.device = device
        self.dg = try_import_deep_gemm()

        # Persistent buffers (same shapes the real Indexer would produce).
        self.q_lora_a = torch.randn(S, q_lora_rank, dtype=torch.bfloat16, device=device)
        self.x = torch.randn(S, hidden_size, dtype=torch.bfloat16, device=device)
        self.positions = torch.arange(S, device=device, dtype=torch.int32)

        # Projections (initialized once; the real model loads weights here).
        self.wq_b = torch.nn.Linear(q_lora_rank, n_heads * head_dim, bias=False, device=device, dtype=torch.bfloat16)
        self.wk = torch.nn.Linear(hidden_size, head_dim, bias=False, device=device, dtype=torch.bfloat16)
        self.weights_proj = torch.nn.Linear(hidden_size, n_heads, bias=False, device=device, dtype=torch.bfloat16)

        # ks/ke for causal masking: each query attends to all earlier positions.
        self.ks = torch.zeros(S, dtype=torch.int32, device=device)
        self.ke = (self.positions + 1).to(torch.int32).clone()

        # Cached FP8 tensors for the "logits-only" hot path (we precompute them).
        self._fp8_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def projections_step(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run all linear / RoPE / layernorm work that the indexer needs."""
        q = self.wq_b(self.q_lora_a).view(self.S, self.n_heads, self.head_dim)
        k = self.wk(self.x)  # [S, head_dim]
        w = self.weights_proj(self.x).float() * (self.n_heads ** -0.5)
        return q, k, w

    def quantize_step(
        self, q_bf16: torch.Tensor, k_bf16: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_fp8, q_scale = fp8_quant_per_block(q_bf16)
        k_fp8, k_scale = fp8_quant_per_block(k_bf16)
        return q_fp8, q_scale, k_fp8, k_scale

    def precompute_fp8(self) -> None:
        q_bf16, k_bf16, w = self.projections_step()
        q_fp8, q_scale, k_fp8, k_scale = self.quantize_step(q_bf16, k_bf16)
        # In the real Indexer, q_scale is folded into the head-gate weights
        # (`weights = weights * q_scale * softmax_scale`). deep_gemm.fp8_mqa_logits
        # does NOT take a q-scale, so we absorb it here.
        sm = self.head_dim ** -0.5
        if q_scale.dim() == 2:  # [T, H] -- one block covers head_dim
            w_eff = (w.float() * q_scale.float() * sm).contiguous()
        else:                    # [T, H, n_blocks] -- average over blocks
            w_eff = (w.float() * q_scale.float().mean(dim=-1) * sm).contiguous()
        self._fp8_cache = (
            q_fp8.contiguous(), k_fp8.contiguous(), k_scale.contiguous(), w_eff,
        )

    def logits_step(self) -> torch.Tensor:
        """Score-and-topk hot path. Returns [S, topk] int indices."""
        assert self._fp8_cache is not None
        q_fp8, k_fp8, k_scale, w = self._fp8_cache
        if self.dg is not None:
            # deep_gemm expects k_scale of shape [S] (one per token).
            ks_arg = k_scale if k_scale.dim() == 1 else k_scale.squeeze(-1)
            logits = self.dg.fp8_mqa_logits(
                q_fp8, (k_fp8, ks_arg.contiguous()), w, self.ks, self.ke,
            )
        else:
            # Reference fallback (FP32) -- slow but works without deep_gemm.
            q = q_fp8.to(torch.float32)
            k_dq = k_fp8.to(torch.float32)
            if k_scale.dim() == 1:
                k_dq = k_dq * k_scale.unsqueeze(-1)
            else:
                k_dq = k_dq * k_scale.unsqueeze(-1).repeat_interleave(
                    self.head_dim // k_scale.shape[-1], dim=-1,
                )
            scores = torch.einsum("thd,sd->ths", q, k_dq)  # [T, H, S]
            scores = (scores * w.unsqueeze(-1)).sum(dim=1)  # [T, S]
            tril = torch.tril(torch.ones(self.S, self.S, dtype=torch.bool, device=self.device))
            scores = scores.masked_fill(~tril, float("-inf"))
            logits = scores
        topk_idx = logits.topk(min(self.topk, self.S), dim=-1).indices
        return topk_idx


# ===========================================================================
# Lightning indexer -- real nn.Module indexers (used by the module bench)
# ===========================================================================

@dataclass
class IndexerSpec:
    """Hyperparameters for a specific model's indexer."""
    name: str
    hidden_size: int
    q_lora_rank: int
    index_n_heads: int
    index_head_dim: int
    rope_head_dim: int
    index_topk: int
    compress_ratio: int = 1  # DSv4 sets this to 4
    rope_interleave: bool = False  # GLM-5 sets True


SPECS: Dict[str, IndexerSpec] = {
    # DSv3.2: 7168 hidden, 1536 q_lora_rank, 64 idx heads, 128 idx dim,
    #         RoPE dim 64 (NeoX), top-k 2048.
    "dsv32": IndexerSpec(
        name="DSv3.2",
        hidden_size=7168,
        q_lora_rank=1536,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        index_topk=2048,
        compress_ratio=1,
        rope_interleave=False,
    ),
    # GLM-5 / 5.1 / 5.2 (zai-org/GLM-5, arXiv 2602.15763; 744B total / 40B active
    # MoE, 78 layers, 256 experts). Verified against the released config.json:
    #   hidden_size=6144, q_lora_rank=2048, index_n_heads=32, index_head_dim=128,
    #   index_topk=2048, indexer_rope_interleave=true, qk_rope_head_dim=64,
    #   qk_nope_head_dim=192, v_head_dim=256.
    # NOTE index_n_heads is 32, NOT DSv3.2's 64: GLM halves the indexer head
    # count just as it halves the main-attention head count (128 -> 64). RoPE on
    # the indexer is GPT-J interleaved, not NeoX. GLM-5.2 layers IndexShare on
    # top of the *same* indexer -- that is a runtime index_topk_freq=4 knob, so
    # it needs no spec field here.
    "glm": IndexerSpec(
        name="GLM-5 / 5.1 / 5.2",
        hidden_size=6144,
        q_lora_rank=2048,
        index_n_heads=32,
        index_head_dim=128,
        rope_head_dim=64,
        index_topk=2048,
        compress_ratio=1,
        rope_interleave=True,
    ),
    # DSv4: same as DSv3.2 but compressor compresses K stream 4x before scoring.
    "dsv4": IndexerSpec(
        name="DSv4",
        hidden_size=7168,
        q_lora_rank=1536,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        index_topk=2048,
        compress_ratio=4,
        rope_interleave=False,
    ),
}


# Cache for RoPE cos/sin to avoid rebuild on every call.
_ROPE_CACHE: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def _rope_cossin(S: int, dim: int, device: torch.device, base: float = 10000.0):
    key = (S, dim, device)
    if key in _ROPE_CACHE:
        return _ROPE_CACHE[key]
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(S, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    _ROPE_CACHE[key] = (cos, sin)
    return cos, sin


def _apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX-style: split last dim in halves, rotate."""
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    # cos/sin are [S, half]; need broadcast to x's leading dims.
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _apply_rope_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved (GPT-J style) rotation. cos / sin are [S, half]."""
    d = x.shape[-1]
    half = d // 2
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    while cos.dim() < x_even.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


@dataclass(frozen=True)
class MLASpec:
    """Shape-only configuration for the readable MLA reference."""

    hidden_size: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_interleave: bool = False


@dataclass
class MLAState:
    """MLA activations; ``kv_latent`` and ``k_rope`` are the persistent cache."""

    q_lora: torch.Tensor
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    kv_latent: torch.Tensor
    k_rope: torch.Tensor


class MLAReference(nn.Module):
    """Pure-PyTorch Multi-head Latent Attention.

    ``forward_unabsorbed`` implements the straightforward paper equation by
    reconstructing per-head K/V. ``forward`` implements mathematically
    equivalent weight-absorbed MLA, which caches only the latent KV and shared
    RoPE key. Production SGLang replaces these tensor operations with paged,
    fused kernels; the attention logic remains visible here.
    """

    def __init__(self, spec: MLASpec, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if spec.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even for RoPE")
        self.spec = spec
        self.q_a_proj = nn.Linear(
            spec.hidden_size, spec.q_lora_rank, bias=False, dtype=dtype
        )
        self.q_a_norm = nn.LayerNorm(spec.q_lora_rank, dtype=dtype)
        self.q_b_proj = nn.Linear(
            spec.q_lora_rank,
            spec.num_heads * (spec.qk_nope_head_dim + spec.qk_rope_head_dim),
            bias=False,
            dtype=dtype,
        )
        self.kv_a_proj = nn.Linear(
            spec.hidden_size,
            spec.kv_lora_rank + spec.qk_rope_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.kv_a_norm = nn.LayerNorm(spec.kv_lora_rank, dtype=dtype)
        self.kv_b_proj = nn.Linear(
            spec.kv_lora_rank,
            spec.num_heads * (spec.qk_nope_head_dim + spec.v_head_dim),
            bias=False,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            spec.num_heads * spec.v_head_dim,
            spec.hidden_size,
            bias=False,
            dtype=dtype,
        )
        self.softmax_scale = (
            spec.qk_nope_head_dim + spec.qk_rope_head_dim
        ) ** -0.5

    def project(self, hidden_states: torch.Tensor) -> MLAState:
        """Run Q/KV compression, per-head Q expansion, and decoupled RoPE."""
        T = hidden_states.shape[0]
        s = self.spec

        q_lora = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_lora).view(
            T, s.num_heads, s.qk_nope_head_dim + s.qk_rope_head_dim
        )
        q_nope, q_rope = q.split(
            [s.qk_nope_head_dim, s.qk_rope_head_dim], dim=-1
        )

        kv = self.kv_a_proj(hidden_states)
        kv_latent, k_rope = kv.split(
            [s.kv_lora_rank, s.qk_rope_head_dim], dim=-1
        )
        kv_latent = self.kv_a_norm(kv_latent)

        cos, sin = _rope_cossin(T, s.qk_rope_head_dim, hidden_states.device)
        rope_fn = (
            _apply_rope_interleave if s.rope_interleave else _apply_rope_neox
        )
        return MLAState(
            q_lora=q_lora,
            q_nope=q_nope,
            q_rope=rope_fn(q_rope, cos, sin),
            kv_latent=kv_latent,
            k_rope=rope_fn(k_rope, cos, sin),
        )

    def _reconstruction_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the per-head K/V up-projection weights."""
        s = self.spec
        weight = self.kv_b_proj.weight.view(
            s.num_heads,
            s.qk_nope_head_dim + s.v_head_dim,
            s.kv_lora_rank,
        )
        return weight.split([s.qk_nope_head_dim, s.v_head_dim], dim=1)

    @staticmethod
    def _causal_mask(scores: torch.Tensor) -> torch.Tensor:
        T_q, T_k = scores.shape[-2:]
        mask = torch.ones(
            T_q, T_k, dtype=torch.bool, device=scores.device
        ).tril(diagonal=T_k - T_q)
        return scores.masked_fill(~mask, float("-inf"))

    def forward_unabsorbed(
        self, hidden_states: torch.Tensor, causal: bool = True
    ) -> torch.Tensor:
        """Paper form: explicitly reconstruct full per-head K and V."""
        state = self.project(hidden_states)
        s = self.spec
        T = hidden_states.shape[0]
        kv = self.kv_b_proj(state.kv_latent).view(
            T, s.num_heads, s.qk_nope_head_dim + s.v_head_dim
        )
        k_nope, value = kv.split(
            [s.qk_nope_head_dim, s.v_head_dim], dim=-1
        )
        query = torch.cat([state.q_nope, state.q_rope], dim=-1)
        key = torch.cat(
            [
                k_nope,
                state.k_rope.unsqueeze(1).expand(-1, s.num_heads, -1),
            ],
            dim=-1,
        )
        scores = torch.einsum("thd,shd->ths", query, key).float()
        scores *= self.softmax_scale
        if causal:
            scores = self._causal_mask(scores)
        probs = scores.softmax(dim=-1).to(value.dtype)
        context = torch.einsum("ths,shd->thd", probs, value)
        return self.o_proj(context.reshape(T, s.num_heads * s.v_head_dim))

    def attend_absorbed(
        self,
        state: MLAState,
        topk_indices: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """Attend to compressed MLA state, optionally at DSA-selected indices."""
        s = self.spec
        T = state.q_nope.shape[0]
        k_up, v_up = self._reconstruction_weights()

        # K absorption: move W_K from every cached key into the current query.
        q_absorbed = torch.einsum("thn,hnc->thc", state.q_nope, k_up)

        if topk_indices is None:
            scores = torch.einsum(
                "thc,sc->ths", q_absorbed, state.kv_latent
            )
            scores += torch.einsum(
                "thr,sr->ths", state.q_rope, state.k_rope
            )
            scores = scores.float() * self.softmax_scale
            if causal:
                scores = self._causal_mask(scores)
            probs = scores.softmax(dim=-1).to(state.kv_latent.dtype)
            latent_context = torch.einsum(
                "ths,sc->thc", probs, state.kv_latent
            )
        else:
            if topk_indices.ndim != 2 or topk_indices.shape[0] != T:
                raise ValueError("topk_indices must have shape [T, K]")
            if topk_indices.numel() and (
                topk_indices.min() < -1
                or topk_indices.max() >= state.kv_latent.shape[0]
            ):
                raise ValueError("topk_indices contains an out-of-range token")
            if causal and topk_indices.numel():
                query_pos = torch.arange(T, device=topk_indices.device)[:, None]
                if torch.any(topk_indices > query_pos):
                    raise ValueError("causal topk_indices contains a future token")

            # A fixed-width [T, K] tensor needs padding for early causal
            # queries that have fewer than K predecessors. -1 is the sentinel.
            valid = topk_indices >= 0
            safe_indices = topk_indices.clamp_min(0)
            selected_latent = state.kv_latent[safe_indices]
            selected_rope = state.k_rope[safe_indices]
            scores = torch.einsum(
                "thc,tkc->thk", q_absorbed, selected_latent
            )
            scores += torch.einsum(
                "thr,tkr->thk", state.q_rope, selected_rope
            )
            scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
            probs = (scores.float() * self.softmax_scale).softmax(dim=-1)
            probs = probs.to(selected_latent.dtype)
            latent_context = torch.einsum(
                "thk,tkc->thc", probs, selected_latent
            )

        # V absorption: aggregate latent values first, then reconstruct V.
        context = torch.einsum("thc,hvc->thv", latent_context, v_up)
        return self.o_proj(context.reshape(T, s.num_heads * s.v_head_dim))

    def forward(
        self, hidden_states: torch.Tensor, causal: bool = True
    ) -> torch.Tensor:
        """Dense MLA in its cache-efficient, weight-absorbed form."""
        return self.attend_absorbed(self.project(hidden_states), causal=causal)


class DsaIndexer(nn.Module):
    """nn.Module mirror of `sglang.srt.layers.attention.dsa.dsa_indexer.Indexer`.

    Reproduces the production module's *layer shapes* and *forward dataflow*,
    but with no KV cache, paged layout, dual-stream, or piecewise CUDA-graph
    plumbing. Used as a portable benchmark target.

    Forward (causal mask, single-sample T = S):
        x       : [T, hidden_size]   bf16
        q_lora_a: [T, q_lora_rank]   bf16   (mimics the post-LoRA Q activation)
        -> wq_b  -> Q index    [T, n_heads, head_dim]
        -> wk    -> K index    [T, head_dim]
        -> weights_proj -> W   [T, n_heads]
        -> k_norm + RoPE on the rope_head_dim slice
        -> per-block FP8 quant of Q and K
        -> deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8, k_scale), w, ks, ke)
        -> topk -> [T, index_topk] int32
    """

    def __init__(self, spec: IndexerSpec, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.spec = spec
        self.dtype = dtype
        self.wq_b = nn.Linear(spec.q_lora_rank, spec.index_n_heads * spec.index_head_dim, bias=False, dtype=dtype)
        self.wk = nn.Linear(spec.hidden_size, spec.index_head_dim, bias=False, dtype=dtype)
        self.weights_proj = nn.Linear(spec.hidden_size, spec.index_n_heads, bias=False, dtype=dtype)
        self.k_norm = nn.LayerNorm(spec.index_head_dim, dtype=dtype)
        self.softmax_scale = spec.index_head_dim ** -0.5
        self._dg = try_import_deep_gemm()

    @torch.no_grad()
    def projections(
        self, x: torch.Tensor, q_lora_a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = x.shape[0]
        q = self.wq_b(q_lora_a).view(T, self.spec.index_n_heads, self.spec.index_head_dim)
        k = self.k_norm(self.wk(x))
        w = self.weights_proj(x).float() * (self.spec.index_n_heads ** -0.5)
        return q, k, w

    @torch.no_grad()
    def rope(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rd = self.spec.rope_head_dim
        T = q.shape[0]
        cos, sin = _rope_cossin(T, rd, q.device)
        rope_fn = _apply_rope_interleave if self.spec.rope_interleave else _apply_rope_neox
        q_rope_slice = q[..., :rd]
        k_rope_slice = k[..., :rd]
        q_rotated = rope_fn(q_rope_slice, cos, sin)
        k_rotated = rope_fn(k_rope_slice, cos, sin)
        q = torch.cat([q_rotated, q[..., rd:]], dim=-1)
        k = torch.cat([k_rotated, k[..., rd:]], dim=-1)
        return q, k

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, q_lora_a: torch.Tensor,
    ) -> torch.Tensor:
        q, k, w = self.projections(x, q_lora_a)
        q, k = self.rope(q, k)
        q_fp8, q_scale = fp8_quant_per_block(q)
        k_fp8, k_scale = fp8_quant_per_block(k)
        T = x.shape[0]
        ks = torch.zeros(T, dtype=torch.int32, device=x.device)
        ke = torch.arange(1, T + 1, dtype=torch.int32, device=x.device)

        # Absorb q_scale into the head gates (production Indexer does the same:
        # `weights = weights * q_scale * softmax_scale`).
        if q_scale.dim() == 2:
            w_eff = (w * q_scale * self.softmax_scale).contiguous()
        else:
            w_eff = (w * q_scale.mean(dim=-1) * self.softmax_scale).contiguous()

        if self._dg is not None:
            ks_arg = k_scale if k_scale.dim() == 1 else k_scale.squeeze(-1)
            logits = self._dg.fp8_mqa_logits(
                q_fp8.contiguous(),
                (k_fp8.contiguous(), ks_arg.contiguous()),
                w_eff,
                ks, ke,
            )
        else:
            qf = q_fp8.to(torch.float32)
            kf = k_fp8.to(torch.float32)
            if k_scale.dim() == 1:
                kf = kf * k_scale.unsqueeze(-1)
            else:
                kf = kf * k_scale.unsqueeze(-1).repeat_interleave(
                    self.spec.index_head_dim // k_scale.shape[-1], dim=-1,
                )
            scores = torch.einsum("thd,sd->ths", qf, kf)
            scores = (scores * w_eff.unsqueeze(-1)).sum(dim=1)
            tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~tril, float("-inf"))
            logits = scores
        topk_idx = logits.topk(min(self.spec.index_topk, T), dim=-1).indices
        return topk_idx


class C4Compressor(nn.Module):
    """Mirror of `sglang.srt.layers.attention.dsv4.compressor.Compressor`.

    Pools every `compress_ratio` consecutive tokens into one "compressor token"
    via a learned linear + RoPE. Output length is `T // compress_ratio`.
    """

    def __init__(self, hidden_size: int, head_dim: int, compress_ratio: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        # `pool_proj` plays the role of the learned compressor projection;
        # the real Compressor uses Hadamard + linear, but the dataflow is the
        # same: hidden -> head_dim per compressor token.
        self.pool_proj = nn.Linear(hidden_size * compress_ratio, head_dim, bias=False, dtype=dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        assert T % self.compress_ratio == 0
        n_comp = T // self.compress_ratio
        # Stack `compress_ratio` consecutive tokens into one row, then project.
        x_grouped = x.view(n_comp, self.compress_ratio * H)
        c = self.pool_proj(x_grouped)  # [n_comp, head_dim]
        return c


class Dsv4Indexer(DsaIndexer):
    """DSA-C4: same as DsaIndexer, but K stream is compressed `compress_ratio`x.

    The Q stream is **not** compressed. Each Q token still scores against all
    `S / compress_ratio` compressor K tokens (4x fewer than full S). After
    top-k, the resulting indices live in *compressor token space* and must be
    expanded back to raw token positions; we leave that to the sparse-FA
    step downstream.

    The full DeepSeek-V4 CSA / HCA *layer* (overlapping compressor, SWA
    branch, grouped o_proj, HCA with no indexer) is `dsv4_layer.py`.
    """

    def __init__(self, spec: IndexerSpec, dtype: torch.dtype = torch.bfloat16):
        super().__init__(spec, dtype=dtype)
        assert spec.compress_ratio > 1, "Dsv4Indexer requires compress_ratio > 1"
        self.compressor = C4Compressor(
            hidden_size=spec.hidden_size,
            head_dim=spec.index_head_dim,
            compress_ratio=spec.compress_ratio,
            dtype=dtype,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, q_lora_a: torch.Tensor) -> torch.Tensor:
        # Q path: unchanged.
        T = x.shape[0]
        q = self.wq_b(q_lora_a).view(T, self.spec.index_n_heads, self.spec.index_head_dim)
        # K path: compress THEN normalize. The real DSv4 compresses inside the
        # compressor and applies RoPE on the compressed sequence.
        k = self.k_norm(self.compressor(x))  # [T/cr, head_dim]
        w = self.weights_proj(x).float() * (self.spec.index_n_heads ** -0.5)

        # RoPE: Q on raw positions, K on compressed positions (stride-cr).
        rd = self.spec.rope_head_dim
        cos_q, sin_q = _rope_cossin(T, rd, q.device)
        cos_k, sin_k = _rope_cossin(T // self.spec.compress_ratio, rd, q.device)
        rope_fn = _apply_rope_interleave if self.spec.rope_interleave else _apply_rope_neox
        q = torch.cat([rope_fn(q[..., :rd], cos_q, sin_q), q[..., rd:]], dim=-1)
        k = torch.cat([rope_fn(k[..., :rd], cos_k, sin_k), k[..., rd:]], dim=-1)

        # FP8 quant on the (now shorter) K stream and full Q.
        q_fp8, q_scale = fp8_quant_per_block(q)
        k_fp8, k_scale = fp8_quant_per_block(k)

        T_kv = T // self.spec.compress_ratio
        ks = torch.zeros(T, dtype=torch.int32, device=x.device)
        # Causal mask in compressor-token space: query t attends compressor
        # tokens [0, ceil((t+1)/cr)).
        ke = ((torch.arange(1, T + 1, device=x.device) + self.spec.compress_ratio - 1)
              // self.spec.compress_ratio).to(torch.int32)

        if q_scale.dim() == 2:
            w_eff = (w * q_scale * self.softmax_scale).contiguous()
        else:
            w_eff = (w * q_scale.mean(dim=-1) * self.softmax_scale).contiguous()

        if self._dg is not None:
            ks_arg = k_scale if k_scale.dim() == 1 else k_scale.squeeze(-1)
            logits = self._dg.fp8_mqa_logits(
                q_fp8.contiguous(),
                (k_fp8.contiguous(), ks_arg.contiguous()),
                w_eff,
                ks, ke,
            )
        else:
            qf = q_fp8.to(torch.float32)
            kf = k_fp8.to(torch.float32)
            if k_scale.dim() == 1:
                kf = kf * k_scale.unsqueeze(-1)
            else:
                kf = kf * k_scale.unsqueeze(-1).repeat_interleave(
                    self.spec.index_head_dim // k_scale.shape[-1], dim=-1,
                )
            scores = torch.einsum("thd,sd->ths", qf, kf)
            scores = (scores * w_eff.unsqueeze(-1)).sum(dim=1)
            mask = torch.arange(T_kv, device=x.device).unsqueeze(0) < ke.unsqueeze(1)
            scores = scores.masked_fill(~mask, float("-inf"))
            logits = scores
        # Top-k over (smaller) compressor-token space.
        K = min(self.spec.index_topk // self.spec.compress_ratio, T_kv)
        return logits.topk(K, dim=-1).indices


@dataclass
class DSAOutput:
    """End-to-end DSA result and the token indices chosen by the indexer."""

    hidden_states: torch.Tensor
    topk_indices: torch.Tensor


class DeepSeekSparseAttentionReference(nn.Module):
    """Readable end-to-end DSA: MLA → lightning indexer → sparse MLA.

    The Python code owns all model logic and tensor transformations. The
    ``DsaIndexer`` may call ``deep_gemm.fp8_mqa_logits`` when available, but
    that kernel is only an accelerator for the explicit score-and-top-k math
    in ``DsaIndexer.forward``.
    """

    def __init__(
        self,
        mla_spec: MLASpec,
        indexer_spec: IndexerSpec,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if mla_spec.hidden_size != indexer_spec.hidden_size:
            raise ValueError("MLA and indexer hidden sizes must match")
        if mla_spec.q_lora_rank != indexer_spec.q_lora_rank:
            raise ValueError("MLA and indexer Q-LoRA ranks must match")
        self.mla = MLAReference(mla_spec, dtype=dtype)
        self.indexer = DsaIndexer(indexer_spec, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> DSAOutput:
        # 1. MLA creates Q and the compressed [latent KV, shared RoPE K] cache.
        mla_state = self.mla.project(hidden_states)

        # 2. The independent lightning indexer scores all causal K positions
        #    cheaply and returns one token list per query, shared by MLA heads.
        topk_indices = self.indexer(hidden_states, mla_state.q_lora)
        query_pos = torch.arange(
            hidden_states.shape[0], device=topk_indices.device
        )[:, None]
        topk_indices = topk_indices.masked_fill(topk_indices > query_pos, -1)

        # 3. MLA computes its real softmax attention only on selected tokens.
        output = self.mla.attend_absorbed(
            mla_state, topk_indices=topk_indices, causal=True
        )
        return DSAOutput(hidden_states=output, topk_indices=topk_indices)
