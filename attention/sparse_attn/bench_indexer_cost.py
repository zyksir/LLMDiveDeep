"""bench_indexer_cost.py — DSA cost breakdown at long context.

Purpose
-------
Answer two headline questions on a single B200, with synthetic tensors that
mimic the shapes SGLang / DeepSeek run at production:

  Q1.  How does the *indexer* cost (`deep_gemm.fp8_mqa_logits` + `topk`)
       compare with the *sparse-attention* cost (FA over top-K = 2048
       tokens)?  I.e. inside DSA, where does time go?

  Q2.  How does *sparse-attention* cost compare with *dense-attention* cost
       (FA over all S tokens)?  I.e. what does DSA save vs. plain MLA?

Shapes (decode-like, MLA-absorbed MQA form; no model weights loaded)
--------------------------------------------------------------------
  Indexer   : Q [T, H_I=64, D_I=128] fp8,  K [S, D_I=128] fp8 + scale [S] f32
              (T = B queries; deep_gemm.fp8_mqa_logits)
  Main attn : Q [B, 1, H_A=32, D=128]      K,V [B, S, 1, D=128] bf16 (MQA)
              (Q_len=1 per request, decode; enable_gqa=True with H_kv=1)
              Torch SDPA with FLASH_ATTENTION backend.

Sweep is over S ∈ {32_768, 131_072, 1_048_576} and B ∈ {1, 16}.  Small S like
2 K is intentionally omitted: below ~8 K the indexer isn't amortized and the
story is misleading.

Two indexer breakdowns
----------------------
* `mqa_logits_ms` : the fp8 WGMMA + ReLU + gate + head-sum epilogue alone
                    (deep_gemm.fp8_mqa_logits call only).
* `full_indexer_ms` : one full decode-step indexer as run in production —
                      wq_b + wk + k_norm + RoPE + Hadamard + FP8 quant +
                      weights_proj + mqa_logits + topk. Steps 1–8 of
                      `SURVEY.md §2.3`.

Notes
-----
* We use PyTorch SDPA (flash-attention backend) instead of FA4-cute because
  outside the sglang dev container the cutlass-DSL JIT fails
  (`GPUModuleOp.__init__(): incompatible function arguments`, README §2).
  SDPA hits the same FA family and gives correct scaling.
* Head dim D=128 and H_A=32 are simplifications: real DSv3.2 MLA-absorbed
  has D_qk=576, H_A=128. The *ratios* here (indexer vs sparse, sparse vs
  dense) generalise; the absolute times do not.
* The Hadamard step uses sglang's `rotate_activation` (JIT-compiled 128-pt
  Hadamard). If that import fails the pipeline falls back to a manual matmul.

Outputs
-------
  * Readable table on stdout.
  * `bench_indexer_cost.csv` — all raw timings.
  * `bench_indexer_cost.png` — 2 log-log panels (indexer vs sparse; sparse vs dense).

Runs in ~1 min at default sweep on B200.  Only deps: torch, deep_gemm,
matplotlib, sglang (for Hadamard + act_quant kernels).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import gc
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import deep_gemm

try:
    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation as _sgl_hadamard
    _HAS_SGL_HADAMARD = True
except Exception:  # noqa: BLE001
    _HAS_SGL_HADAMARD = False
    _sgl_hadamard = None


# ----------------------------------------------------------------------------
# Shapes  (tweak here for a different model configuration)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    # Indexer (DSv3.2 defaults)
    H_I: int = 64          # indexer query heads
    D_I: int = 128         # indexer head_dim (== fp8 block size)
    TOPK: int = 2048       # top-K KV tokens returned by the indexer
    ROPE_DIM: int = 64     # rope_head_dim (first slice of D_I is rotated)
    HIDDEN: int = 7168     # model hidden_size (for wk, weights_proj)
    Q_LORA_RANK: int = 1536  # MLA Q-LoRA rank (input to wq_b)

    # Main attention (MQA, matches MLA absorbed form)
    H_A: int = 32          # main-attention query heads (Q-side)
    D_A: int = 128         # main-attention head_dim (Q/K/V); real MLA=576, simplified here
    H_KV: int = 1          # KV heads (MQA=1; matches MLA absorbed)

    BLOCK_SIZE: int = 128  # fp8 quant block size (matches deep_gemm alignment)


# ----------------------------------------------------------------------------
# fp8 per-block quantiser (mirrors sglang's act_quant for the indexer)
# ----------------------------------------------------------------------------

def fp8_quant(x_bf16: torch.Tensor, block: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16 -> fp8_e4m3 with one fp32 scale per `block` elements along last dim."""
    orig = x_bf16.shape
    last = orig[-1]
    assert last % block == 0
    n_blocks = last // block
    x = x_bf16.reshape(-1, last).to(torch.float32)
    b = x.view(-1, n_blocks, block)
    absmax = b.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = absmax / fp8_max
    q = (b / scale).to(torch.float8_e4m3fn).view(*orig)
    scale = scale.squeeze(-1).to(torch.float32)
    if n_blocks == 1:
        scale = scale.view(*orig[:-1])
    else:
        scale = scale.view(*orig[:-1], n_blocks)
    return q, scale


# ----------------------------------------------------------------------------
# Hadamard: prefer sglang's fused 128-pt kernel; fallback to a matmul.
# ----------------------------------------------------------------------------

def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """128-point Hadamard transform on the last dim; matches DSA's `rotate_activation`."""
    if _HAS_SGL_HADAMARD:
        return _sgl_hadamard(x)
    # Fallback: manual matmul against a precomputed H matrix (slower, but correct).
    d = x.shape[-1]
    assert (d & (d - 1)) == 0, "Hadamard needs power-of-2 last dim"
    # Sylvester construction.
    H = torch.tensor([[1.]], device=x.device, dtype=torch.float32)
    while H.shape[-1] < d:
        H = torch.cat([torch.cat([H, H], dim=-1), torch.cat([H, -H], dim=-1)], dim=0)
    H = H.to(x.dtype) * (d ** -0.5)
    return x @ H


# ----------------------------------------------------------------------------
# RoPE (partial, on the first ROPE_DIM dims). Cached cos/sin per (S, dim).
# ----------------------------------------------------------------------------

_ROPE_CACHE: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def _rope_cossin(max_pos: int, dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (max_pos, dim, device)
    if key not in _ROPE_CACHE:
        half = dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        t = torch.arange(max_pos, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        _ROPE_CACHE[key] = (freqs.cos(), freqs.sin())
    return _ROPE_CACHE[key]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX-style RoPE on the leading `dim` = 2·half channels of x's last dim."""
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ----------------------------------------------------------------------------
# IndexerModule — real projections + norm, for the full-pipeline timing.
# ----------------------------------------------------------------------------

class IndexerModule(nn.Module):
    """Holds wq_b / wk / k_norm / weights_proj at DSv3.2 shapes."""

    def __init__(self, sh: Shape, device: torch.device):
        super().__init__()
        self.sh = sh
        self.wq_b = nn.Linear(sh.Q_LORA_RANK, sh.H_I * sh.D_I, bias=False, dtype=torch.bfloat16, device=device)
        self.wk = nn.Linear(sh.HIDDEN, sh.D_I, bias=False, dtype=torch.bfloat16, device=device)
        self.k_norm = nn.LayerNorm(sh.D_I, dtype=torch.bfloat16, device=device)
        self.weights_proj = nn.Linear(sh.HIDDEN, sh.H_I, bias=False, dtype=torch.bfloat16, device=device)

    @torch.no_grad()
    def forward_full(
        self,
        x_new: torch.Tensor,        # [T, HIDDEN] bf16 — new-token activations
        q_lora_new: torch.Tensor,   # [T, Q_LORA_RANK] bf16 — MLA-Q-latent for new tokens
        positions: torch.Tensor,    # [T] int32 — RoPE positions for new tokens
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run **all** per-new-token indexer work: proj + norm + RoPE + Hadamard + fp8 quant.
        Returns (q_fp8, q_scale, k_fp8, k_scale, w_eff, positions) ready to feed to fp8_mqa_logits.
        """
        sh = self.sh
        T = x_new.shape[0]

        # 1. Q up-projection (from MLA Q latent).
        q = self.wq_b(q_lora_new).view(T, sh.H_I, sh.D_I)
        # 2. K down-projection + LayerNorm (single MQA head).
        k = self.k_norm(self.wk(x_new))                              # [T, D_I]
        # 3. Head-gate projection.
        w = self.weights_proj(x_new).float() * (sh.H_I ** -0.5)      # [T, H_I]

        # 4. RoPE on the first ROPE_DIM channels of Q (per head) and K.
        cos, sin = _rope_cossin(int(positions.max().item()) + 1, sh.ROPE_DIM, x_new.device)
        cos_t = cos[positions]                                       # [T, ROPE_DIM/2]
        sin_t = sin[positions]
        q_rope = apply_rope(q[..., : sh.ROPE_DIM], cos_t, sin_t)
        k_rope = apply_rope(k[..., : sh.ROPE_DIM], cos_t, sin_t)
        q = torch.cat([q_rope, q[..., sh.ROPE_DIM :]], dim=-1)
        k = torch.cat([k_rope, k[..., sh.ROPE_DIM :]], dim=-1)

        # 5. Hadamard transform (128-pt) on Q per-head and on K.
        q = hadamard_transform(q)
        k = hadamard_transform(k)

        # 6. FP8 per-block quant of Q and K, matching production `act_quant`.
        q_fp8, q_scale = fp8_quant(q, sh.BLOCK_SIZE)
        k_fp8, k_scale = fp8_quant(k, sh.BLOCK_SIZE)

        # 7. Absorb q_scale + softmax_scale into weights (matches production Indexer).
        sm = sh.D_I ** -0.5
        if q_scale.dim() == 2:
            w_eff = (w * q_scale.float() * sm).contiguous()
        else:
            w_eff = (w * q_scale.float().mean(dim=-1) * sm).contiguous()

        return q_fp8.contiguous(), q_scale, k_fp8.contiguous(), k_scale, w_eff, positions


# ----------------------------------------------------------------------------
# Timing primitive (CUDA events, trimmed mean)
# ----------------------------------------------------------------------------

def cuda_time(fn: Callable[[], object], *, warmup: int = 5, iters: int = 25) -> float:
    """Mean wall-time in ms for `fn()`; trims outer decile for stability."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    lo = max(1, len(times) // 10)
    return sum(times[lo:-lo]) / max(1, len(times) - 2 * lo)


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------

def prepare_indexer_inputs(B: int, S: int, sh: Shape, device: torch.device):
    """Inputs for deep_gemm.fp8_mqa_logits with T = B (one query per request)."""
    T = B
    q_bf = torch.randn(T, sh.H_I, sh.D_I, dtype=torch.bfloat16, device=device)
    k_bf = torch.randn(S, sh.D_I, dtype=torch.bfloat16, device=device)
    weights = torch.randn(T, sh.H_I, dtype=torch.float32, device=device) * (sh.H_I ** -0.5)

    q_fp8, q_scale = fp8_quant(q_bf, sh.BLOCK_SIZE)
    k_fp8, k_scale = fp8_quant(k_bf, sh.BLOCK_SIZE)

    # Absorb q_scale + softmax_scale into weights (matches production Indexer).
    sm = sh.D_I ** -0.5
    if q_scale.dim() == 2:
        w_eff = (weights * q_scale.float() * sm).contiguous()
    else:
        w_eff = (weights * q_scale.float().mean(dim=-1) * sm).contiguous()

    ks_arg = k_scale if k_scale.dim() == 1 else k_scale.squeeze(-1)
    ks = torch.zeros(T, dtype=torch.int32, device=device)
    ke = torch.full((T,), S, dtype=torch.int32, device=device)
    return q_fp8.contiguous(), k_fp8.contiguous(), ks_arg.contiguous(), w_eff, ks, ke


def prepare_attention_inputs(B: int, S: int, sh: Shape, device: torch.device):
    """Main attention Q/K/V in [B, H, S, D] layout for SDPA."""
    q = torch.randn(B, sh.H_A, 1, sh.D_A, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, sh.H_KV, S, sh.D_A, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, sh.H_KV, S, sh.D_A, dtype=torch.bfloat16, device=device)
    return q, k, v


# ----------------------------------------------------------------------------
# Hot-path callables for each (B, S)
# ----------------------------------------------------------------------------

def make_step_fns(B: int, S: int, sh: Shape, device: torch.device):
    q_fp8, k_fp8, k_scale, w_eff, ks, ke = prepare_indexer_inputs(B, S, sh, device)
    q_bf, k_bf, v_bf = prepare_attention_inputs(B, S, sh, device)
    q_bf_sparse, k_bf_sparse, v_bf_sparse = prepare_attention_inputs(B, sh.TOPK, sh, device)

    # Full-indexer inputs: one decode step adds T = B new tokens on top of an
    # existing S-long cache. The cache is pre-quantised (as it would be in prod).
    T = B
    module = IndexerModule(sh, device)
    x_new = torch.randn(T, sh.HIDDEN, dtype=torch.bfloat16, device=device)
    q_lora_new = torch.randn(T, sh.Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    positions_new = torch.arange(S, S + T, device=device, dtype=torch.int32)

    logits_buf: Optional[torch.Tensor] = None

    def step_mqa_logits():
        """Score-kernel only (steps 7 of §2.3): fp8_mqa_logits call."""
        nonlocal logits_buf
        logits_buf = deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8, k_scale), w_eff, ks, ke)

    def step_topk():
        assert logits_buf is not None
        _ = logits_buf.topk(min(sh.TOPK, S), dim=-1).indices

    def step_full_indexer():
        """FULL per-decode-step indexer: proj + norm + RoPE + Hadamard + fp8-quant
        for the T new tokens, then fp8_mqa_logits against the S-long cache, then topk.
        This is what a deployed DSA layer actually spends per decode step."""
        q_fp8_new, _, k_fp8_new, k_scale_new, w_new, _ = module.forward_full(
            x_new, q_lora_new, positions_new,
        )
        # In a real decode we would also cat k_fp8_new onto the cached K
        # (deep_gemm can accept it), but the S read dominates so we just re-use
        # the pre-cached [S, D_I] buffer to keep the S-scaling clean.
        logits = deep_gemm.fp8_mqa_logits(q_fp8_new, (k_fp8, k_scale), w_new, ks, ke)
        _ = logits.topk(min(sh.TOPK, S), dim=-1).indices

    def step_dense_fa():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(
                q_bf, k_bf, v_bf, is_causal=False, enable_gqa=True,
            )

    def step_sparse_fa():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(
                q_bf_sparse, k_bf_sparse, v_bf_sparse,
                is_causal=False, enable_gqa=True,
            )

    def step_gather():
        # k[topk_idx], v[topk_idx] — cost of materialising the sparse KV slice.
        idx = torch.randint(0, S, (B, sh.TOPK), device=device, dtype=torch.int64)
        # k_bf is [B, H_kv, S, D]; gather along S.
        idx_exp = idx[:, None, :, None].expand(-1, sh.H_KV, -1, sh.D_A)
        _ = torch.gather(k_bf, 2, idx_exp)
        _ = torch.gather(v_bf, 2, idx_exp)

    # Prime the logits buffer so step_topk has something to sort.
    step_mqa_logits()
    torch.cuda.synchronize()

    return {
        "mqa_logits": step_mqa_logits,
        "topk": step_topk,
        "full_indexer": step_full_indexer,
        "gather": step_gather,
        "dense_fa": step_dense_fa,
        "sparse_fa": step_sparse_fa,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run(B_list: List[int], S_list: List[int], sh: Shape, warmup: int, iters: int) -> List[Dict]:
    device = torch.device("cuda")
    rows: List[Dict] = []
    total = len(B_list) * len(S_list)
    idx = 0
    for B in B_list:
        for S in S_list:
            idx += 1
            free, total_mem = torch.cuda.mem_get_info()
            print(f"[{idx}/{total}] B={B:<3d} S={S:>8d}   free={free/2**30:.1f} GB", flush=True)
            try:
                fns = make_step_fns(B, S, sh, device)
            except torch.OutOfMemoryError as e:
                print(f"    OOM building inputs: {e}")
                gc.collect(); torch.cuda.empty_cache()
                continue
            row: Dict = {"B": B, "S": S}
            for name, fn in fns.items():
                try:
                    row[name + "_ms"] = cuda_time(fn, warmup=warmup, iters=iters)
                except Exception as e:  # noqa: BLE001
                    print(f"    {name} skipped: {type(e).__name__}: {str(e)[:120]}")
                    row[name + "_ms"] = float("nan")

            # Two views of "indexer": score-kernel-only and full pipeline.
            row["scorer_only_ms"] = row.get("mqa_logits_ms", float("nan")) + row.get("topk_ms", 0.0)
            row["indexer_ms"] = row.get("full_indexer_ms", float("nan"))
            row["setup_overhead_ms"] = row["indexer_ms"] - row["scorer_only_ms"]
            row["dsa_total_ms"] = (
                row["indexer_ms"] + row.get("gather_ms", 0.0) + row.get("sparse_fa_ms", 0.0)
            )
            row["speedup_vs_dense"] = row.get("dense_fa_ms", float("nan")) / max(row["dsa_total_ms"], 1e-9)
            row["indexer_frac"] = row["indexer_ms"] / max(row["dsa_total_ms"], 1e-9)
            row["idx_over_sparse"] = row["indexer_ms"] / max(row.get("sparse_fa_ms", 1e-9), 1e-9)
            row["sparse_over_dense"] = row.get("sparse_fa_ms", float("nan")) / max(row.get("dense_fa_ms", 1e-9), 1e-9)
            rows.append(row)
            del fns
            gc.collect(); torch.cuda.empty_cache()
    return rows


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_table(rows: List[Dict]) -> None:
    print()
    print("=" * 118)
    print("  Q1: indexer cost vs sparse-attention cost   |   Q2: sparse vs dense")
    print("=" * 118)
    hdr = (
        f"{'B':>3} {'S':>10} | "
        f"{'mqa_lgt':>8} {'topk':>7} {'setup':>7} {'indexer':>8} | "
        f"{'gather':>7} {'sparse':>7} {'dense':>7} | "
        f"{'idx/spa':>7} {'spa/dns':>7} {'dns/dsa':>7} {'idx%':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['B']:>3} {r['S']:>10} | "
            f"{r.get('mqa_logits_ms', float('nan')):>8.3f} "
            f"{r.get('topk_ms', float('nan')):>7.3f} "
            f"{r.get('setup_overhead_ms', float('nan')):>7.3f} "
            f"{r['indexer_ms']:>8.3f} | "
            f"{r.get('gather_ms', float('nan')):>7.3f} "
            f"{r.get('sparse_fa_ms', float('nan')):>7.3f} "
            f"{r.get('dense_fa_ms', float('nan')):>7.3f} | "
            f"{r['idx_over_sparse']:>7.2f} "
            f"{r['sparse_over_dense']:>7.3f} "
            f"{r['speedup_vs_dense']:>7.2f} "
            f"{100*r['indexer_frac']:>4.0f}%"
        )
    print()
    print("mqa_lgt = deep_gemm.fp8_mqa_logits (score kernel only).")
    print("setup   = wq_b + wk + k_norm + weights_proj + RoPE + Hadamard + FP8 quant.")
    print("indexer = full pipeline (setup + mqa_logits + topk).  dsa = indexer + gather + sparse.")
    print("idx/spa = indexer / sparse-FA.   spa/dns = sparse-FA / dense-FA.   dns/dsa = dense-FA / dsa_total.")


def plot(rows: List[Dict], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    by_B: Dict[int, List[Dict]] = {}
    for r in rows:
        by_B.setdefault(r["B"], []).append(r)
    colours = {1: "tab:blue", 16: "tab:orange"}

    # Q1 — indexer vs sparse-FA
    for B, rs in by_B.items():
        rs = sorted(rs, key=lambda r: r["S"])
        S_vals = [r["S"] for r in rs]
        c = colours.get(B, None)
        ax1.plot(S_vals, [r["indexer_ms"] for r in rs], "o-", color=c, label=f"indexer  (B={B})")
        ax1.plot(S_vals, [r.get("sparse_fa_ms", float("nan")) for r in rs], "s--", color=c, label=f"sparse-FA (B={B})")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("KV sequence length S")
    ax1.set_ylabel("time (ms)")
    ax1.set_title("Q1 — Indexer vs sparse-FA cost (B200)\n(Q_len=1 per request, decode)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # Q2 — sparse-FA vs dense-FA
    for B, rs in by_B.items():
        rs = sorted(rs, key=lambda r: r["S"])
        S_vals = [r["S"] for r in rs]
        c = colours.get(B, None)
        ax2.plot(S_vals, [r.get("dense_fa_ms", float("nan")) for r in rs], "^-", color=c, label=f"dense-FA  (B={B})")
        ax2.plot(S_vals, [r.get("sparse_fa_ms", float("nan")) for r in rs], "s--", color=c, label=f"sparse-FA (B={B})")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("KV sequence length S")
    ax2.set_ylabel("time (ms)")
    ax2.set_title("Q2 — Sparse-FA vs dense-FA cost (B200)\n(sparse = FA over top-K=2048 tokens)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"wrote plot: {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--B", nargs="+", type=int, default=[1, 16])
    p.add_argument(
        "--S", nargs="+", type=int, default=[32_768, 131_072, 1_048_576],
        help="KV sequence lengths to sweep (default: 32K 128K 1M)",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--csv", default=str(Path(__file__).resolve().parents[1] / "results" / "legacy" / "bench_indexer_cost.csv"))
    p.add_argument("--plot", default=str(Path(__file__).resolve().parents[1] / "results" / "legacy" / "bench_indexer_cost.png"))
    args = p.parse_args()

    print(f"device = {torch.cuda.get_device_name(0)}   torch = {torch.__version__}")
    print(f"B sweep = {args.B}   S sweep = {args.S}   warmup/iters = {args.warmup}/{args.iters}")

    sh = Shape()
    rows = run(args.B, args.S, sh, args.warmup, args.iters)
    print_table(rows)
    write_csv(rows, args.csv)
    print(f"wrote csv:  {args.csv}")
    plot(rows, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())

