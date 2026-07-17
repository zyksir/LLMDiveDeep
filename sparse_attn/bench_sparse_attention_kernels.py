"""Kernel-level micro-benchmark for DeepSeek-style sparse attention.

This file is the **driver** only: it sweeps sequence length / sparsity and
prints the tables. All the attention math (the lightning-indexer hot path,
FP8 quant, the FA4 dense backend, the block-sparse FA backend) lives in
`DSA.py` next to this file and is imported below. Splitting the two keeps
"what DSA does" separate from "how we time it".

Requirements
------------
See `requirements.txt` next to this file for the full list. In short:

- Python 3.10+ with a CUDA-enabled `torch>=2.6` and `triton>=3.0`.
- `deep_gemm` with `fp8_mqa_logits` (a torch FP32 fallback is used if
  missing -- correct but ~50x slower).
- `flash_attn.cute` (FA4 via CuTe DSL) for the dense baseline; falls
  back to `sglang.jit_kernel.flash_attention_v4`.
- The block-sparse Triton kernel from sglang's `b10_kernels/sparse_attn/`.
  Auto-discovered via `SGLANG_B10_KERNELS_DIR` or a set of well-known
  workspace paths (see `DSA.resolve_b10_sparse_path`).
- An NVIDIA Hopper (SM_90) or Blackwell (SM_100) GPU. Numbers in the README
  are from a single NVIDIA B200.

Overview
--------
On synthetic tensors with no model loading and no server context:

    PART A  dense attention baseline (FA4 on GQA shapes).
    PART B  block-sparse FA at the *same* shapes with a random top-k LUT --
            the algorithmic cost of the sparse-attention step, isolated from
            the indexer.
    PART C  the lightning indexer in isolation -- the FP8 MQA logits + top-k
            step that DSv3.2 / GLM-5 run *before* the sparse attention step.
    PART D  headline table: (indexer + sparse-FA) vs dense-FA, with the
            indexer's cost amortized over `index_topk_freq` layers to match
            GLM-5.2's IndexShare pattern.
    PART E  quality: cos-sim of sparse-attention output vs dense-attention
            output, for both a random top-k baseline (adversarial floor) and
            an oracle block-max top-k (ceiling).
    PART F  LINEAR attention: Qwen3-Next's Gated Delta Net via sglang's FLA
            `chunk_gated_delta_rule` Triton kernel. Shown side-by-side with
            the dense baseline so the O(S) vs O(S^2) scaling gap is visible.

NOTE (language-model attention first): the block-sparse kernel PART B/D/E use
is the generic Triton block-sparse kernel that also backs WanVideo's *video*
sparse attention. We reuse it here purely as the LM sparse-attention step;
the video-specific (VSA) investigation is deferred.

Layout: q,k,v are [B, S, H, D] bf16 (BSHD). Backends that need BHSD do the
transpose internally so all sites share the same `run` signature.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

# Make `common` (one dir up) and `DSA` (same dir) importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_LLMDIVEDEEP = os.path.dirname(_HERE)
if _LLMDIVEDEEP not in sys.path:
    sys.path.insert(0, _LLMDIVEDEEP)

from common.bench_utils import bench_function, print_section  # noqa: E402
from DSA import (  # noqa: E402
    INDEXER_BLOCK_SIZE,
    INDEXER_HEAD_DIM,
    INDEXER_N_HEADS,
    INDEXER_TOPK,
    SPARSE_BLOCK,
    IndexerSimulator,
    env_report,
    register_dense_fn,
    register_fa4,
    register_sparse_fa,
    try_import_deep_gemm,
)

# Sequence lengths we sweep over.
DEFAULT_SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


# ===========================================================================
# PART A -- DENSE ATTENTION BASELINE (FA4 on GQA shapes)
# ===========================================================================

def bench_dense_gqa(
    seq_lens,
    causal_modes,
    warmup: int,
    iters: int,
) -> None:
    """PART A: dense attention on GQA shapes. The 'before DSA' baseline.

    Only FA4 is reported. `sgl_fa4` is the same kernel family (redundant as a
    column) and cuDNN SDPA is unfair here -- its BSHD<->BHSD transpose would be
    timed inside the attention call -- so neither is shown.
    """
    print_section("PART A  --  Dense attention baseline (FA4, GQA)")
    fa4 = register_fa4()
    if fa4 is None:
        print("  no FA4 dense backend available; skipping PART A")
        return
    backends = {"fa4 (cute)": fa4}

    device = torch.device("cuda")
    dtype = torch.bfloat16
    # GQA shapes that approximate DSv3.2 / GLM-5 per-rank. head_dim=128 is what
    # FA4 supports and what the indexer also uses.
    H, H_kv, D = 32, 8, 128
    batch = 1

    for causal in causal_modes:
        print(f"\n  [{'causal' if causal else 'non-causal'}]   H={H} H_kv={H_kv} D={D}")
        header = f"  {'seq_len':>10}"
        for name in backends:
            header += f"  {name + ' (ms)':>18}  {name + ' TFLOPS':>18}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for S in seq_lens:
            sm_scale = D ** -0.5
            q = torch.randn(batch, S, H, D, dtype=dtype, device=device)
            k = torch.randn(batch, S, H_kv, D, dtype=dtype, device=device)
            v = torch.randn(batch, S, H_kv, D, dtype=dtype, device=device)
            flops = (2.0 if causal else 4.0) * batch * H * S * S * D
            row = f"  {S:>10}"
            for name, fn in backends.items():
                try:
                    ms = bench_function(
                        fn, q, k, v, S, causal, sm_scale,
                        warmup=warmup, iters=iters,
                    )
                    tflops = flops / (ms / 1000) / 1e12
                    row += f"  {ms:>18.3f}  {tflops:>18.2f}"
                except Exception as e:  # noqa: BLE001
                    row += f"  {'-- err --':>18}  {str(e)[:18]:>18}"
            print(row)


# ===========================================================================
# PART B -- BLOCK-SPARSE FA on GQA shapes (the "sparse-MLA" algorithmic core)
# ===========================================================================

def bench_sparse_gqa(
    seq_lens,
    sparsities,
    warmup: int,
    iters: int,
) -> None:
    """PART B: block-sparse FA on the same GQA shape as PART A.

    Reports absolute ms and speedup vs the dense baseline at the *same* shape.
    """
    print_section("PART B  --  Block-sparse FA on GQA shapes (sparse-MLA core, non-causal)")
    pair = register_sparse_fa()
    if pair is None:
        print("  sparse FA backend unavailable; skipping PART B")
        return
    build_lut, run = pair

    # Pre-register an FA4 dense backend so we can report speedup in-place.
    dense_fn = register_dense_fn()
    if dense_fn is None:
        print("  no dense backend for speedup column")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    # Must be MHA for the block-sparse kernel (it does not implement GQA).
    H, D = 24, 128
    batch = 1

    print(f"\n  H={H} H_kv={H} D={D}   block={SPARSE_BLOCK}")
    header = f"  {'seq_len':>10}  {'sparsity':>10}"
    header += f"  {'sparse ms':>14}  {'dense ms':>14}  {'speedup':>10}  {'sparse TFLOPS':>15}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for S in seq_lens:
        if S % SPARSE_BLOCK != 0:
            continue
        sm_scale = D ** -0.5
        q = torch.randn(batch, S, H, D, dtype=dtype, device=device)
        k = torch.randn(batch, S, H, D, dtype=dtype, device=device)
        v = torch.randn(batch, S, H, D, dtype=dtype, device=device)

        if dense_fn is not None:
            try:
                dense_ms = bench_function(
                    dense_fn, q, k, v, S, False, sm_scale,
                    warmup=warmup, iters=iters,
                )
            except Exception:  # noqa: BLE001
                dense_ms = float("nan")
        else:
            dense_ms = float("nan")

        for sp in sparsities:
            idx, num, topk = build_lut(q, S, sp)
            try:
                ms = bench_function(
                    run, q, k, v, idx, num, sm_scale,
                    warmup=warmup, iters=iters,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  S={S} sparsity={sp}: sparse FA err: {e}")
                continue
            # FLOPs at the selected work only.
            flops = 4.0 * batch * H * S * (topk * SPARSE_BLOCK) * D
            tflops = flops / (ms / 1000) / 1e12
            speedup = dense_ms / ms if not math.isnan(dense_ms) else float("nan")
            print(
                f"  {S:>10}  {sp:>10.3f}"
                f"  {ms:>14.3f}  {dense_ms:>14.3f}  {speedup:>10.2f}  {tflops:>15.2f}"
            )


# ===========================================================================
# PART C -- THE LIGHTNING INDEXER IN ISOLATION
# ===========================================================================

def bench_indexer(
    seq_lens,
    warmup: int,
    iters: int,
) -> None:
    """PART C: the lightning indexer in isolation.

    Reports four numbers per sequence length:

        proj_ms     : wq_b + wk + weights_proj forward (bf16)
        quant_ms    : per-block FP8 quantize of Q and K
        logits_ms   : deep_gemm.fp8_mqa_logits + topk (the bulk of indexer cost)
        total_ms    : sum (one indexer call)

    The point is to give direct numbers for "how much does the indexer cost vs
    the sparse attention that follows it" alongside the PART B table.
    """
    print_section("PART C  --  Lightning indexer (FP8 MQA logits + top-k) in isolation")
    dg = try_import_deep_gemm()
    print(f"  deep_gemm.fp8_mqa_logits: {'present' if dg else 'MISSING -- using torch FP32 fallback'}")
    print(
        f"  H_idx={INDEXER_N_HEADS}  D_idx={INDEXER_HEAD_DIM}  topk={INDEXER_TOPK}  "
        f"block_size={INDEXER_BLOCK_SIZE}"
    )

    header = (
        f"  {'seq_len':>10}  {'proj ms':>10}  {'quant ms':>10}"
        f"  {'logits+topk ms':>16}  {'total ms':>10}  {'GB/s':>10}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for S in seq_lens:
        try:
            sim = IndexerSimulator(S=S)
        except RuntimeError as e:  # OOM at very long S
            print(f"  {S:>10}  OOM: {e}")
            continue

        # Step 1: projections.
        try:
            proj_ms = bench_function(sim.projections_step, warmup=warmup, iters=iters)
        except Exception as e:  # noqa: BLE001
            print(f"  {S:>10}  proj err: {e}")
            continue

        q_bf16, k_bf16, _ = sim.projections_step()

        # Step 2: FP8 quantize.
        quant_ms = bench_function(
            sim.quantize_step, q_bf16, k_bf16, warmup=warmup, iters=iters,
        )

        # Step 3: logits + topk.
        try:
            sim.precompute_fp8()
            logits_ms = bench_function(sim.logits_step, warmup=warmup, iters=iters)
        except Exception as e:  # noqa: BLE001
            print(f"  {S:>10}  logits err: {e}")
            continue

        total_ms = proj_ms + quant_ms + logits_ms
        # Approximate effective bandwidth for the logits step. KV cache scan
        # dominates: S tokens * (head_dim FP8 + scale FP32 / block_size).
        kv_bytes = S * (INDEXER_HEAD_DIM + (INDEXER_HEAD_DIM // INDEXER_BLOCK_SIZE) * 4) * S
        gbs = kv_bytes / (logits_ms / 1000) / 1e9
        print(
            f"  {S:>10}  {proj_ms:>10.3f}  {quant_ms:>10.3f}"
            f"  {logits_ms:>16.3f}  {total_ms:>10.3f}  {gbs:>10.1f}"
        )


# ===========================================================================
# PART D -- combined view: (indexer + sparse-FA) vs dense-FA at same shape
# ===========================================================================

def bench_combined(
    seq_lens,
    sparsity: float,
    index_topk_freq: int,
    warmup: int,
    iters: int,
) -> None:
    """Headline table: 'what does adding DSA buy us, after paying the indexer?'

    Computes for each S:

        dense_ms     = FA4 dense baseline
        sparse_ms    = block-sparse FA at the given sparsity
        indexer_ms   = lightning indexer (PART C total) / index_topk_freq
        attention_ms = sparse_ms + indexer_ms
        speedup      = dense_ms / attention_ms

    `index_topk_freq=4` is GLM-5.2's IndexShare default; `=1` is DSv3.2 / GLM-5.
    """
    print_section(
        f"PART D  --  Headline: (indexer + sparse-FA) vs dense-FA  "
        f"(sparsity={sparsity}, index_topk_freq={index_topk_freq})"
    )
    dense_fn = register_dense_fn()
    sparse_pair = register_sparse_fa()
    if dense_fn is None or sparse_pair is None:
        print("  combined bench needs both dense FA and sparse FA; skipping")
        return
    build_lut, sparse_fn = sparse_pair

    device = torch.device("cuda")
    dtype = torch.bfloat16
    H, D = 24, 128
    batch = 1

    print(
        f"  H={H} D={D}  topk_idx={INDEXER_TOPK}  block={SPARSE_BLOCK}\n"
        f"  Note: indexer cost amortized over {index_topk_freq} layers (IndexShare)."
    )
    header = (
        f"  {'seq_len':>10}  {'dense ms':>10}  {'indexer ms':>12}"
        f"  {'sparse ms':>10}  {'total ms':>10}  {'speedup':>10}  {'idx frac':>10}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for S in seq_lens:
        if S % SPARSE_BLOCK != 0:
            continue
        sm_scale = D ** -0.5
        q = torch.randn(batch, S, H, D, dtype=dtype, device=device)
        k = torch.randn(batch, S, H, D, dtype=dtype, device=device)
        v = torch.randn(batch, S, H, D, dtype=dtype, device=device)

        try:
            dense_ms = bench_function(
                dense_fn, q, k, v, S, False, sm_scale,
                warmup=warmup, iters=iters,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {S:>10}  dense err: {e}")
            continue

        idx, num, _ = build_lut(q, S, sparsity)
        try:
            sparse_ms = bench_function(
                sparse_fn, q, k, v, idx, num, sm_scale,
                warmup=warmup, iters=iters,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {S:>10}  sparse err: {e}")
            continue

        try:
            sim = IndexerSimulator(S=S)
            sim.precompute_fp8()
            proj_ms = bench_function(sim.projections_step, warmup=warmup, iters=iters)
            q_bf16, k_bf16, _ = sim.projections_step()
            quant_ms = bench_function(
                sim.quantize_step, q_bf16, k_bf16, warmup=warmup, iters=iters,
            )
            logits_ms = bench_function(sim.logits_step, warmup=warmup, iters=iters)
            indexer_ms_raw = proj_ms + quant_ms + logits_ms
        except Exception as e:  # noqa: BLE001
            print(f"  {S:>10}  indexer err: {e}")
            continue

        indexer_ms = indexer_ms_raw / max(index_topk_freq, 1)
        total_ms = sparse_ms + indexer_ms
        speedup = dense_ms / total_ms
        idx_frac = indexer_ms / total_ms
        print(
            f"  {S:>10}  {dense_ms:>10.3f}  {indexer_ms:>12.3f}"
            f"  {sparse_ms:>10.3f}  {total_ms:>10.3f}  {speedup:>10.2f}  {idx_frac:>10.2%}"
        )


# ===========================================================================
# PART E -- QUALITY: cosine similarity between sparse and dense attention
# ===========================================================================
#
# The timing tables above only tell us *how fast* sparse attention is.
# The user question "what is the cosine similarity?" is really asking
# "how close does sparse attention get to dense attention on the same
# inputs?" -- i.e. how much accuracy do we sacrifice per unit of speedup.
#
# We compare two selection policies at matched sparsity:
#
#   1) RANDOM  top-k blocks (adversarial baseline: the sparse kernel with
#      no indexer at all). This is the *lower bound* on quality.
#   2) ORACLE  top-k blocks (block-max of Q @ K^T). This is what a perfect
#      indexer would pick, and is the *upper bound* on quality that a top-k
#      selection policy can achieve.
#
# The real DSA / GLM-5 / MiniMax indexer sits *between* random and oracle.
# ===========================================================================


def _dense_reference(q, k, v, sm_scale) -> torch.Tensor:
    """SDPA reference for cos-sim ground truth. Streams over query blocks
    with FP32 accumulation so we don't materialise a full [B,H,S,S] score
    matrix in FP32 (which is 4 * B * H * S^2 bytes -- 96 GB at B=1 H=24
    S=32K). Returns BSHD.
    """
    B, S, H, D = q.shape
    q_bhsd = q.transpose(1, 2)  # BHSD
    k_bhsd = k.transpose(1, 2)
    v_bhsd = v.transpose(1, 2)
    # Query-block streaming; 1024 was chosen so each per-block scratch
    # tensor is well under 1 GB even at S = 128K.
    q_block = 1024
    out = torch.empty(B, H, S, D, dtype=torch.float32, device=q.device)
    k32 = k_bhsd.to(torch.float32)
    v32 = v_bhsd.to(torch.float32)
    for i in range(0, S, q_block):
        qi = q_bhsd[:, :, i:i + q_block].to(torch.float32)
        scores = torch.matmul(qi, k32.transpose(-1, -2)) * sm_scale
        p = torch.softmax(scores, dim=-1)
        out[:, :, i:i + q_block] = torch.matmul(p, v32)
    return out.transpose(1, 2)  # BSHD


def _block_max_topk(
    q: torch.Tensor, k: torch.Tensor, block: int, topk: int,
):
    """Oracle indexer: score each (query-block, kv-block) pair with the
    block-max of the *signed* Q @ K^T, then pick top-k KV blocks per query
    block, per head. Signed is what the softmax cares about (positive
    logits dominate the probability mass); using |QK^T| would waste top-k
    slots on keys with large *negative* correlation.

    Streamed over query blocks so we never materialise a full [B, H, Nq,
    Nkv, block, block] scratch tensor (~48 GB at S=32K, H=24, block=64).
    Used only for quality analysis, never on the hot path.
    """
    B, S, H, D = q.shape
    Nq = S // block
    Nkv = S // block
    # Reshape into (B, H, Nkv, block, D) once for K; reuse across Q blocks.
    k_bhkbd = k.view(B, Nkv, block, H, D).permute(0, 3, 1, 2, 4).contiguous()
    per_block = torch.empty(B, H, Nq, Nkv, dtype=torch.float32, device=q.device)
    q_stride = max(1, 128 // block)  # process ~128-token Q chunks
    for qs in range(0, Nq, q_stride):
        qe = min(qs + q_stride, Nq)
        q_chunk = q.view(B, Nq, block, H, D)[:, qs:qe]              # B nQ block H D
        q_chunk = q_chunk.permute(0, 3, 1, 2, 4).contiguous().float()  # B H nQ block D
        # (B, H, nQ, block, D) x (B, H, Nkv, block, D) -> B H nQ Nkv block block
        scores = torch.einsum("bhqid,bhkjd->bhqkij", q_chunk, k_bhkbd.float())
        per_block[:, :, qs:qe] = scores.amax(dim=(-1, -2))
    idx = per_block.topk(topk, dim=-1).indices.to(torch.int32).contiguous()
    num = torch.full((B, H, Nq), topk, dtype=torch.int32, device=q.device)
    return idx, num


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.reshape(-1).to(torch.float32)
    b32 = b.reshape(-1).to(torch.float32)
    num = (a32 * b32).sum()
    denom = a32.norm() * b32.norm()
    return (num / denom.clamp_min(1e-12)).item()


def bench_quality(
    seq_lens,
    sparsities,
) -> None:
    """PART E: cos-sim(sparse, dense) at matched sparsity, random vs oracle.

    For each (S, sparsity), reports:

        random cos    : cos-sim of random-top-k sparse vs dense
        oracle cos    : cos-sim of block-max-top-k sparse vs dense
        speedup       : sparse ms / dense ms (from PART B kernels)
    """
    print_section("PART E  --  Quality: cos-sim of sparse vs dense (random vs oracle top-k)")
    sparse_pair = register_sparse_fa()
    dense_fn = register_dense_fn()
    if sparse_pair is None or dense_fn is None:
        print("  needs sparse FA + a dense FA backend; skipping PART E")
        return
    _, sparse_fn = sparse_pair

    device = torch.device("cuda")
    dtype = torch.bfloat16
    H, D = 24, 128
    batch = 1

    header = (
        f"  {'seq_len':>10}  {'sparsity':>10}  {'random cos':>12}"
        f"  {'oracle cos':>12}  {'sparse ms':>12}  {'dense ms':>12}  {'speedup':>9}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for S in seq_lens:
        if S % SPARSE_BLOCK != 0:
            continue
        sm_scale = D ** -0.5
        # Same inputs are re-used for random, oracle and dense so cos-sim
        # is a fair apples-to-apples comparison. We *inject* heavy-hitter
        # structure -- ~5% of K rows have a large norm -- because that is
        # what the softmax actually sees in real LLMs (a small number of
        # "attention sink" / "heavy hitter" tokens absorb most of the
        # probability mass). With i.i.d. Gaussian K, block-max top-k and
        # random top-k are indistinguishable and "oracle" is meaningless.
        q = torch.randn(batch, S, H, D, dtype=dtype, device=device) * 0.5
        k = torch.randn(batch, S, H, D, dtype=dtype, device=device) * 0.5
        v = torch.randn(batch, S, H, D, dtype=dtype, device=device) * 0.5
        hh_frac = 0.05
        n_hh = max(1, int(hh_frac * S))
        hh_idx = torch.randperm(S, device=device)[:n_hh]
        k[:, hh_idx] = k[:, hh_idx] * 8.0
        v[:, hh_idx] = v[:, hh_idx] * 8.0

        # Dense reference (FA4 kernel output; matches production).
        try:
            _ = dense_fn(q, k, v, S, False, sm_scale)
            dense_ms = bench_function(
                dense_fn, q, k, v, S, False, sm_scale, warmup=3, iters=10,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {S:>10}  dense err: {e}")
            continue

        # FP32 SDPA reference for cos-sim -- FA4 output tends to have small
        # bit-level noise vs an FP32 SDPA reference. Use FP32 as the ground
        # truth so both random and oracle are measured against the same
        # baseline.
        dense_ref = _dense_reference(q, k, v, sm_scale)

        Nkv = S // SPARSE_BLOCK
        for sp in sparsities:
            topk = max(1, int(round(sp * Nkv)))

            # 1) Random top-k baseline.
            scores_rand = torch.rand(batch, H, S // SPARSE_BLOCK, Nkv, device=device)
            idx_rand = scores_rand.topk(topk, dim=-1).indices.to(torch.int32).contiguous()
            num_rand = torch.full(
                (batch, H, S // SPARSE_BLOCK), topk, dtype=torch.int32, device=device,
            )

            # 2) Oracle top-k -- block-max of QK^T.
            try:
                idx_ora, num_ora = _block_max_topk(q, k, SPARSE_BLOCK, topk)
            except RuntimeError as e:
                print(f"  {S:>10}  oracle OOM: {e}")
                idx_ora = None

            try:
                out_rand = sparse_fn(q, k, v, idx_rand, num_rand, sm_scale)
                sparse_ms = bench_function(
                    sparse_fn, q, k, v, idx_rand, num_rand, sm_scale,
                    warmup=3, iters=10,
                )
                cos_rand = _cosine_similarity(out_rand, dense_ref)
            except Exception as e:  # noqa: BLE001
                cos_rand = float("nan")
                sparse_ms = float("nan")
                print(f"  {S:>10}  random sparse err: {e}")

            if idx_ora is not None:
                try:
                    out_ora = sparse_fn(q, k, v, idx_ora, num_ora, sm_scale)
                    cos_ora = _cosine_similarity(out_ora, dense_ref)
                except Exception as e:  # noqa: BLE001
                    cos_ora = float("nan")
                    print(f"  {S:>10}  oracle sparse err: {e}")
            else:
                cos_ora = float("nan")

            speedup = dense_ms / sparse_ms if not math.isnan(sparse_ms) else float("nan")
            print(
                f"  {S:>10}  {sp:>10.3f}"
                f"  {cos_rand:>12.4f}  {cos_ora:>12.4f}"
                f"  {sparse_ms:>12.3f}  {dense_ms:>12.3f}  {speedup:>9.2f}"
            )


# ===========================================================================
# PART F -- LINEAR ATTENTION (Qwen3-Next Gated Delta Net, via FLA)
# ===========================================================================
#
# Everything above (PART A-E) is a variant of *softmax* attention. The other
# major escape hatch from O(N^2) is to drop the softmax entirely and use a
# recurrent update -- what sglang calls "linear attention". Qwen3-Next's
# Gated Delta Net (GDN) is the canonical instance, and its kernel is the
# `chunk_gated_delta_rule` Triton kernel from sglang's Flash Linear
# Attention (FLA) submodule.
# ===========================================================================


def _try_import_gdn():
    """Return the GDN Triton entry point, or None if unavailable."""
    try:
        from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
        return chunk_gated_delta_rule
    except Exception:  # noqa: BLE001
        return None


def bench_linear_attention(
    seq_lens,
    warmup: int,
    iters: int,
) -> None:
    """PART F: Qwen3-Next Gated Delta Net forward on synthetic tensors.

    Reports absolute ms and, when a dense FA backend is available, speedup
    vs dense FA4 at the same (H, D). The GDN forward is linear in S, so we
    expect the speedup to grow monotonically with S while dense's O(S^2)
    term explodes.
    """
    print_section("PART F  --  Linear attention: Qwen3-Next Gated Delta Net (FLA)")
    gdn = _try_import_gdn()
    if gdn is None:
        print("  sglang FLA chunk_gated_delta_rule not importable; skipping PART F")
        print("  (this is expected outside sglang containers)")
        return
    dense_fn = register_dense_fn()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    # Match PART A / D shape so the comparison is fair.
    B, H, D = 1, 24, 128

    print(f"  B={B} H={H} D={D}  kernel=chunk_gated_delta_rule (Triton, bf16 q/k/v, fp32 g/beta)")
    header = (
        f"  {'seq_len':>10}  {'linear ms':>10}  {'dense ms':>10}"
        f"  {'speedup':>9}  {'note':>28}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for S in seq_lens:
        try:
            q = torch.randn(B, S, H, D, dtype=dtype, device=device)
            k = torch.randn(B, S, H, D, dtype=dtype, device=device)
            v = torch.randn(B, S, H, D, dtype=dtype, device=device)
            g = torch.randn(B, S, H, dtype=torch.float32, device=device)
            beta = torch.rand(B, S, H, dtype=torch.float32, device=device)
            initial_state = torch.zeros(B, H, D, D, dtype=torch.float32, device=device)
            init_idx = torch.arange(B, dtype=torch.int32, device=device)
        except RuntimeError as e:
            print(f"  {S:>10}  OOM: {e}")
            continue

        def run_gdn():
            return gdn(
                q, k, v, g, beta, scale=D ** -0.5,
                initial_state=initial_state,
                initial_state_indices=init_idx,
                use_qk_l2norm_in_kernel=True,
            )

        try:
            ms = bench_function(run_gdn, warmup=warmup, iters=iters)
        except Exception as e:  # noqa: BLE001
            print(f"  {S:>10}  gdn err: {e}")
            continue

        note = "O(S) linear recurrence"
        if dense_fn is not None:
            try:
                dense_ms = bench_function(
                    dense_fn, q, k, v, S, False, D ** -0.5,
                    warmup=warmup, iters=iters,
                )
                speedup = dense_ms / ms
            except Exception:  # noqa: BLE001
                dense_ms = float("nan")
                speedup = float("nan")
        else:
            dense_ms = float("nan")
            speedup = float("nan")

        print(
            f"  {S:>10}  {ms:>10.3f}  {dense_ms:>10.3f}  {speedup:>9.2f}  {note:>28}"
        )


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dense", action="store_true", help="PART A: dense FA baseline")
    p.add_argument("--sparse_gqa", action="store_true", help="PART B: block-sparse FA")
    p.add_argument("--indexer", action="store_true", help="PART C: lightning indexer")
    p.add_argument("--combined", action="store_true", help="PART D: headline combined table")
    p.add_argument("--quality", action="store_true", help="PART E: cos-sim sparse vs dense")
    p.add_argument("--linear", action="store_true",
                   help="PART F: linear attention (Qwen3-Next GDN via FLA)")
    p.add_argument("--all", action="store_true", help="run all parts")
    p.add_argument(
        "--seq_lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS,
    )
    p.add_argument("--sparsity", type=float, default=0.05,
                   help="fraction of KV blocks to attend (PART B + D)")
    p.add_argument("--sparsities", type=float, nargs="+", default=None,
                   help="multiple sparsities to sweep in PART B")
    p.add_argument("--index_topk_freq", type=int, default=1,
                   help="GLM-5.2 IndexShare period; 1=DSv3.2/GLM-5, 4=GLM-5.2")
    p.add_argument("--causal", action="store_true",
                   help="dense run only causal (PART A)")
    p.add_argument("--no-causal", dest="non_causal", action="store_true",
                   help="dense run only non-causal (PART A)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    if args.all:
        args.dense = args.sparse_gqa = args.indexer = args.combined = args.quality = args.linear = True
    if not (args.dense or args.sparse_gqa or args.indexer or args.combined or args.quality or args.linear):
        args.dense = args.sparse_gqa = args.indexer = args.combined = args.quality = args.linear = True

    if args.causal and args.non_causal:
        raise SystemExit("--causal and --no-causal are mutually exclusive")
    if args.causal:
        causal_modes = [True]
    elif args.non_causal:
        causal_modes = [False]
    else:
        causal_modes = [False]  # most sparse kernels are non-causal only

    sparsities = args.sparsities or [args.sparsity]

    print("environment:")
    for k, v in env_report().items():
        print(f"  {k:>20} : {v}")
    print("  (see requirements.txt for full dependency notes)")

    if args.dense:
        bench_dense_gqa(args.seq_lens, causal_modes, args.warmup, args.iters)
    if args.sparse_gqa:
        bench_sparse_gqa(args.seq_lens, sparsities, args.warmup, args.iters)
    if args.indexer:
        bench_indexer(args.seq_lens, args.warmup, args.iters)
    if args.combined:
        bench_combined(
            args.seq_lens, args.sparsity, args.index_topk_freq,
            args.warmup, args.iters,
        )
    if args.quality:
        quality_sparsities = sparsities if args.sparsities else [0.02, 0.05, 0.10, 0.25]
        bench_quality(args.seq_lens, quality_sparsities)
    if args.linear:
        bench_linear_attention(args.seq_lens, args.warmup, args.iters)


if __name__ == "__main__":
    main()
