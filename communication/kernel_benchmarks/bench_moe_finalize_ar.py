#!/usr/bin/env python3
"""MoE decode-tail communication duel: stock fused finalize+AR vs ours.

Three real tail strategies over IDENTICAL synthetic expert outputs
(uniform routing, the production-realistic case — see
communication/MOE_FINALIZE_AR.md):

  fused      ONE kernel: finalize (gather+weighted-sum over top-k) + one-shot
             lamport AR of concat([latent, hidden]) + rmsnorm(latent).
             torch.ops via tensorrt_llm MoEAllReduce (f17ea3ab32).
  packed     our previous tail: finalize (torch reference), cat copy,
             Collectives.all_reduce on [B, latent+hidden], split, rmsnorm.
  multimem   the B10 multimem tail comm: finalize (torch reference),
             Collectives.allreduce_norm(latent) + all_reduce(hidden).

Run: mpirun -n 8 --allow-run-as-root python3 \
        communication/kernel_benchmarks/bench_moe_finalize_ar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from communication.collective import Collectives
from communication.kernel_benchmarks.common import init, latency_us, skip

LATENT, HIDDEN, TOP_K = 3584, 7168, 16
SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


def torch_finalize(fc2_output, expanded_idx, scales, batch):
    """Reference finalize: gather permuted rows, weighted-sum over top-k."""
    gathered = fc2_output[expanded_idx.view(-1).long()].view(
        batch, TOP_K, LATENT)
    return (gathered * scales.view(batch, TOP_K, 1)).sum(dim=1)


def main() -> None:
    rank, world = init()
    if world not in (2, 4, 8, 16):
        skip("fused MoE finalize+AR needs TP 2/4/8/16", rank)
        return
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm._torch.distributed.ops import MoEAllReduce

    mapping = Mapping(world_size=world, tp_size=world, rank=rank,
                      gpus_per_node=world)
    moe_ar = MoEAllReduce(mapping)
    comm = Collectives(dist.group.WORLD,
                       max_numel=max(SIZES) * (LATENT + HIDDEN),
                       max_hidden=LATENT + HIDDEN)
    # MUST be rank-identical (model weight): the sharded-fc2 tails mix
    # rank-local normed latents across ranks — an unseeded norm_w made
    # every such tail "mismatch" by ~0.2 (2026-08-23 war story).
    gen_nw = torch.Generator(device="cuda").manual_seed(7)
    norm_w = torch.randn(LATENT, generator=gen_nw, device="cuda",
                         dtype=torch.bfloat16)
    eps = 1e-5
    from communication.kernels.finalize_ar_norm import (FinalizeARNorm,
                                                        fc2_shard_pdl)
    ours_engine = FinalizeARNorm(dist.group.WORLD, rank, world,
                                 max_tokens=max(SIZES), latent=LATENT)
    # tail-to-tail comparators: stock tail = fused + FULL fc2 GEMM;
    # ours tail = ours + PDL fc2-shard + one hidden AR.
    gen_w = torch.Generator(device="cuda").manual_seed(99)
    fc2_full_w = (torch.randn(LATENT, HIDDEN, generator=gen_w,
                              device="cuda") * 0.02).bfloat16()
    width = LATENT // world
    col_off = rank * width
    fc2_shard_w = fc2_full_w[col_off:col_off + width].contiguous()

    if rank == 0:
        print(f"{'B':>4} {'fused us':>9} {'ours us':>9} {'packed us':>10}"
              f" {'multimem us':>12} {'ours/fused':>10}")
    for batch in SIZES:
        if batch > moe_ar.max_tokens_for_message(LATENT + HIDDEN,
                                                 torch.bfloat16):
            skip(f"B={batch} beyond fused cap", rank)
            continue
        gen = torch.Generator(device="cuda").manual_seed(batch)
        n_perm = batch * TOP_K
        fc2 = torch.randn(n_perm, LATENT, generator=gen, device="cuda",
                          dtype=torch.float32).to(torch.bfloat16)
        idx = torch.randperm(n_perm, generator=gen, device="cuda").view(
            batch, TOP_K).to(torch.int32)  # [B, top_k] per the op contract
        scales = torch.softmax(
            torch.randn(batch, TOP_K, generator=gen, device="cuda",
                        dtype=torch.float32), dim=-1).to(torch.bfloat16)
        shared = torch.randn(batch, HIDDEN, generator=gen, device="cuda",
                             dtype=torch.float32).to(torch.bfloat16)

        def fused():
            return moe_ar.finalize_allreduce_rmsnorm_concat(
                fc2, shared, norm_w, idx, scales, eps)

        # In-layer, run_moe hands the kernel a per-expert TILE-PADDED
        # permuted buffer (~112 local experts x tile 8), so the top-k
        # gather scatters across ~896 rows even at B=8. Reproduce that
        # to isolate the gather penalty the in-layer 38.7us showed.
        n_padded = max(n_perm, 112 * 8)
        fc2_pad = torch.randn(n_padded, LATENT, generator=gen, device="cuda",
                              dtype=torch.float32).to(torch.bfloat16)
        idx_pad = (torch.randperm(n_padded, generator=gen, device="cuda")
                   [:n_perm].view(batch, TOP_K).to(torch.int32))

        def fused_padded():
            return moe_ar.finalize_allreduce_rmsnorm_concat(
                fc2_pad, shared, norm_w, idx_pad, scales, eps)

        def packed():
            routed = torch_finalize(fc2, idx, scales, batch)
            packed_t = comm.all_reduce(
                torch.cat((routed, shared), dim=-1))
            r, s = torch.split(packed_t, (LATENT, HIDDEN), dim=-1)
            return torch.nn.functional.rms_norm(
                r.float(), (LATENT,), norm_w.float(), eps).to(r.dtype), s

        scales_f32 = scales.float()

        def ours():
            return ours_engine(fc2, idx, scales_f32, norm_w, eps)

        def multimem():
            routed = torch_finalize(fc2, idx, scales, batch)
            r = comm.allreduce_norm(routed, norm_w, eps)
            s = comm.all_reduce(shared)
            return r, s

        def stock_tail():
            r, s = moe_ar.finalize_allreduce_rmsnorm_concat(
                fc2, shared, norm_w, idx, scales, eps)
            return s + r.view(-1, LATENT) @ fc2_full_w

        def ours_tail():
            r = ours_engine(fc2, idx, scales_f32, norm_w, eps)
            partial = fc2_shard_pdl(r, shared, fc2_shard_w, col_off)
            return comm.all_reduce(partial)

        def ours_tail_addmm():
            r = ours_engine(fc2, idx, scales_f32, norm_w, eps)
            partial = shared.addmm(r[:, col_off:col_off + width],
                                   fc2_shard_w)
            return comm.all_reduce(partial)

        # correctness: fused vs packed reference (both = finalize+AR+norm)
        rf, sf = fused()
        rp, sp = packed()
        err = (rf.float() - rp.float()).abs().max().item()
        base = rp.float().abs().max().item() + 1e-6
        ro = ours()
        rp_lat = torch.nn.functional.rms_norm(
            torch_finalize(fc2, idx, scales, batch).float(), (LATENT,),
            norm_w.float(), eps)
        err_ours = (ro.float() - rp_lat).abs().max().item() / (
            rp_lat.abs().max().item() + 1e-6)
        # tail correctness: ours_tail vs stock_tail (same full math)
        ts_ref = stock_tail()
        ts_ours = ours_tail()
        err_tail = ((ts_ours.float() - ts_ref.float()).abs().max().item()
                    / (ts_ref.float().abs().max().item() + 1e-6))
        us = {name: latency_us(fn) for name, fn in
              (("fused", fused), ("ours", ours),
               ("packed", packed), ("multimem", multimem),
               ("stock_tail", stock_tail), ("ours_tail", ours_tail),
               ("ours_tail_addmm", ours_tail_addmm))}
        if rank == 0:
            flag = "" if err / base < 0.05 else f"  MISMATCH rel={err/base:.3f}"
            oflag = "" if err_ours < 0.05 else f" OURS-MISMATCH rel={err_ours:.3f}"
            tflag = "" if err_tail < 0.05 else f" TAIL-MISMATCH rel={err_tail:.3f}"
            print(f"{batch:>4} {us['fused']:>9.1f} {us['ours']:>9.1f}"
                  f" {us['packed']:>10.1f} {us['multimem']:>12.1f}"
                  f" {us['ours']/us['fused']:>10.2f}x"
                  f" | tail stock={us['stock_tail']:>6.1f}"
                  f" ours={us['ours_tail']:>6.1f}"
                  f" ours_addmm={us['ours_tail_addmm']:>6.1f}"
                  f"{flag}{oflag}{tflag}", flush=True)


if __name__ == "__main__":
    main()
