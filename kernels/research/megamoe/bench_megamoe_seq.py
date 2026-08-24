#!/usr/bin/env python3
"""Task-1 comparator 2: the SAME a2a-EP algorithm as Mega MoE, UNFUSED.

Pipeline per iteration (all separate ops, bf16, K3 shapes):
  dispatch  torch.distributed all_to_all_single of duplicated token rows
            (one row per (token, pick) pair, grouped by owner rank)
  compute   sort-by-expert -> m-aligned contiguous layout ->
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous FC1 -> swiglu ->
            grouped FC2
  combine   a2a back + weighted per-token reduction

vs `bf16_mega_moe` (one fused kernel) at the same tokens/rank. The gap
isolates what fusion+in-kernel overlap buys over the sequential
pipeline. Caveat recorded: the permutation/index ops here are plain
torch (a tuned seq pipeline would use dedicated permute kernels), so
this arm is an upper bound on sequential cost — but dispatch/combine
bytes and GEMM work are identical.

mpirun -n 8 --allow-run-as-root python3 bench_megamoe_seq.py
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

HIDDEN, INTER, EXPERTS, TOP_K = 3584, 3072, 896, 16
SIZES = (8, 64, 512, 2048)


def main() -> None:
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29545")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    import deep_gemm

    e_local = EXPERTS // world
    gen = torch.Generator(device="cuda").manual_seed(7 + rank)
    l1 = (torch.randn(e_local, 2 * INTER, HIDDEN, generator=gen,
                      device="cuda") * 0.02).bfloat16()
    l2 = (torch.randn(e_local, HIDDEN, INTER, generator=gen,
                      device="cuda") * 0.02).bfloat16()
    m_align = deep_gemm.get_m_alignment_for_contiguous_layout()

    gen_in = torch.Generator(device="cuda").manual_seed(1234)
    for tokens in SIZES:
        x = torch.randn(tokens, HIDDEN, generator=gen_in,
                        device="cuda").bfloat16()
        logits = torch.randn(tokens, EXPERTS, generator=gen_in,
                             device="cuda")
        topk_w, topk_idx = torch.topk(torch.softmax(logits, -1), TOP_K, -1)
        topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).float()

        # ---- static routing plan (fixed per size; splits constant)
        flat_e = topk_idx.reshape(-1)                       # [T*K] global
        dest = (flat_e // e_local).to(torch.int64)
        order = torch.argsort(dest, stable=True)
        send_tok = torch.arange(tokens, device="cuda").repeat_interleave(
            TOP_K)[order]
        send_loc = (flat_e % e_local)[order].to(torch.int32)
        send_counts = torch.bincount(dest, minlength=world)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        sc = send_counts.tolist()
        rc = recv_counts.tolist()
        n_recv = sum(rc)

        # receive-side expert ids (constant per plan): exchange once
        e_recv = torch.empty(n_recv, dtype=torch.int32, device="cuda")
        dist.all_to_all_single(e_recv, send_loc.contiguous(),
                               output_split_sizes=rc, input_split_sizes=sc)
        ord2 = torch.argsort(e_recv, stable=True)
        inv2 = torch.empty_like(ord2)
        inv2[ord2] = torch.arange(n_recv, device="cuda")
        cnt = torch.bincount(e_recv.long(), minlength=e_local)
        pad_cnt = ((cnt + m_align - 1) // m_align) * m_align
        offs = torch.zeros(e_local + 1, dtype=torch.int64, device="cuda")
        offs[1:] = torch.cumsum(pad_cnt, 0)
        m_total = int(offs[-1].item())
        # padded destination row for each sorted-received row
        grp_start = offs[:-1].repeat_interleave(cnt)
        within = (torch.arange(n_recv, device="cuda")
                  - torch.cumsum(cnt, 0).repeat_interleave(cnt)
                  + cnt.repeat_interleave(cnt))
        dst_row = (grp_start + within).long()
        # m_indices: group id per padded row (pads get their group id too;
        # zero inputs produce zero outputs, harmless)
        m_indices = torch.repeat_interleave(
            torch.arange(e_local, dtype=torch.int32, device="cuda"),
            pad_cnt)

        x_recv = torch.empty(n_recv, HIDDEN, dtype=torch.bfloat16,
                             device="cuda")
        x_pad = torch.zeros(m_total, HIDDEN, dtype=torch.bfloat16,
                            device="cuda")
        h1 = torch.empty(m_total, 2 * INTER, dtype=torch.bfloat16,
                         device="cuda")
        y_pad = torch.empty(m_total, HIDDEN, dtype=torch.bfloat16,
                            device="cuda")
        y_back = torch.empty(tokens * TOP_K, HIDDEN, dtype=torch.bfloat16,
                             device="cuda")
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(tokens * TOP_K, device="cuda")

        def run():
            x_send = x[send_tok]
            dist.all_to_all_single(x_recv, x_send,
                                   output_split_sizes=rc,
                                   input_split_sizes=sc)
            x_pad[dst_row] = x_recv[ord2]
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                x_pad, l1, h1, m_indices)
            g, u = h1[:, :INTER], h1[:, INTER:]
            act = (torch.nn.functional.silu(g.float())
                   * u.float()).bfloat16()
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                act, l2, y_pad, m_indices)
            y_sorted = y_pad[dst_row]           # unpad
            dist.all_to_all_single(
                y_back, y_sorted[inv2],
                output_split_sizes=sc, input_split_sizes=rc)
            y = y_back[inv_order].view(tokens, TOP_K, HIDDEN)
            return (y.float() * topk_w.view(tokens, TOP_K, 1)).sum(1)

        out = run()
        assert torch.isfinite(out).all().item()
        for _ in range(8):
            run()
        torch.cuda.synchronize()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            run()
        stop.record()
        stop.synchronize()
        us = start.elapsed_time(stop) * 1000 / 20
        val = torch.tensor([us], device="cuda", dtype=torch.float64)
        dist.all_reduce(val, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"tokens/rank={tokens:>5}  seq(unfused a2a) "
                  f"{val.item():9.1f} us", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
