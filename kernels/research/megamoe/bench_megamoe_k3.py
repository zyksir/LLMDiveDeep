#!/usr/bin/env python3
"""Task 1: run DeepGEMM's fp8_fp4_mega_moe at Kimi-K3 shapes (EP8, SiTU).

Kernel-only timing of the fused dispatch+FC1+SiTU+FC2+combine across
tokens/rank, to place the a2a-EP fused kernel against our replicated-EP
numbers. mpirun -n 8 --allow-run-as-root python3 bench_megamoe_k3.py
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

HIDDEN, INTER, EXPERTS, TOP_K = 3584, 3072, 896, 16
SIZES = (8, 64, 512, 2048, 8192)
MAX_TOKENS = max(SIZES)


def main() -> None:
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29544")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    import deep_gemm

    mma = os.environ.get("K3_MEGA_MMA", "fp8xfp4")
    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD, EXPERTS, MAX_TOKENS, TOP_K, HIDDEN, INTER,
        mma_type=mma)
    if rank == 0:
        print("symm buffer:", type(buf).__name__,
              [a for a in dir(buf) if a.startswith("buf")][:8], flush=True)

    e_local = EXPERTS // world
    gen = torch.Generator(device="cuda").manual_seed(7 + rank)
    if mma == "bf16xbf16":
        l1 = torch.randn(e_local, 2 * INTER, HIDDEN, generator=gen,
                         device="cuda").bfloat16() * 0.02
        l2 = torch.randn(e_local, HIDDEN, INTER, generator=gen,
                         device="cuda").bfloat16() * 0.02
        l1_t, l2_t = deep_gemm.transform_weights_for_mega_moe(l1, l2)
    else:
        # synthetic MXFP4 checkpoint: packed-fp4 payload + int32-packed UE8M0
        # scales (4 bytes/word, K/128 words; byte 124 = 2^-3 keeps sums sane)
        def q(e, n, k):
            w = torch.randint(0, 256, (e, n, k // 2), generator=gen,
                              dtype=torch.uint8, device="cuda").view(
                                  torch.int8)  # kPackedFP4 == torch::kInt8
            # MN-major per TMA check: stride(-2)==1, stride(-1)==mn
            sf = torch.full((e, k // 128, n), 0x7C7C7C7C, dtype=torch.int32,
                            device="cuda").transpose(-2, -1)
            return w, sf
        l1_t, l2_t = deep_gemm.transform_weights_for_mega_moe(
            q(e_local, 2 * INTER, HIDDEN), q(e_local, HIDDEN, INTER))
    if rank == 0:
        def desc(t):
            if isinstance(t, tuple):
                return [(x.shape, x.dtype) for x in t]
            return (t.shape, t.dtype)
        print("l1 transformed:", desc(l1_t), flush=True)

    gen_in = torch.Generator(device="cuda").manual_seed(1234)  # rank-identical
    for tokens in SIZES:
        x = torch.randn(tokens, HIDDEN, generator=gen_in, device="cuda",
                        dtype=torch.float32).to(torch.bfloat16)
        logits = torch.randn(tokens, EXPERTS, generator=gen_in, device="cuda")
        topk_w, topk_idx = torch.topk(torch.softmax(logits, -1), TOP_K, dim=-1)
        topk_idx = topk_idx.to(torch.int64)
        topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).float()
        y = torch.empty(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)

        # stage inputs through the symm-buffer views (mok harness pattern;
        # sglang fuses quant+staging into its own pre_dispatch kernel — our
        # fp8 staging copies a pre-quantized activation, so add ~a quant
        # kernel of cost when comparing against a real serving path)
        if mma != "bf16xbf16":
            xq = x.clamp(-448, 448).to(torch.float8_e4m3fn)
            x_sf = torch.full((tokens, HIDDEN // 128), 0x7F7F7F7F,
                              dtype=torch.int32, device="cuda")  # scale 1.0
        if mma == "bf16xbf16":
            def run():
                buf.x[:tokens].copy_(x)
                buf.topk_idx[:tokens].copy_(topk_idx)
                buf.topk_weights[:tokens].copy_(topk_w)
                deep_gemm.bf16_mega_moe(y, l1_t, l2_t, buf, fast_math=True)
        else:
            def run():
                buf.x[:tokens].copy_(xq)
                buf.x_sf[:tokens].copy_(x_sf)
                buf.topk_idx[:tokens].copy_(topk_idx)
                buf.topk_weights[:tokens].copy_(topk_w)
                deep_gemm.fp8_fp4_mega_moe(
                    y, l1_t, l2_t, buf, recipe=(1, 1, 32),
                    activation=os.environ.get("K3_MEGA_ACT", "swiglu"),
                    fast_math=True)

        for _ in range(10):
            run()
        torch.cuda.synchronize()
        assert torch.isfinite(y).all(), "non-finite output"
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            run()
        stop.record()
        stop.synchronize()
        us = start.elapsed_time(stop) * 1000 / 50
        val = torch.tensor([us], device="cuda", dtype=torch.float64)
        dist.all_reduce(val, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"tokens/rank={tokens:>5}  mega (staged+fused)    "
                  f"{val.item():9.1f} us", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
