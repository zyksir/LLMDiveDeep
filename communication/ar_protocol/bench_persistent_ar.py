#!/usr/bin/env python3
"""Persistent-AR protocol duel: 279 rounds, one kernel vs 279 launches.

Per-round cost at K3's decode-AR shape ([8, 3584] bf16, TP8), with
emulated per-rank compute skew injected before each round's push
(skew_ns * rank — rank 7 lags rank 0 by 7*skew).

mpirun -n 8 --allow-run-as-root python3 bench_persistent_ar.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from communication.ar_protocol.persistent_ar import PersistentAR

DIM, B, ROUNDS = 3584, 8, 279


def main() -> None:
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29553")
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank,
                            world_size=world,
                            device_id=torch.device("cuda", torch.cuda.current_device()))
    eng = PersistentAR(dist.group.WORLD, rank, world, max_tokens=B, dim=DIM)
    gen = torch.Generator(device="cuda").manual_seed(3 + rank)
    src = torch.randn(B, DIM, generator=gen, device="cuda").bfloat16()
    out = torch.empty_like(src)

    # correctness: one persistent 3-round pass vs nccl reference
    eng.persistent(src, out, rounds=3)
    torch.cuda.synchronize()
    ref = src.float().clone()
    dist.all_reduce(ref)
    rel = ((out.float() - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
    flag = "OK" if rel < 0.05 else f"MISMATCH {rel:.4f}"

    results = {}
    for skew_us in (0.0, 2.0, 5.0):
        skew_ns = int(skew_us * 1000)
        for arm in ("persistent", "chain"):
            def run():
                if arm == "persistent":
                    eng.persistent(src, out, rounds=ROUNDS, skew_ns=skew_ns)
                else:
                    for _ in range(ROUNDS):
                        eng.single(src, out, skew_ns=skew_ns)
            run()
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            dist.barrier()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(3):
                g.replay()
            t1.record()
            t1.synchronize()
            us = t0.elapsed_time(t1) * 1000 / (3 * ROUNDS)
            v = torch.tensor([us], device="cuda", dtype=torch.float64)
            dist.all_reduce(v, op=dist.ReduceOp.MAX)
            results[(arm, skew_us)] = v.item()
    if rank == 0:
        print(f"correctness: {flag}")
        print(f"{'skew/rank':>10} {'persistent':>11} {'chain':>8} "
              f"{'saving/AR':>10}  (max skew across ranks = 7x)")
        for skew_us in (0.0, 2.0, 5.0):
            p = results[('persistent', skew_us)]
            c = results[('chain', skew_us)]
            print(f"{skew_us:>8.1f}us {p:>10.2f}us {c:>7.2f}us "
                  f"{c - p:>9.2f}us", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
