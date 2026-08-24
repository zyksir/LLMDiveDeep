#!/usr/bin/env python3
"""Can a large-batch all_gather fully overlap an INDEPENDENT GEMM?

Kimi-K3 framing: at prefill the fc1 column-AG runs while the gate +
shared-up GEMMs (independent of it) compute. Per backend and per B
(>= 1024), this measures, all CUDA-graph timed (median, MAX over
ranks):

  ag_alone     the AG by itself
  gemm_chain   N back-to-back GEMMs sized so the chain ~= ag_alone
               ([B,7168] x [7168,2432] bf16 — the gate|shared-up shape)
  overlapped   the SAME graph with the AG on a side stream and the
               GEMM chain on the main stream, joined at the end

  ideal = max(ag_alone, gemm_chain); overlap efficiency
  = (ag_alone + gemm_chain - overlapped) / min(ag_alone, gemm_chain)
  (1.0 = the cheaper one fully hidden; 0.0 = pure serialization).
  gemm dilation = overlapped - ag_alone when the AG is the longer leg
  (how much the AG's presence stretches the compute).

  mpirun --allow-run-as-root -np 8 python3 \\
      communication/bench_ag_gemm_overlap.py            # ~3 min
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from communication.collective import Collectives  # noqa: E402

BACKENDS = ("b10_copy_engine", "b10_copy_engine:sm",
            "torch_symm:multimem", "nccl")
GEMM_N = 2432  # gate (896) + per-rank shared gate/up (1536)


def graph_time_us(fn, calls=1, repeats=7):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / calls)
        dist.barrier()
    del g
    t = torch.tensor([statistics.median(ts)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bs", default="1024,2048,4096,8192")
    ap.add_argument("--dim", type=int, default=7168)
    ap.add_argument(
        "--layer", action="store_true",
        help="layer-realistic mode: AG = the fc1 col shard [B,448]; "
             "compute = ONE pass of the real independent window "
             "(gate [7168x896] + shared up [7168x1536] + shared down "
             "[768x7168]) instead of the synthetic equal-length chain")
    args = ap.parse_args()

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                              os.environ.get("RANK", "0")))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                               os.environ.get("WORLD_SIZE", "1")))
    # multi-node: the device index is the LOCAL rank
    torch.cuda.set_device(int(os.environ.get(
        "OMPI_COMM_WORLD_LOCAL_RANK",
        os.environ.get("LOCAL_RANK", rank))) % torch.cuda.device_count())
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29557")
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world,
        device_id=torch.device("cuda", torch.cuda.current_device()))

    sizes = [int(v) for v in args.bs.split(",")]
    comm = Collectives(
        dist.group.WORLD, max_numel=max(sizes) * args.dim * world,
        dtype=torch.bfloat16, max_hidden=10752,
        flashinfer_max_tokens=128, enable_b10=True)
    side = torch.cuda.Stream()
    fork = torch.cuda.Event()
    join = torch.cuda.Event()

    if rank == 0:
        print(f"{'B':>6} {'backend':<20} {'ag':>8} {'gemm xN':>10} "
              f"{'overlap':>8} {'ideal':>8} {'eff':>5} {'dilate':>7}",
              flush=True)
    for bs in sizes:
        g = torch.Generator(device="cuda").manual_seed(bs + rank)
        h = torch.randn(bs, args.dim, device="cuda",
                        dtype=torch.bfloat16, generator=g)
        if args.layer:
            x_ag = torch.randn(bs, 448, device="cuda",
                               dtype=torch.bfloat16, generator=g)
            w_gate = torch.randn(896, args.dim, device="cuda",
                                 dtype=torch.bfloat16, generator=g) * .02
            w_up = torch.randn(1536, args.dim, device="cuda",
                               dtype=torch.bfloat16, generator=g) * .02
            w_down = torch.randn(args.dim, 768, device="cuda",
                                 dtype=torch.bfloat16, generator=g) * .02

            def compute_once():
                h @ w_gate.T
                act = (h @ w_up.T)[:, :768]
                act @ w_down.T
        else:
            x_ag = torch.randn(bs, args.dim, device="cuda",
                               dtype=torch.bfloat16, generator=g)
            w = torch.randn(GEMM_N, args.dim, device="cuda",
                            dtype=torch.bfloat16, generator=g) * 0.02

        gemm1 = graph_time_us(
            compute_once if args.layer else (lambda: h @ w.T))
        for name in BACKENDS:
            ag = graph_time_us(
                lambda: comm.all_gather(x_ag, impl=name))
            n = 1 if args.layer else max(1, round(ag / gemm1))
            chain = gemm1 if args.layer else graph_time_us(
                lambda: h @ w.T, calls=n)

            def overlapped():
                cur = torch.cuda.current_stream()
                fork.record(cur)
                fork.wait(side)
                with torch.cuda.stream(side):
                    comm.all_gather(x_ag, impl=name)
                join.record(side)
                if args.layer:
                    compute_once()
                else:
                    for _ in range(n):
                        h @ w.T
                join.wait(cur)

            total = graph_time_us(overlapped)
            ideal = max(ag, chain)
            eff = (ag + chain - total) / max(1e-9, min(ag, chain))
            dilate = total - ag
            if rank == 0:
                print(f"{bs:>6} {name:<20} {ag:8.1f} "
                      f"{chain:7.1f}x{n:<2d} {total:8.1f} {ideal:8.1f} "
                      f"{eff:5.2f} {dilate:+7.1f}", flush=True)
        dist.barrier()
    if rank == 0:
        print("PASS bench_ag_gemm_overlap", flush=True)
    comm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
