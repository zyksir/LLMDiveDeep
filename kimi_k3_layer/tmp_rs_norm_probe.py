"""Probe: best-version comm menu at the pipeline shapes, TP8,
graph-captured. One column per collective, each the BEST variant:

  rs_cols   column reduce-scatter (bare)
  rs+norm   RS with the exact RMSNorm, best COMPLETE variant of
            {f: fused in-kernel wait, d: deferred pair (push partials,
            separate scale kernel)}
  fi_ar     flashinfer oneshot AR (reference)
  fi+norm   flashinfer fused AR+RMSNorm (reference)
  ag        column all-gather, best grid in {8,16,32,64}
  ag+q      AG + MXFP8 quantize, best of {receiver-fused (r) /
            sender-push fp8-wire (s)} x grids; label = time(grid,var)

  mpirun -n 8 --allow-run-as-root python3 kimi_k3_layer/tmp_rs_norm_probe.py
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3_layer.comm import (  # noqa: E402
    FlashInferAllReduce, OneShotComm, nccl_all_gather_cols)

RMS_EPS = 1e-6
GRIDS = (8, 16, 32, 64)


def bench_graph(fn, world, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    dist.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    g.replay()
    torch.cuda.synchronize()
    s.record()
    for _ in range(5):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    t = s.elapsed_time(e) * 1000 / (5 * iters)
    tt = torch.tensor([t], device="cuda")
    dist.all_reduce(tt, op=dist.ReduceOp.MAX)
    return tt.item()


def check_ag(comm, rank, chunk, max_b, push):
    """Mixed sizes big->small->big exercise the message-sized
    re-sentinel (clear_hist) on both wire formats."""
    for b in (max_b, 1, 8, 2, max_b):
        g0 = torch.Generator(device="cuda").manual_seed(b + rank)
        xs = torch.randn(b, chunk, generator=g0, device="cuda",
                         dtype=torch.float32).to(torch.bfloat16)
        ref = nccl_all_gather_cols(xs)
        if push:
            q, sfq = comm.all_gather_mxfp8_push(xs)
        else:
            got = comm.all_gather(xs)
            assert torch.equal(got, ref), f"ag mismatch b={b}"
            q, sfq = comm.all_gather_mxfp8(xs)
        deq = q.float() * torch.pow(
            2.0, sfq.float() - 127).repeat_interleave(32, dim=1)
        err = (deq - ref.float()).abs() / \
            ref.float().abs().clamp_min(0.25)
        assert err.max().item() < 0.13, f"agq mismatch b={b} push={push}"


def main():
    rank = int(os.environ.get(
        "RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world = int(os.environ.get(
        "WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank,
                            world_size=world,
                            device_id=torch.device("cuda", rank))
    from flashinfer.norm import rmsnorm

    max_b = 64
    for cols in (3584, 7168, 7168 + 3584):
        nbytes = max_b * cols * 2 + 256 + world * max_b * 4
        mk = lambda g=8, w="bf16": OneShotComm(  # noqa: E731
            rank, world, grid=g, wire=w, max_bytes=nbytes)
        rs1, rs2, rs3 = mk(), mk(), mk()
        agc = {g: mk(g) for g in GRIDS}
        agp = {g: mk(g, "fp8") for g in GRIDS}
        fi = FlashInferAllReduce(rank, world, max_tokens=max_b,
                                 max_hidden=cols)

        norm_w = torch.randn(cols, device="cuda", dtype=torch.bfloat16)
        chunk = cols // world
        wslice = norm_w[rank * chunk:(rank + 1) * chunk].contiguous()

        # correctness (mixed-size rotation) on smallest/largest grids
        for g in (GRIDS[0], GRIDS[-1]):
            check_ag(agc[g], rank, chunk, max_b, push=False)
            check_ag(agp[g], rank, chunk, max_b, push=True)
        # sender-push must be BIT-identical to the receiver-fused
        # variant (itself bit-exact vs torch.ops.trtllm.mxfp8_quantize)
        xs = torch.randn(16, chunk, device="cuda",
                         dtype=torch.float32).to(torch.bfloat16)
        q_r, sf_r = agc[8].all_gather_mxfp8(xs)
        q_s, sf_s = agp[8].all_gather_mxfp8_push(xs)
        assert torch.equal(q_r.view(torch.uint8), q_s.view(torch.uint8))
        assert torch.equal(sf_r, sf_s)

        if rank == 0:
            print(f"\n== hidden = {cols} ==")
            print(f"{'B':>4} {'rs_cols':>8} {'rs+norm':>11} "
                  f"{'fi_ar':>7} {'fi+norm':>8} "
                  f"{'ag':>11} {'ag+q':>12}  "
                  f"(us, graph replay, max over ranks; "
                  f"f/d=fused/deferred, (g)=grid, r/s=recv/send quant)")
        for b in (1, 4, 8, 16, 32, 64):
            g0 = torch.Generator(device="cuda").manual_seed(7 + rank)
            x = torch.randn(b, cols, generator=g0, device="cuda",
                            dtype=torch.float32).to(torch.bfloat16)
            x_sh = x[:, rank * chunk:(rank + 1) * chunk].contiguous()

            want_full = x.clone()
            dist.all_reduce(want_full)
            normed = rmsnorm(want_full, norm_w, RMS_EPS)
            want = normed[:, rank * chunk:(rank + 1) * chunk]
            got = rs2.reduce_scatter_cols(x, norm_w=wslice, eps=RMS_EPS)
            err = (got.float() - want.float()).abs().max().item()
            assert err < 0.15, f"rs+norm err {err}"

            t_rs = bench_graph(lambda: rs1.reduce_scatter_cols(x), world)

            def norm_deferred():
                o = rs3.reduce_scatter_cols(
                    x, norm_w=wslice, eps=RMS_EPS, defer_scale=True)
                rs3.scale_deferred(o, wslice, eps=RMS_EPS)
            t_rsn, rsn_v = min(
                (bench_graph(
                    lambda: rs2.reduce_scatter_cols(
                        x, norm_w=wslice, eps=RMS_EPS), world), "f"),
                (bench_graph(norm_deferred, world), "d"))
            zero_res = torch.zeros_like(x)
            t_fiar = bench_graph(lambda: fi(x), world)
            t_fi = bench_graph(
                lambda: fi.norm_reduce(x, norm_w, zero_res, RMS_EPS),
                world)

            ag_best, ag_g = min(
                (bench_graph(lambda g=g: agc[g].all_gather(x_sh), world),
                 g) for g in GRIDS)
            aq = []
            for g in GRIDS:
                aq.append((bench_graph(
                    lambda g=g: agc[g].all_gather_mxfp8(x_sh), world),
                    g, "r"))
                aq.append((bench_graph(
                    lambda g=g: agp[g].all_gather_mxfp8_push(x_sh),
                    world), g, "s"))
            aq_best, aq_g, aq_v = min(aq)

            if rank == 0:
                print(f"{b:>4} {t_rs:8.2f} {t_rsn:8.2f}({rsn_v}) "
                      f"{t_fiar:7.2f} {t_fi:8.2f} "
                      f"{ag_best:7.2f}({ag_g:>2}) "
                      f"{aq_best:7.2f}({aq_g:>2}{aq_v})",
                      flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
