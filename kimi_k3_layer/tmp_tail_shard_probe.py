"""Probe: standalone "reduce+norm+GEMM" tail - which weight sharding
wins? X partial [B, 3584] -> AR+RMSNorm -> W [3584, 7168] -> full Y
on every rank.

  BASE plain AR -> SEPARATE RMSNorm kernel -> FULL-W GEMM (the
       unfused production tail: allreduce + _rmsnorm + full fc2)
  REF  fused AR+norm (flashinfer) -> FULL-W GEMM (51 MB weight load,
       redundant identical GEMM on all 8 ranks, no output comm)
  A    fused AR+norm -> col-shard GEMM [3584, 896] -> one-shot AG(Y)
  B    fused RS+norm -> row-shard GEMM [448, 7168] -> AR(Y)
       (split-K across ranks: partials round to bf16 before the AR)
  B2   RS push-only (NO norm wait) -> gamma-folded row-shard GEMM
       -> AR(Y) -> per-row 1/rms scale. Legal because rms is the
       same scalar for every rank's partial of a row, so it commutes
       past the GEMM and the AR. (Scale timed as an elementwise
       [B, 7168] pass with a precomputed row vector; the math is
       verified in the torch check below.)

  mpirun -n 8 --allow-run-as-root python3 kimi_k3_layer/tmp_tail_shard_probe.py
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3_layer.comm import (  # noqa: E402
    FlashInferAllReduce, OneShotComm)
from kimi_k3_layer.tmp_rs_norm_probe import bench_graph  # noqa: E402

LATENT = 3584
HIDDEN = 7168
RMS_EPS = 1e-6
MAX_B = 64


def main():
    rank = int(os.environ.get(
        "RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world = int(os.environ.get(
        "WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29516")
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank,
                            world_size=world,
                            device_id=torch.device("cuda", rank))
    from flashinfer.norm import rmsnorm

    chunk_in = LATENT // world     # 448
    chunk_out = HIDDEN // world    # 896

    fi = FlashInferAllReduce(rank, world, max_tokens=MAX_B,
                             max_hidden=HIDDEN)
    ag_a = {g: OneShotComm(rank, world, grid=g,
                           max_bytes=MAX_B * HIDDEN * 2)
            for g in (8, 32)}
    rs_nbytes = MAX_B * LATENT * 2 + 256 + world * MAX_B * 4
    rs_b = OneShotComm(rank, world, max_bytes=rs_nbytes)
    rs_b2 = OneShotComm(rank, world, max_bytes=rs_nbytes)

    torch.manual_seed(7)  # same W on every rank
    w = (torch.randn(HIDDEN, LATENT, device="cuda",
                     dtype=torch.bfloat16) * 0.02)
    gamma = torch.randn(LATENT, device="cuda", dtype=torch.bfloat16)
    w_col = w[rank * chunk_out:(rank + 1) * chunk_out].contiguous()
    w_row = w[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous()
    g_slice = gamma[rank * chunk_in:(rank + 1) * chunk_in].contiguous()
    # B2: fold gamma into the row-shard weight (per input column)
    w_row_g = (w_row.float() * g_slice.float()[None, :]).to(
        torch.bfloat16)

    if rank == 0:
        print(f"{'B':>4} {'BASE':>8} {'REF':>8} {'A:col+AG':>9} "
              f"{'B:row+AR':>9} "
              f"{'B2:row+scale':>13} {'errA':>9} {'errB':>9} "
              f"{'errB2':>9}  (us, graph replay, max over ranks)")
    for b in (1, 2, 4, 8, 16, 32, 64):
        torch.manual_seed(100 + rank)  # per-rank partials
        x = torch.randn(b, LATENT, device="cuda",
                        dtype=torch.bfloat16)
        zero_res = torch.zeros_like(x)

        # fp32 torch ground truth
        xs = x.float().clone()
        dist.all_reduce(xs)
        rms = torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        y_ref = ((xs * rms * gamma.float()) @ w.float().T)

        # --- BASE: plain AR -> separate norm -> full GEMM -----------
        def base_fn():
            r = fi(x)
            n = rmsnorm(r, gamma, RMS_EPS)
            return n @ w.T

        # --- REF: AR+norm fused -> full GEMM ------------------------
        def ref_fn():
            n = fi.norm_reduce(x, gamma, zero_res, RMS_EPS)
            return n @ w.T

        # --- A: AR+norm fused -> col GEMM -> AG ---------------------
        def a_fn(comm):
            n = fi.norm_reduce(x, gamma, zero_res, RMS_EPS)
            y_c = n @ w_col.T
            return comm.all_gather(y_c)

        # --- B: RS+norm fused -> row GEMM -> AR ---------------------
        def b_fn():
            n_s = rs_b.reduce_scatter_cols(
                x, norm_w=g_slice, eps=RMS_EPS)
            y_p = n_s @ w_row.T
            return fi(y_p)

        # --- B2: RS push-only -> gamma-folded GEMM -> AR -> scale ---
        rms_row = rms.to(torch.bfloat16)  # timing stand-in (see doc)

        def b2_fn():
            u = rs_b2.reduce_scatter_cols(
                x, norm_w=g_slice, eps=RMS_EPS, defer_scale=True)
            y_p = u @ w_row_g.T
            y = fi(y_p)
            return y * rms_row

        # correctness (vs fp32 ground truth; bf16-rounding class)
        def err(y):
            return (y.float() - y_ref).abs().max().item() / \
                y_ref.abs().max().item()

        e_ref = err(ref_fn())
        e_a = err(a_fn(ag_a[8]))
        e_b = err(b_fn())
        # B2 math check: scale applied AFTER AR with torch-side rms
        e_b2 = err(b2_fn())
        assert e_a < 3 * max(e_ref, 1e-3), f"A err {e_a} vs {e_ref}"
        assert e_b < 5 * max(e_ref, 1e-3), f"B err {e_b} vs {e_ref}"
        assert e_b2 < 5 * max(e_ref, 1e-3), f"B2 err {e_b2} vs {e_ref}"

        t_base = bench_graph(base_fn, world)
        t_ref = bench_graph(ref_fn, world)
        t_a = min(bench_graph(lambda c=c: a_fn(c), world)
                  for c in ag_a.values())
        t_b = bench_graph(b_fn, world)
        t_b2 = bench_graph(b2_fn, world)
        if rank == 0:
            print(f"{b:>4} {t_base:8.2f} {t_ref:8.2f} {t_a:9.2f} "
                  f"{t_b:9.2f} "
                  f"{t_b2:13.2f} {e_a:9.1e} {e_b:9.1e} {e_b2:9.1e}",
                  flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
