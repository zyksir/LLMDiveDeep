"""Probe: REF-style MoE tail with the shared-expert AR split out and
OVERLAPPED, vs the current fc2-shard tail and the baseline packed AR.

Insight: after AR+norm the full normed latent is IDENTICAL on every
rank, so the full-weight fc2 output needs NO collective; only the
shared-expert partial must be reduced, and that AR(7168) is
independent of the whole latent chain -> side stream, overlapped
under AR+norm(3584) + full fc2 GEMM.

  cur   RS+norm(3584) -> fc2-shard GEMM (addmm on shared partial)
        -> one AR(7168) carrying fc2+shared partials  (today's tail)
  arsh  AR+norm(3584) -> rank slice -> fc2-shard GEMM (addmm on
        shared partial) -> one AR(7168). The FAIR shard arm for the
        REF_TAIL_MIN_TOKENS crossover: same FlashInfer AR+norm
        primitive as `new`, only the tail structure differs.
  pack  packed [latent|shared] AR(10752) -> norm -> full fc2 + add
        (the baseline / -fc2shard tail)
  new   side stream: AR(7168) of shared partial
        main stream: AR+norm(3584) -> full fc2 GEMM
        join: out = y_fc2 + y_shared

  mpirun -n 8 --allow-run-as-root python3 kimi_k3_layer/tmp_tail_ref_overlap.py
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
MAX_B = 80


def main():
    rank = int(os.environ.get(
        "RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world = int(os.environ.get(
        "WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank,
                            world_size=world,
                            device_id=torch.device("cuda", rank))
    from flashinfer.norm import rmsnorm

    chunk_in = LATENT // world  # 448

    fi_lat = FlashInferAllReduce(rank, world, max_tokens=MAX_B,
                                 max_hidden=LATENT)
    fi_sh = FlashInferAllReduce(rank, world, max_tokens=MAX_B,
                                max_hidden=HIDDEN)
    fi_pack = FlashInferAllReduce(rank, world, max_tokens=MAX_B,
                                  max_hidden=LATENT + HIDDEN)
    rs_nbytes = MAX_B * LATENT * 2 + 256 + world * MAX_B * 4
    rs = OneShotComm(rank, world, max_bytes=rs_nbytes)

    torch.manual_seed(7)  # same weights on every rank
    w = (torch.randn(HIDDEN, LATENT, device="cuda",
                     dtype=torch.bfloat16) * 0.02)
    gamma = torch.randn(LATENT, device="cuda", dtype=torch.bfloat16)
    w_row = w[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous()
    g_slice = gamma[rank * chunk_in:(rank + 1) * chunk_in].contiguous()

    side = torch.cuda.Stream()
    ev_fork = torch.cuda.Event()
    ev_join = torch.cuda.Event()

    if rank == 0:
        print(f"{'B':>4} {'cur:shard+1AR':>14} {'arsh:AR+shard':>14} "
              f"{'pack:fatAR':>11} "
              f"{'new:REF+ovl':>12} {'errC':>9} {'errA':>9} "
              f"{'errP':>9} "
              f"{'errN':>9}  (us, graph replay, max over ranks)")
    for b in (1, 2, 4, 8, 16, 32, 64, 80):
        torch.manual_seed(100 + rank)
        x_lat = torch.randn(b, LATENT, device="cuda",
                            dtype=torch.bfloat16)   # routed partial
        x_sh = (torch.randn(b, HIDDEN, device="cuda",
                            dtype=torch.bfloat16) * 0.1)  # shared part.
        zero_lat = torch.zeros_like(x_lat)

        # fp32 ground truth
        xs = x_lat.float().clone()
        dist.all_reduce(xs)
        sh = x_sh.float().clone()
        dist.all_reduce(sh)
        rms = torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        y_ref = (xs * rms * gamma.float()) @ w.float().T + sh

        # --- cur: RS+norm -> shard GEMM (+shared addmm) -> one AR ---
        def cur_fn():
            n_s = rs.reduce_scatter_cols(
                x_lat, norm_w=g_slice, eps=RMS_EPS)
            out = x_sh.clone()
            out.addmm_(n_s, w_row.T)
            return fi_sh(out)

        # --- arsh: AR+norm -> rank slice -> shard GEMM -> one AR ----
        # (the FAIR shard arm: identical AR+norm primitive as `new`)
        def arsh_fn():
            n = fi_lat.norm_reduce(x_lat, gamma, zero_lat, RMS_EPS)
            n_s = n[:, rank * chunk_in:(rank + 1) * chunk_in]
            out = x_sh.clone()
            out.addmm_(n_s, w_row.T)
            return fi_sh(out)

        # --- pack: [latent|shared] fat AR -> norm -> full fc2 + add -
        def pack_fn():
            packed = torch.cat((x_lat, x_sh), dim=-1)
            red = fi_pack(packed)
            n = rmsnorm(red[:, :LATENT].contiguous(), gamma, RMS_EPS)
            return torch.addmm(red[:, LATENT:], n, w.T)

        # --- new: shared AR on side stream || AR+norm -> full fc2 ---
        def new_fn():
            cur_s = torch.cuda.current_stream()
            ev_fork.record(cur_s)
            with torch.cuda.stream(side):
                ev_fork.wait(side)
                y_sh = fi_sh(x_sh)
                ev_join.record(side)
            n = fi_lat.norm_reduce(x_lat, gamma, zero_lat, RMS_EPS)
            y = n @ w.T
            ev_join.wait(cur_s)
            return y + y_sh

        def err(y):
            return (y.float() - y_ref).abs().max().item() / \
                y_ref.abs().max().item()

        e_c, e_a, e_p, e_n = (err(cur_fn()), err(arsh_fn()),
                              err(pack_fn()), err(new_fn()))
        assert max(e_c, e_a, e_p, e_n) < 5e-2, (e_c, e_a, e_p, e_n)

        t_c = bench_graph(cur_fn, world)
        t_a = bench_graph(arsh_fn, world)
        t_p = bench_graph(pack_fn, world)
        t_n = bench_graph(new_fn, world)
        if rank == 0:
            print(f"{b:>4} {t_c:14.2f} {t_a:14.2f} {t_p:11.2f} "
                  f"{t_n:12.2f} "
                  f"{e_c:9.1e} {e_a:9.1e} {e_p:9.1e} {e_n:9.1e}",
                  flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
