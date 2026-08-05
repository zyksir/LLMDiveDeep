#!/usr/bin/env python3
"""Benchmark one-shot CUDA collectives vs NCCL / flashinfer / TRT-LLM.

Launch (trt-dev container; mpirun enables the trtllm AR rows):

  docker exec trt-dev bash -c "cd /workspace/diffusion_inference/LLMDiveDeep \
      && mpirun --allow-run-as-root -np 8 python3 kimi_k3_layer/bench_comm.py"

  # large messages (separate long-running command)
  ... bench_comm.py --tokens 1024 4096 16384

Message size N = tokens x 7168 bf16 (the K3 hidden width). For every op
the FULL message is N: AG gathers N/world shards into N, RS reduces N
into N/world, AR reduces N into N. Every impl is timed end-to-end from
a regular (non-symm) input tensor.

Impls
  oneshot        ours (comm_cuda Lamport kernels; best of a grid x
                 block sweep = what TunedAllGather dispatches)
  oneshot_cols   ours, column-RS variant (optionally norm-fused)
  ce / ce_cols   ours, copy-engine collectives (cudaMemcpy2DAsync peer
                 copies, ZERO SMs on the data path; best of a stream-
                 fanout sweep). RS pays one extra local slab-sum kernel.
  nccl           torch.distributed on regular tensors
  nccl_symmbuf   torch.distributed on symm-mem-registered tensors
  symm_1shot     torch.ops.symm_mem.one_shot_all_reduce
  multimem       torch.ops.symm_mem.multimem_* (NVLS; needs multicast)
  flashinfer     trtllm Lamport one-shot AR (production reference)
  trtllm         tensorrt_llm AllReduce module (AUTO strategy; needs
                 mpirun + tensorrt_llm importable, else skipped)

Roofline: a one-shot AG/RS moves (w-1)/w * N bytes over this rank's
NVLink; a one-shot AR moves (w-1) * N. sol_us = wire bytes / NVLINK_GBS;
eff = sol/measured. Small messages are LATENCY bound (NVLink hop
~1.5-2 us), so eff far below 1 at tiny sizes is expected; the honest
target there is the flashinfer AR latency, which our AG/RS must beat
(they move 1/world of its wire bytes).
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.kernel_bench import _try_graph_capture  # noqa: E402
from kimi_k3_layer.comm import CeComm, OneShotComm  # noqa: E402

HIDDEN = 7168
NVLINK_GBS = 900.0  # B200 NVLink5, per direction


def dist_max(value: float) -> float:
    t = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def bench_fn(fn, warmup=5, iters=30, repeats=5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    graph = _try_graph_capture(fn, iters)
    ok = torch.tensor([graph is not None], dtype=torch.int32)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not ok.item():
        graph = None
    samples = []
    for _ in range(repeats):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph is not None:
            graph.replay()
        else:
            for _ in range(iters):
                fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iters)
    del graph
    return dist_max(statistics.median(samples))


def rank_input(r: int, numel: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(7000 + r)
    return torch.randn(numel, generator=g, device="cuda",
                       dtype=torch.float32).to(torch.bfloat16)


class Ctx:
    def __init__(self, rank, world, max_elems, fi_max_tokens):
        self.rank, self.world = rank, world
        # construct BEFORE flashinfer touches the process (importing
        # flashinfer first breaks the lazy `tensorrt` import chain)
        self.trtllm_ar = self._make_trtllm_ar()
        # our instances are sized per message (the Lamport clear covers
        # the full slot capacity, so oversizing is pure overhead)
        self._comms: dict[tuple, OneShotComm] = {}
        self._ces: dict[int, CeComm] = {}
        self.ce_bytes = max_elems * 2
        self.group_name = dist.group.WORLD.group_name
        dev = torch.device("cuda", rank)
        self.symm_in = symm_mem.empty(max_elems, dtype=torch.bfloat16,
                                      device=dev)
        self.symm_out = symm_mem.empty(max_elems, dtype=torch.bfloat16,
                                       device=dev)
        symm_mem.rendezvous(self.symm_in, self.group_name)
        symm_mem.rendezvous(self.symm_out, self.group_name)

        self.fi_max = fi_max_tokens * HIDDEN
        from flashinfer.comm import (
            trtllm_create_ipc_workspace_for_all_reduce_fusion,
        )
        self._fi_ipc, self.fi_ws = (
            trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=rank, tp_size=world,
                max_token_num=fi_max_tokens, hidden_dim=HIDDEN,
                use_fp32_lamport=False,
            )
        )

    def comm(self, nbytes: int, grid: int) -> OneShotComm:
        key = (nbytes, grid)
        if key not in self._comms:
            self._comms[key] = OneShotComm(
                self.rank, self.world, max_bytes=nbytes, grid=grid)
        return self._comms[key]

    def ce(self, sync_threads: int) -> CeComm:
        # CE has no clear cost, so one max-size instance serves all
        # message sizes (unlike the per-size Lamport instances).
        # streams stays 0: P2P copies serialize on one engine anyway
        # (measured); the only tunable is the sync-kernel layout.
        if sync_threads not in self._ces:
            self._ces[sync_threads] = CeComm(
                self.rank, self.world, max_bytes=self.ce_bytes,
                streams=0, sync_threads=sync_threads)
        return self._ces[sync_threads]

    def _make_trtllm_ar(self):
        try:
            import tensorrt_llm  # noqa: F401 - full init first
            from tensorrt_llm._torch.distributed import AllReduce
            from tensorrt_llm.mapping import Mapping

            mapping = Mapping(world_size=self.world, tp_size=self.world,
                              rank=self.rank,
                              gpus_per_node=self.world)
            return AllReduce(mapping=mapping)
        except Exception as e:  # noqa: BLE001 - optional reference
            if self.rank == 0:
                import traceback
                traceback.print_exc()
                print(f"trtllm AR unavailable: "
                      f"{str(e).splitlines()[0][:70]}")
            return None

    def flashinfer_ar(self, x, out):
        from flashinfer.comm import (
            AllReduceFusionPattern,
            trtllm_allreduce_fusion,
        )
        trtllm_allreduce_fusion(
            allreduce_in=x, world_size=self.world, world_rank=self.rank,
            token_num=x.numel() // HIDDEN, hidden_dim=HIDDEN,
            workspace_ptrs=self.fi_ws, launch_with_pdl=True,
            trigger_completion_at_end=True, fp32_acc=False,
            pattern_code=AllReduceFusionPattern.kAllReduce,
            use_oneshot=True, allreduce_out=out,
            residual_in=None, residual_out=None, norm_out=None,
            quant_out=None, scale_out=None, rms_gamma=None, rms_eps=None,
            scale_factor=None, layout_code=None,
        )

    def destroy(self):
        from flashinfer.comm import (
            trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
        )
        try:
            trtllm_destroy_ipc_workspace_for_all_reduce_fusion(self._fi_ipc)
        except Exception:  # noqa: BLE001
            pass


def sweep_ours(candidates, check, label):
    """Best config by measured time from [(cfg_name, callable)]."""
    best_t, best_cfg = float("inf"), None
    for cfg, run in candidates:
        err = check(run())
        assert err < 0.1, f"{label} {cfg} err={err}"
        t = bench_fn(run)
        if t < best_t:
            best_t, best_cfg = t, cfg
    return best_t, best_cfg


def ours_candidates(ctx, nbytes, method, *args, **kwargs):
    out = []
    for grid in (1, 2, 4, 8, 16, 32):
        if grid > 8 and nbytes < (1 << 20):
            continue  # tiny messages never win with big grids
        if grid < 4 and nbytes > (1 << 18):
            continue
        comm = ctx.comm(nbytes, grid)
        for blk in (128, 256, 512):
            out.append((
                f"g{grid}x{blk}",
                lambda comm=comm, blk=blk: getattr(comm, method)(
                    *args, block=blk, **kwargs),
            ))
    return out


def impl_rows(ctx, tokens, args):
    rank, world = ctx.rank, ctx.world
    numel = tokens * HIDDEN
    nbytes = numel * 2
    x = rank_input(rank, numel)

    ref_sum = torch.zeros(numel, device="cuda", dtype=torch.float32)
    for r in range(world):
        ref_sum += rank_input(r, numel).float()
    shard = numel // world
    x_sh = x[:shard].clone()  # my AG shard (dim0-concat convention)

    def ag_err(got):
        got = got.reshape(-1)
        err = 0.0
        for r in range(world):
            want = rank_input(r, numel)[:shard]
            err = max(err, (got[r * shard:(r + 1) * shard].float()
                            - want.float()).abs().max().item())
        return err

    def rs_err(got):
        want = ref_sum[rank * shard:(rank + 1) * shard]
        return (got.reshape(-1).float() - want).abs().max().item()

    def rs_cols_err(got):
        # column chunks of a [tokens, HIDDEN] view
        chunk = HIDDEN // world
        want = ref_sum.view(tokens, HIDDEN)[
            :, rank * chunk:(rank + 1) * chunk]
        return (got.float() - want).abs().max().item()

    def ar_err(got):
        return (got.reshape(-1).float() - ref_sum).abs().max().item()

    rows = []

    def add(op, impl, t, cfg=""):
        wire = nbytes * (world - 1) * (1 if op == "all_reduce" else 1 / world)
        sol = wire / (NVLINK_GBS * 1e3)  # us
        rows.append({
            "op": op, "impl": impl, "tokens": tokens, "mbytes":
            round(nbytes / 1e6, 3), "latency_us": round(t, 2),
            "sol_us": round(sol, 2), "eff": round(sol / t, 3), "cfg": cfg,
        })
        if rank == 0:
            print(f"  {op:<14} {impl:<13} {t:9.2f} us   "
                  f"eff {sol / t:6.1%}  {cfg}", flush=True)

    # ---------------- all_gather ----------------
    t, cfg = sweep_ours(
        ours_candidates(ctx, nbytes, "all_gather", x_sh.view(1, -1)),
        ag_err, "ag_oneshot")
    add("all_gather", "oneshot", t, cfg)

    # copy-engine AG (memcpy data path, zero SMs; sweep sync layout)
    t, cfg = sweep_ours(
        [(f"sync{s}t",
          lambda ce=ctx.ce(s): ce.all_gather(x_sh.view(1, -1)))
         for s in (1, 32)],
        ag_err, "ag_ce")
    add("all_gather", "ce", t, cfg)

    out = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
    dist.all_gather_into_tensor(out, x_sh)
    assert ag_err(out) < 0.1
    add("all_gather", "nccl",
        bench_fn(lambda: dist.all_gather_into_tensor(out, x_sh)))

    sym_sh = ctx.symm_in[:shard]
    sym_out = ctx.symm_out[:numel]

    def nccl_symm_ag():
        sym_sh.copy_(x_sh)
        dist.all_gather_into_tensor(sym_out, sym_sh)
    nccl_symm_ag()
    assert ag_err(sym_out) < 0.1
    add("all_gather", "nccl_symmbuf", bench_fn(nccl_symm_ag))

    try:
        def mm_ag():
            sym_sh.copy_(x_sh)
            torch.ops.symm_mem.multimem_all_gather_out(
                sym_sh, ctx.group_name, sym_out)
        mm_ag()
        assert ag_err(sym_out) < 0.1
        add("all_gather", "multimem", bench_fn(mm_ag))
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"  all_gather     multimem      skipped: "
                  f"{str(e).splitlines()[0][:60]}")

    # ---------------- reduce_scatter ----------------
    t, cfg = sweep_ours(
        ours_candidates(ctx, nbytes, "reduce_scatter", x),
        rs_err, "rs_oneshot")
    add("reduce_scatter", "oneshot", t, cfg)

    # column-RS at the [tokens, HIDDEN] layout (dedicated instance)
    if HIDDEN % (world * 8) == 0:
        rsc = ctx.comm(nbytes + 256 + world * tokens * 4, 3)
        x2d = x.view(tokens, HIDDEN)
        got = rsc.reduce_scatter_cols(x2d)
        assert rs_cols_err(got) < 0.1
        add("reduce_scatter", "oneshot_cols",
            bench_fn(lambda: rsc.reduce_scatter_cols(x2d)))

    # copy-engine column-RS (memcpy pushes + one local slab-sum kernel)
    if HIDDEN % world == 0:
        x2d = x.view(tokens, HIDDEN)
        t, cfg = sweep_ours(
            [(f"sync{s}t",
              lambda ce=ctx.ce(s): ce.reduce_scatter_cols(x2d))
             for s in (1, 32)],
            rs_cols_err, "rs_ce")
        add("reduce_scatter", "ce_cols", t, cfg)

    # NCCL/multimem reduce in bf16 on the wire; our kernels accumulate
    # fp32, hence the looser bound for the former.
    BF16_TOL = 1.5

    rs_out = torch.empty(shard, dtype=torch.bfloat16, device="cuda")
    dist.reduce_scatter_tensor(rs_out, x)
    assert rs_err(rs_out) < BF16_TOL
    add("reduce_scatter", "nccl",
        bench_fn(lambda: dist.reduce_scatter_tensor(rs_out, x)))

    sym_in = ctx.symm_in[:numel]
    sym_rs = ctx.symm_out[:shard]

    def nccl_symm_rs():
        sym_in.copy_(x)
        dist.reduce_scatter_tensor(sym_rs, sym_in)
    nccl_symm_rs()
    assert rs_err(sym_rs) < BF16_TOL
    add("reduce_scatter", "nccl_symmbuf", bench_fn(nccl_symm_rs))

    # ---------------- all_reduce ----------------
    # (no home-grown AR row: it measured 6-8x behind flashinfer's
    # oneshot fusion kernel and was deleted; flashinfer IS our AR)
    if numel <= ctx.fi_max:
        fi_out = torch.empty_like(x)
        ctx.flashinfer_ar(x, fi_out)
        assert ar_err(fi_out) < BF16_TOL
        add("all_reduce", "flashinfer",
            bench_fn(lambda: ctx.flashinfer_ar(x, fi_out)))

    if ctx.trtllm_ar is not None:
        x2d = x.view(tokens, HIDDEN)
        got = ctx.trtllm_ar(x2d)
        assert ar_err(got) < BF16_TOL
        add("all_reduce", "trtllm",
            bench_fn(lambda: ctx.trtllm_ar(x2d)))

    ar_buf = x.clone()

    def nccl_ar():
        ar_buf.copy_(x)
        dist.all_reduce(ar_buf)
    nccl_ar()
    assert ar_err(ar_buf) < BF16_TOL
    add("all_reduce", "nccl", bench_fn(nccl_ar))

    try:
        def symm_1shot():
            sym_in.copy_(x)
            return torch.ops.symm_mem.one_shot_all_reduce(
                sym_in, "sum", ctx.group_name)
        got = symm_1shot()
        assert ar_err(got) < BF16_TOL
        add("all_reduce", "symm_1shot", bench_fn(symm_1shot))
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"  all_reduce     symm_1shot    skipped: "
                  f"{str(e).splitlines()[0][:60]}")

    try:
        def mm_ar():
            sym_in.copy_(x)
            torch.ops.symm_mem.multimem_all_reduce_(
                sym_in, "sum", ctx.group_name)
        mm_ar()
        assert ar_err(sym_in) < BF16_TOL
        add("all_reduce", "multimem", bench_fn(mm_ar))
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"  all_reduce     multimem      skipped: "
                  f"{str(e).splitlines()[0][:60]}")

    del ref_sum
    return rows


def summary(rows):
    """Concise pivot: one markdown table per op, tokens x impl (us)."""
    ops = ["all_gather", "reduce_scatter", "all_reduce"]
    lines = []
    for op in ops:
        sub = [r for r in rows if r["op"] == op]
        if not sub:
            continue
        impls = sorted({r["impl"] for r in sub},
                       key=lambda i: (not i.startswith("oneshot"), i))
        toks = sorted({r["tokens"] for r in sub})
        cell = {(r["tokens"], r["impl"]): r["latency_us"] for r in sub}
        sol = {r["tokens"]: r["sol_us"] for r in sub}
        lines.append(f"\n### {op} (us, lower is better)")
        lines.append("| tokens | " + " | ".join(impls) + " | SOL |")
        lines.append("|" + "---|" * (len(impls) + 2))
        for t in toks:
            best = min(v for (tt, _), v in cell.items() if tt == t)
            def fmt(i, t=t, best=best):
                v = cell.get((t, i))
                if v is None:
                    return "-"
                return f"**{v:.2f}**" if v == best else f"{v:.2f}"
            lines.append(f"| {t} | " + " | ".join(fmt(i) for i in impls)
                         + f" | {sol[t]:.2f} |")
    text = "\n".join(lines)
    print(text)
    return text


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = ["all_gather", "reduce_scatter", "all_reduce"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False)
    for ax, op in zip(axes, ops):
        sub = [r for r in rows if r["op"] == op]
        for impl in sorted({r["impl"] for r in sub}):
            pts = sorted((r["mbytes"], r["latency_us"])
                         for r in sub if r["impl"] == impl)
            ax.plot(*zip(*pts), marker="o", label=impl)
        sol = sorted({(r["mbytes"], r["sol_us"]) for r in sub})
        ax.plot(*zip(*sol), "k--", alpha=0.5, label="SOL (NVLink 900GB/s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(op)
        ax.set_xlabel("message MB")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("latency (us)")
    fig.suptitle("One-shot CUDA collectives vs NCCL (B200)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int,
                        default=[1, 4, 16, 64, 256])
    parser.add_argument("--fi-max-tokens", type=int, default=None)
    args = parser.parse_args()
    if args.fi_max_tokens is None:
        args.fi_max_tokens = min(max(args.tokens), 4096)

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
    max_elems = max(args.tokens) * HIDDEN
    ctx = Ctx(rank, world, max_elems, args.fi_max_tokens)
    if rank == 0:
        print(f"{torch.cuda.get_device_name()}  world={world}  "
              f"sizes={[t * HIDDEN * 2 // 1024 for t in args.tokens]} KB",
              flush=True)

    rows = []
    for tokens in args.tokens:
        if rank == 0:
            print(f"\ntokens={tokens}  ({tokens * HIDDEN * 2 / 1e6:.2f} MB)",
                  flush=True)
        rows += impl_rows(ctx, tokens, args)

    dist.barrier()
    ctx.destroy()
    if rank == 0:
        summary(rows)
        res = Path(__file__).parent / "results"
        res.mkdir(exist_ok=True)
        csv_path = res / f"bench_comm_tp{world}.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {csv_path}")
        try:
            png = plot(rows, csv_path.with_suffix(".png"))
            print(f"Wrote {png}")
        except Exception as e:  # noqa: BLE001 - plot is optional
            print(f"plot skipped: {e}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
