#!/usr/bin/env python3
"""Kimi-K3 MoE bench: TRT-LLM baseline vs b10, with per-opt ablation.

Runs inside the trt-dev container (installed tensorrt_llm 1.3.0rc23):

  # TP1 quick check
  docker exec trt-dev python3 \
      /workspace/diffusion_inference/LLMDiveDeep/kimi_k3_layer/bench_moe_kimi_k3.py

  # TP8, small batches (<= 256 tokens: interactive dev), full-opt only
  docker exec trt-dev bash -c "cd /workspace/diffusion_inference/LLMDiveDeep \
      && mpirun -n 8 --allow-run-as-root \
         python3 kimi_k3_layer/bench_moe_kimi_k3.py"

  # ablation: each optimization switched off one at a time
  ... bench_moe_kimi_k3.py --ablate

  # TP8, large batches (separate long-running command; falls back to
  # the baseline path beyond OPT_MAX_TOKENS by design)
  ... bench_moe_kimi_k3.py --sizes large

  # capture a torch-profiler trace of baseline + full-opt at one batch
  ... bench_moe_kimi_k3.py --profile 8

Correctness: every configuration is compared against the baseline
KimiK3MoE.forward ON THE SAME module instance (identical weights).
Timing: CUDA-graph replay, max over ranks.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import types
from pathlib import Path

import torch

_LLMDIR = Path(__file__).resolve().parents[1]
_WORKSPACE = _LLMDIR.parent

SMALL_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
LARGE_SIZES = (512, 1024, 2048, 4096, 8192)

# ablation matrix: label -> set_opt_flags kwargs (first = everything on)
FULL = dict(merged_gemm=True, routing="ours", fc1_shard=True,
            comm="fi", overlap_shared=True, prefill_opt=True)
ABLATIONS = {
    "opt": FULL,
    "-merged": {**FULL, "merged_gemm": False},
    "-routing": {**FULL, "routing": "noaux"},
    "-fc1shard": {**FULL, "fc1_shard": False},
    "-fc2shard": {**FULL, "fc2_shard": False},
    # ref_tail off: the fc2-shard tail (RS+norm -> 1/world fc2 ->
    # AR) runs at ALL sizes instead of switching to the REF-overlap
    # tail (AR+norm -> full fc2, shared AR on the aux stream) at
    # B>=REF_TAIL_MIN_TOKENS
    "-reftail": {**FULL, "ref_tail": False},
    # finalize+AR+norm fusion off: ref tail keeps the separate
    # finalizeKernel + fused AR+norm pair
    "-finfuse": {**FULL, "fin_fuse": False},
    # three-way input merge off: shared gate/up runs as its own GEMM
    # on the aux stream instead of riding the wide input GEMM
    "-merge3": {**FULL, "merge3": False},
    # comm default is now "fi" (won on this node with merge3);
    # this column re-enables the custom Lamport RS+norm tail
    "+customrs": {**FULL, "comm": "custom"},
    "-overlap": {**FULL, "overlap_shared": False},
    # TIMING PROBE, output intentionally wrong (err column is
    # expected to be large): drops the shared chain entirely. opt
    # minus -shared = shared-chain time NOT hidden by the routed path
    "-shared": {**FULL, "skip_shared": True},
    # every switch off ~= the baseline re-implemented in our forward:
    # separates real optimization effects from harness/tactic noise
    "-all": {**FULL, "merged_gemm": False, "routing": "noaux",
             "fc1_shard": False, "fc2_shard": False, "comm": "fi",
             "overlap_shared": False, "ref_tail": False},
}


def bootstrap() -> None:
    """Container's installed tensorrt_llm stays authoritative; register
    kimi_k3_layer as a plain namespace package (its __init__ is empty
    but the parent dir also holds unrelated benchmark stacks)."""
    import tensorrt_llm  # noqa: F401  (real runtime, first!)

    if "kimi_k3_layer" not in sys.modules:
        pkg = types.ModuleType("kimi_k3_layer")
        pkg.__path__ = [str(_LLMDIR / "kimi_k3_layer")]
        sys.modules["kimi_k3_layer"] = pkg
    if str(_LLMDIR) not in sys.path:
        sys.path.insert(0, str(_LLMDIR))


def _plot(rows, path, world):
    """Two panels: latency curves + improvement-% bars per batch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(r["config"] for r in rows))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]})
    for label in labels:
        pts = [(r["batch"], r["latency_us"]) for r in rows
               if r["config"] == label]
        pts.sort()
        style = {"baseline": dict(color="k", lw=2.5),
                 "opt": dict(color="tab:green", lw=2.5)}.get(
                     label, dict(lw=1.2, alpha=0.7))
        ax1.plot(*zip(*pts), marker="o", ms=3, label=label, **style)
    ax1.set_xscale("log", base=2)
    ax1.set_ylabel("latency (us)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)

    batches = sorted({r["batch"] for r in rows})
    ab_labels = [lb for lb in labels if lb != "baseline"]
    width = 0.8 / max(len(ab_labels), 1)
    for i, label in enumerate(ab_labels):
        wins = {r["batch"]: r["win_pct"] for r in rows
                if r["config"] == label}
        xs = [x * (1 + (i - len(ab_labels) / 2) * width * 0.5)
              for x in batches]
        color = "tab:green" if label == "opt" else None
        bars = ax2.bar(xs, [wins.get(b, 0) for b in batches],
                       width=[x * width * 0.5 for x in batches],
                       label=label, color=color,
                       alpha=1.0 if label == "opt" else 0.5)
        if label == "opt":
            for rect, b in zip(bars, batches):
                ax2.annotate(f"+{wins.get(b, 0):.0f}%",
                             (rect.get_x() + rect.get_width() / 2,
                              rect.get_height()),
                             ha="center", va="bottom", fontsize=8)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(batches, [str(b) for b in batches])
    ax2.set_xlabel("batch (tokens)")
    ax2.set_ylabel("improvement vs baseline (%)")
    ax2.grid(alpha=0.3, axis="y")
    if len(ab_labels) > 1:
        ax2.legend(fontsize=8, ncol=3)
    fig.suptitle(f"TP{world} Kimi-K3 MoE decode: TRT-LLM baseline vs "
                 "b10 (graph replay, max over ranks)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _write_md_table(rows, path, configs):
    """baseline / opt / improvement per batch, plus ablation columns."""
    by = {(r["batch"], r["config"]): r for r in rows}
    batches = sorted({r["batch"] for r in rows})
    labels = [lb for lb in configs if lb != "opt"]
    with open(path, "w") as f:
        head = "| B | baseline us | opt us | improvement |"
        sep = "|---|---|---|---|"
        head += "".join(f" {lb} us |" for lb in labels)
        sep += "---|" * len(labels)
        f.write(head + "\n" + sep + "\n")
        for b in batches:
            base = by.get((b, "baseline"))
            opt = by.get((b, "opt"))
            if not base or not opt:
                continue
            line = (f"| {b} | {base['latency_us']:.1f} "
                    f"| {opt['latency_us']:.1f} "
                    f"| **{opt['win_pct']:+.0f}%** |")
            for lb in labels:
                r = by.get((b, lb))
                line += (f" {r['latency_us']:.1f} ({r['win_pct']:+.0f}%) |"
                         if r else " - |")
            f.write(line + "\n")
    return path


def init_weights(moe, world: int, rank: int, seed: int = 0) -> None:
    """Seed-identical GLOBAL tensors on every rank, sliced into the
    rank's shard, so TP outputs are comparable across world sizes."""
    from kimi_k3_layer.config import (
        HIDDEN,
        MOE_INTER,
        MOE_LATENT,
        NUM_EXPERTS,
        SHARED_INTER,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, std=0.02):
        return torch.randn(*shape, generator=gen, device="cuda",
                           dtype=torch.bfloat16) * std

    with torch.no_grad():
        moe.gate.weight.copy_(randn(NUM_EXPERTS, HIDDEN))
        # bias std must stay well below the logits' std (~1.7): a large
        # per-expert bias acts as a lottery all tokens agree on and
        # concentrates top-16 onto 7-14 hot experts/rank (3x EP load
        # spread, artificially fast + imbalanced experts). Trained K3
        # gates are aux-loss balanced -> near-uniform expert load.
        moe.gate.e_score_correction_bias.copy_(
            randn(NUM_EXPERTS, std=0.02).to(
                moe.gate.e_score_correction_bias.dtype))
        moe.fc1_latent_proj.weight.copy_(randn(MOE_LATENT, HIDDEN))
        moe.fc2_latent_proj.weight.copy_(randn(HIDDEN, MOE_LATENT))
        moe.latent_norm.weight.copy_(
            1 + randn(MOE_LATENT).to(moe.latent_norm.weight.dtype))

        # shared expert: column-TP gate_up [2i_l, H], row-TP down [H, i_l]
        i_local = SHARED_INTER // world
        g_full = randn(SHARED_INTER, HIDDEN)
        u_full = randn(SHARED_INTER, HIDDEN)
        d_full = randn(HIDDEN, SHARED_INTER)
        rows = slice(rank * i_local, (rank + 1) * i_local)
        moe.shared_experts.gate_up_proj.weight.copy_(
            torch.cat([g_full[rows], u_full[rows]]))
        moe.shared_experts.down_proj.weight.copy_(d_full[:, rows])

        # EP experts: full [896, ...] tensors, rank owns a contiguous
        # slice. CUTLASS backend: bf16 tensors. TRTLLM (production)
        # backend: packed MXFP4 weights (w4a8_mxfp4_mxfp8: MXFP8
        # activations quantized at runtime) - random fp4 nibbles (e2m1
        # has no NaN encodings), ue8m0 block scales fixed at
        # 124 = 2^-3, biases zero.
        backend = getattr(moe.experts, "backend", moe.experts)
        if type(backend).__name__ == "TRTLLMGenFusedMoE":
            for name, p in backend.named_parameters():
                full_shape = (NUM_EXPERTS, *p.shape[1:])
                e_local = p.shape[0]
                es = slice(rank * e_local, (rank + 1) * e_local)
                if p.dtype == torch.uint8 and "scale" in name:
                    p.fill_(124)
                elif p.dtype == torch.uint8:
                    full = torch.randint(
                        0, 256, full_shape, generator=gen,
                        device="cuda", dtype=torch.int64,
                    ).to(torch.uint8)
                    p.copy_(full[es])
                else:
                    p.zero_()
            if hasattr(backend, "post_load_weights"):
                backend.post_load_weights()
        else:
            w31_full = randn(NUM_EXPERTS, 2 * MOE_INTER, MOE_LATENT)
            w2_full = randn(NUM_EXPERTS, MOE_LATENT, MOE_INTER)
            w31_l, w2_l = moe._expert_weights()
            e_local = w31_l.shape[0]
            es = slice(rank * e_local, (rank + 1) * e_local)
            w31_l.copy_(w31_full[es])
            w2_l.copy_(w2_full[es])
    torch.cuda.synchronize()


def capture(fn, iters: int, world: int = 1):
    """Warmup + capture a CUDA graph (production decode replays graphs,
    so this measures the production configuration). Production's
    cuda_graph_runner wraps capture in with_multi_stream(True), which
    is what lets the baseline's maybe_execute_in_parallel fork the
    shared expert onto its aux stream - replicate that here. The first
    warmup pass runs under the AUTOTUNER (production's
    _run_autotuner_warmup populates the tactic cache before capture -
    without it every tunable op runs its default heuristic)."""
    import torch.distributed as dist

    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.modules.multi_stream_utils import (
        with_multi_stream,
    )

    with with_multi_stream(True):
        with autotune():
            fn()
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iters):
                fn()
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    return graph


def time_graph(graph, iters: int, world: int = 1, *,
               h: torch.Tensor | None = None, n_inputs: int = 1,
               seed: int = 0):
    """Replay timing, max over ranks. With h + n_inputs > 1 the input
    buffer is REWRITTEN with a fresh rank-identical tensor before each
    timed replay, so the reported number is the MEAN over n_inputs
    routings (EP expert load - and the skew every collective absorbs -
    is data-dependent; a single input is one draw from that
    distribution). Restores h afterwards."""
    import statistics

    import torch.distributed as dist

    h_orig = h.clone() if (h is not None and n_inputs > 1) else None
    samples = []
    for k in range(max(n_inputs, 1)):
        if h_orig is not None:
            gen = torch.Generator(device="cuda").manual_seed(
                97 + 131 * seed + k)
            h.copy_(torch.randn(h.shape, generator=gen, device="cuda",
                                dtype=torch.float32).to(h.dtype))
            torch.cuda.synchronize()
            if world > 1:
                dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        t = start.elapsed_time(end) * 1000 / iters
        if world > 1:
            tt = torch.tensor([t], dtype=torch.float64)
            dist.all_reduce(tt, op=dist.ReduceOp.MAX)
            t = tt.item()
        samples.append(t)
    if h_orig is not None:
        h.copy_(h_orig)
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        graph.replay()  # regenerate graph outputs with the original h
        torch.cuda.synchronize()
    return statistics.mean(samples)


def profile_graph(graph, path: Path, rank: int, world: int = 1,
                  iters: int = 5, perturb=None) -> None:
    """Profile CUDA-graph REPLAYS only: kernel durations are the
    production steady state (no launch overhead, minimal arrival
    skew - comm kernels show their real cost). Eager profiling was
    removed on purpose: its traces are dominated by CPU launch
    overhead and per-rank arrival skew, which made every collective
    look pathologically long.

    perturb(i): rewrites the graph's input buffer before replay i
    (identical on every rank). Each replay then routes DIFFERENT
    tokens, so per-rank expert load - and which rank the RS waits
    for - reshuffles per replay block."""
    import torch.distributed as dist

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU],
    ) as prof:
        for i in range(iters):
            if perturb is not None:
                perturb(i)
                torch.cuda.synchronize()
                if world > 1:
                    dist.barrier()
            graph.replay()
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
    _export_trace(prof, path, rank, world)


ALL_RANK_TRACES = os.environ.get("BENCH_ALL_RANK_TRACES", "0") == "1"


def _export_trace(prof, path: Path, rank: int, world: int = 1) -> None:
    """Rank 0 exports; BENCH_ALL_RANK_TRACES=1 adds per-rank files
    (_r{rank} suffix; same host clock -> cross-rank alignment works).
    The trailing barrier is LOAD-BEARING: without it, fast ranks race
    ahead into the next phase's spin-wait collectives while slow
    ranks are still serializing JSON, and the pipeline can wedge."""
    import torch.distributed as dist

    path.parent.mkdir(parents=True, exist_ok=True)
    if ALL_RANK_TRACES:
        # Kineto allows exactly one export per profiler session, so rank 0
        # exports to the per-rank file and copies it to the main path.
        per_rank = path.with_name(
            path.name.replace(".trace.json", f"_r{rank}.trace.json"))
        prof.export_chrome_trace(str(per_rank))
        if rank == 0:
            shutil.copyfile(per_rank, path)
            print(f"[profile] wrote {path}", flush=True)
    elif rank == 0:
        prof.export_chrome_trace(str(path))
        print(f"[profile] wrote {path}", flush=True)
    if world > 1:
        dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="small",
                        help="small | large | comma list of token counts")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--n-inputs", type=int, default=8,
        help="average the timing over this many fresh rank-identical "
        "inputs (routing/EP-skew is data-dependent; 1 = old behavior)")
    parser.add_argument(
        "--ablate", action="store_true",
        help="run every opt configuration (each switch off one at a "
        "time), not just full-opt")
    parser.add_argument(
        "--profile", default="", metavar="B[,B...]",
        help="export CUDA-graph-replay traces of baseline + full-opt "
        "at these batch sizes (results/); graph replay only - eager "
        "traces are dominated by CPU launch overhead")
    parser.add_argument(
        "--moe-class", default="b10",
        choices=("b10", "agg", "disagg"),
        help="b10 = research class (all switches; required for "
        "--ablate); agg = KimiK3MoEForAgg (deployment layout: fc1+fc2 "
        "sharded, prefill-first; B10_AGG_FULL_FC2=1 adds the one-time "
        "fc2 weight all-gather); disagg = KimiK3MoEForDisAggDecode "
        "(decode-only layout, full fc2, ref tail at B>=16)")
    parser.add_argument(
        "--backend", default="trtllm", choices=("trtllm", "cutlass"),
        help="expert backend: trtllm = PRODUCTION K3 stack "
        "(TRTLLMGenFusedMoE, packed MXFP4, in-kernel routing; the b10 "
        "opt path is not adapted to it yet, so only the baseline is "
        "benched); cutlass = bf16 comparison stack for the opt "
        "ablations")
    args = parser.parse_args()
    if args.sizes == "small":
        sizes = SMALL_SIZES
    elif args.sizes == "large":
        sizes = LARGE_SIZES
    else:
        sizes = tuple(int(s) for s in args.sizes.split(","))
    profile_sizes = {int(b) for b in args.profile.split(",") if b}
    sizes = tuple(sorted(set(sizes) | profile_sizes))

    bootstrap()
    import torch.distributed as dist

    from tensorrt_llm._torch.utils import AuxStreamType

    from kimi_k3_layer.config import HIDDEN, MOE_LATENT
    from kimi_k3_layer.moe_b10_kimi_k3 import (
        FC1_SHARD_MAX_TOKENS,
        PREFILL_MIN_TOKENS,
        KimiK3MoEB10,
    )
    from kimi_k3_layer.moe_trtllm_kimi_k3 import KimiK3MoE, k3_model_config

    # launched via mpirun (trtllm's native session; its AllReduce needs
    # MPI in this build) - torch.distributed is inited from the MPI env
    rank = int(os.environ.get(
        "RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world = int(os.environ.get(
        "WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world,
            device_id=torch.device("cuda", rank),
        )

    if args.moe_class != "b10":
        from kimi_k3_layer.moe_deploy_kimi_k3 import (
            KimiK3MoEForAgg,
            KimiK3MoEForDisAggDecode,
        )
        if args.ablate:
            parser.error("--ablate needs --moe-class b10 (deployment "
                         "classes freeze their configuration at init)")
    moe_cls = {"b10": KimiK3MoEB10}.get(args.moe_class) or (
        KimiK3MoEForAgg if args.moe_class == "agg"
        else KimiK3MoEForDisAggDecode)

    model_config = k3_model_config(
        rank, world, moe_backend=args.backend.upper())
    aux = {k: torch.cuda.Stream() for k in AuxStreamType}
    moe = moe_cls(
        model_config, layer_idx=0, aux_stream_dict=aux,
        reduce_output=world > 1,
    ).cuda()
    init_weights(moe, world, rank)

    # opt runs on both backends: cutlass drives fused_moe with our
    # routing; trtllm (production) calls run_moe with our pre-computed
    # top-k so the runner skips its in-kernel scores+top-k stage
    # (the "-routing" ablation restores in-kernel routing)
    opt_enabled = True
    opt_sizes = [s for s in sizes if s <= KimiK3MoEB10.OPT_MAX_TOKENS]
    max_batch = max(opt_sizes) if opt_sizes else 1
    fi_ar = fi_ar_sh = ag = rs_comm = ag_ce = None
    if world > 1 and opt_sizes:
        from kimi_k3_layer.comm import (
            FlashInferAllReduce,
            build_fc1_ag,
            build_latent_rs,
        )

        fi_ar = FlashInferAllReduce(rank, world, max_tokens=max_batch)
        # separate workspace for the ref_tail shared AR (it runs
        # concurrently with fi_ar's latent norm_reduce)
        fi_ar_sh = FlashInferAllReduce(rank, world, max_tokens=max_batch)
        ag = build_fc1_ag(rank, world, max_batch)
        rs_comm = build_latent_rs(rank, world, max_batch)
    pf_sizes = [s for s in sizes if s >= PREFILL_MIN_TOKENS
                and opt_enabled]
    if world > 1 and pf_sizes:
        from kimi_k3_layer.comm import CeComm

        ag_ce = CeComm(rank, world,
                       max_bytes=max(pf_sizes) * MOE_LATENT * 2)
    if opt_enabled:
        moe.init_opt(max_batch=max_batch, fi_ar=fi_ar,
                     fi_ar_shared=fi_ar_sh, ag=ag,
                     rs_comm=rs_comm, ag_ce=ag_ce)

    configs = (ABLATIONS if args.ablate else {"opt": ABLATIONS["opt"]}) \
        if opt_enabled else {}
    if rank == 0:
        print(f"TP{world} Kimi-K3 MoE ({args.backend} experts): "
              f"TRT-LLM baseline vs b10 "
              f"(configs: {', '.join(configs) or 'baseline only'}; "
              f"fc1-shard active at B<={FC1_SHARD_MAX_TOKENS})")

    sizes = tuple(sorted(sizes))
    inputs = {}
    outs, times = {}, {}

    # Phase 1: all opt configurations, graph-captured. (The opt path
    # calls flashinfer only for the oneshot AR, which owns a dedicated
    # IPC workspace - no allocator-lifetime interplay with phase 2.)
    for batch in sizes:
        torch.manual_seed(batch)
        h = torch.randn(batch, HIDDEN, device="cuda",
                        dtype=torch.bfloat16)
        # non-DP TP: the scheduler fills this with [num_tokens] (the
        # installed configurable_moe asserts a single-element list)
        meta = types.SimpleNamespace(all_rank_num_tokens=[batch])
        inputs[batch] = (h, meta)
        prefill_ok = ag_ce is not None and batch >= PREFILL_MIN_TOKENS
        if batch > max_batch and not prefill_ok:
            continue  # opt falls back to baseline; bench in phase 2

        # tune the AG plan for this shape OUTSIDE graph capture
        if ag is not None and batch <= FC1_SHARD_MAX_TOKENS:
            ag.tune(torch.zeros(
                batch, moe._latent_width, device="cuda",
                dtype=torch.bfloat16))

        for label, flags in configs.items():
            if args.moe_class == "b10":
                moe.set_opt_flags(**flags)
            if rank == 0:
                print(f"[phase1] B={batch}: capture {label}", flush=True)
            res = {}

            def opt_fn(h=h, meta=meta, res=res):
                with torch.no_grad():
                    res["out"] = moe(h, meta)

            g = capture(opt_fn, iters=args.iters, world=world)
            times[batch, label] = time_graph(
                g, args.iters, world,
                h=h, n_inputs=args.n_inputs, seed=batch)
            outs[batch, label] = res["out"].clone()
            # BENCH_PROFILE_CONFIG selects which ablation config to
            # trace (default "opt"); needs --ablate for other labels.
            if batch in profile_sizes and label == os.environ.get(
                    "BENCH_PROFILE_CONFIG", "opt"):
                tag = label.replace("-", "no_")
                res_dir = _LLMDIR / "kimi_k3_layer" / "results"
                perturb = None
                if os.environ.get("BENCH_PERTURB_INPUT", "0") == "1":
                    def perturb(i, h=h):
                        # identical fresh input on every rank -> the
                        # routing (and per-rank expert load) changes
                        # per replay block
                        gen = torch.Generator().manual_seed(31337 + i)
                        h.copy_(torch.randn(
                            h.shape, generator=gen,
                            dtype=torch.float32).to(h.dtype))
                h_orig = h.clone() if perturb else None
                profile_graph(
                    g,
                    res_dir
                    / f"moe_kimi_k3_tp{world}_b{batch}_{tag}_graph"
                    ".trace.json",
                    rank, world,
                    iters=10 if perturb else 5, perturb=perturb)
                if h_orig is not None:
                    # phase 2 reuses this buffer for the baseline
                    # correctness ref - undo the perturbation
                    h.copy_(h_orig)
                    torch.cuda.synchronize()
            del g
        if args.moe_class == "b10":
            moe.set_opt_flags(**ABLATIONS["opt"])

    # Phase 2: baseline (real TRT-LLM modules, eager ref + captured).
    rows = []
    if rank == 0:
        cols = " ".join(f"{c + '_us':>13}" for c in configs)
        errs = " ".join(f"{'err(' + c + ')':>13}" for c in configs)
        print(f"{'B':>5} {'base_us':>9} {cols} {errs}")
    for batch in sizes:
        h, meta = inputs[batch]
        if rank == 0:
            print(f"[phase2] B={batch}: baseline ref + capture",
                  flush=True)
        with torch.no_grad():
            ref = KimiK3MoE.forward(moe, h, meta)
        ref_max = ref.float().abs().max()

        def base_fn(h=h, meta=meta):
            with torch.no_grad():
                KimiK3MoE.forward(moe, h, meta)

        g = capture(base_fn, iters=args.iters, world=world)
        t_base = time_graph(g, args.iters, world,
                            h=h, n_inputs=args.n_inputs, seed=batch)
        if batch in profile_sizes:
            res_dir = _LLMDIR / "kimi_k3_layer" / "results"
            profile_graph(
                g,
                res_dir
                / f"moe_kimi_k3_tp{world}_b{batch}_base_graph"
                ".trace.json",
                rank, world)
        del g
        if rank == 0:
            rows.append({"batch": batch, "config": "baseline",
                         "latency_us": t_base, "win_pct": 0.0,
                         "rel_err": 0.0})
            tcols, ecols = [], []
            for label in configs:
                t = times.get((batch, label))
                o = outs.get((batch, label))
                if t is None:
                    tcols.append(f"{'-':>13}")
                    ecols.append(f"{'-':>13}")
                    continue
                win = (t_base - t) / t_base * 100
                rel = ((o.float() - ref.float()).abs().max()
                       / ref_max).item()
                rows.append({"batch": batch, "config": label,
                             "latency_us": t, "win_pct": win,
                             "rel_err": rel})
                tcols.append(f"{t:>7.2f}{win:>+5.0f}%")
                ecols.append(f"{rel:>13.2e}")
            print(f"{batch:>5} {t_base:>9.2f} {' '.join(tcols)} "
                  f"{' '.join(ecols)}", flush=True)
        if world > 1:
            dist.barrier()

    if rank == 0 and rows:
        out = (_LLMDIR / "kimi_k3_layer" / "results"
               / f"bench_moe_kimi_k3_tp{world}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        figure = _plot(rows, out.with_suffix(".png"), world)
        table = _write_md_table(rows, out.with_suffix(".md"), configs)
        print(f"Wrote {out}")
        print(f"Wrote {figure}")
        print(f"Wrote {table}")

    if fi_ar is not None:
        fi_ar.destroy()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
