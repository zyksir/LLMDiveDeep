#!/usr/bin/env python3
"""End-to-end 4-block Kimi-K3 bench: TP vs CP (vs CP+MegaMoE) at world N.

Drives the existing ``block4_demo`` stack (3 KDA + 1 MLA transformer blocks,
each wrapping the shared tp_baseline MoE layer) over a decode/prefill grid.
One code path; ``--modes`` selects configuration only:

- ``tp``    attention TP-sharded (o_proj allreduce per block), shared experts
            TP-sharded + allreduce, routed EP+A2A on the local slice with an
            allgather restoring the full token set — the finalized TRT-LLM
            reference wiring.
- ``cp``    every rank owns whole sequences (batch/world). Zero attention
            collectives; the MoE A2A is the only cross-rank communication
            (verified via comm_instrument on the first case).
- ``cp_mm`` same as cp with the MegaMoEDeepGemm routed path (comm fused
            in-kernel).

Weights are the stack's deterministic synthetic init (perf/comm fidelity;
TP shards are not slices of CP's weights, so no cross-mode numerical check —
the CP-mode gate is comm_instrument's "MoE A2A only" verdict). Decode runs
from zero-filled caches with num_cached=[isl] (kernel shapes and traffic
identical to a real history; values synthetic), per runtime.py.

Launch (inside trt-k3-bench):
  mpirun --allow-run-as-root -np 8 python3 kimi_k3/block4_demo/bench_block4.py \
      --modes tp cp cp_mm --decode-batch-sizes 8 32 128 --isl 4096
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

_PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PACKAGE_DIR))
sys.path.insert(0, str(_PACKAGE_DIR.parent / "parallel_strategy_search" / "moe"))

from harness import REPO_ROOT, setup_sys_path, verify_provenance  # noqa: E402

setup_sys_path()
sys.path.insert(0, str(REPO_ROOT / "kimi_k3"))

import torch  # noqa: E402

from bench_moe.mapping import _build_mapping_from_config  # noqa: E402
from bench_moe.specs import ConfigSpec  # noqa: E402
from bench_moe.utils import _set_device_from_local_rank  # noqa: E402
from tensorrt_llm._torch.autotuner import AutoTuner  # noqa: E402
from tensorrt_llm._utils import mpi_allgather, mpi_barrier, mpi_rank, mpi_world_size  # noqa: E402

import bench_layer  # noqa: E402  (tp_baseline driver: builders + inputs)
from bench_layer import _make_case_inputs, _run_layer_autotune, _stats  # noqa: E402
from layer import KimiK3MoELayerBaseline, KimiK3SharedExperts  # noqa: E402

import comm_instrument  # noqa: E402


def _install_wide_gated_rmsnorm_fallback() -> None:
    """Torch fallback for the b10 gated-RMSNorm epilogue above 64 heads/rank.

    The b10 kernel caps at H*D/8 <= 1024 threads (64 heads at D=128); CP-mode
    attention keeps all 96 heads on one rank, so its prefill epilogue needs a
    fallback. Semantics mirror the kernel doc: y = rmsnorm(x)*act(z), applied
    only when the kernel would reject the shape — TP-mode shapes still take
    the fused kernel.
    """
    import tensorrt_llm._torch.modules.mamba.kda_mixer as kda_mixer

    original = kda_mixer.b10_gated_rmsnorm

    def gated_rmsnorm(x, z, weight, eps, gate_activation="sigmoid", out=None):
        tokens, heads, head_dim = x.shape
        if heads * head_dim // 8 <= 1024:
            return original(x, z, weight, eps, gate_activation=gate_activation, out=out)
        xf = x.float()
        normed = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        normed = normed * weight.float()
        zf = z.float()
        gate = torch.sigmoid(zf) if gate_activation == "sigmoid" else zf * torch.sigmoid(zf)
        result = (normed * gate).to(x.dtype)
        if out is not None:
            out_view = out.view(tokens, heads, head_dim)
            out_view.copy_(result)
            return out_view
        return result

    kda_mixer.b10_gated_rmsnorm = gated_rmsnorm
from runtime import DemoRuntime  # noqa: E402
from stack import KimiK3Block4Stack  # noqa: E402

LOCAL_RESULTS = _PACKAGE_DIR / "local_results"
NUM_EXPERTS = bench_layer.NUM_EXPERTS
HIDDEN = 7168


def _build_moe_layer(mode: str, args, mapping, max_local: int, device,
                     max_global: int):
    """MoE sublayer per mode.

    tp / cp: the PRODUCTION KimiK3MoE (SiTU experts, latent 3584; tp = fused
    finalize-AR epilogue, cp = NVLinkOneSided A2A) — mode is the mapping flag.
    cp_mm: legacy bench layer with the MegaMoE routed path (KimiK3MoE cannot
    host MegaMoE: SiTU requires TRTLLMGenFusedMoE).
    """
    if mode in ("tp", "tp_te", "cp"):
        # b10 fast paths segfault in this bench context (open issue); the
        # standard fused finalize-AR epilogue is unaffected.
        import os as _os

        _os.environ.setdefault("ENABLE_B10_SHARD_FC1", "0")
        _os.environ.setdefault("ENABLE_B10_COLLECTIVES", "0")
        from bench_k3_moe import build_k3_moe

        moe = build_k3_moe(
            mode=mode,
            world=mpi_world_size(),
            rank=mpi_rank(),
            device=device,
            max_num_tokens=max_local if mode == "cp" else max_global,
        )
        moe.is_production_k3_moe = True
        return moe, moe.experts
    routed = bench_layer._build_routed_cached(
        mapping=mapping,
        max_local_tokens=max_local,
        device=device,
        weight_cache_dir=args.weight_cache_dir,
        backend="MEGAMOE_DEEPGEMM",
        comm_method="NONE",
    )
    shared = KimiK3SharedExperts(
        mode="cp", world_size=mpi_world_size(), rank=mpi_rank(), device=device
    )
    return KimiK3MoELayerBaseline(
        mode="cp", routed_moe=routed, shared_experts=shared, rank=mpi_rank()
    ), routed


def _multi_stream():
    # Production decode overlap (shared experts on the MoeShared stream,
    # fc1-shard chain, 6-aux-stream layout) is gated on do_multi_stream();
    # capture AND replay must run under it or the overlap is silently off
    # (k3-production-moe-bench trap #6 — was missing here until 2026-08-31).
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    return with_multi_stream(True)


def _benchmark(run: Callable[[], torch.Tensor], warmup: int, iters: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            run()
    torch.cuda.synchronize()
    mpi_barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.inference_mode():
        for i in range(iters):
            starts[i].record()
            run()
            ends[i].record()
    torch.cuda.synchronize()
    return _stats([starts[i].elapsed_time(ends[i]) for i in range(iters)])


def _benchmark_graph(run: Callable[[], torch.Tensor], warmup: int, iters: int) -> dict:
    """Event-timed CUDA-graph replay loop (serving-realistic latencies)."""
    with torch.inference_mode(), _multi_stream():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        mpi_barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        mpi_barrier()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
    return _stats([starts[i].elapsed_time(ends[i]) for i in range(iters)])


def _capture_trace(run: Callable[[], torch.Tensor], out_path: Path) -> str:
    with torch.inference_mode():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        mpi_barrier()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            run()
            torch.cuda.synchronize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out_path))
    return str(out_path)


def _capture_graph_trace(run: Callable[[], torch.Tensor], out_path: Path) -> str:
    """Trace ONE CUDA-graph replay of the full 4-block forward.

    Eager traces at ~1ms latencies are dominated by launch gaps; a graph
    replay shows the kernels back-to-back, so the trace reflects kernel-level
    behavior instead of scheduling noise. Capture is barrier-synchronized so
    the collectives (MoE A2A / allreduces) are captured on all ranks together.
    """
    with torch.inference_mode(), _multi_stream():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        mpi_barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()
        mpi_barrier()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        mpi_barrier()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(3):  # read steady state from replay 2+ (skew absorb)
                graph.replay()
            torch.cuda.synchronize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out_path))
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=("tp", "tp_te", "cp", "cp_mm"),
                        default=["tp", "cp"])
    parser.add_argument("--decode-batch-sizes", type=int, nargs="*",
                        default=[8, 32, 128])
    parser.add_argument("--prefill-seqs", type=int, nargs="*", default=[8])
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--phases", nargs="+", choices=("decode", "prefill"),
                        default=["decode", "prefill"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--graph-timing", action="store_true",
                        help="time CUDA-graph replays instead of eager launches")
    parser.add_argument("--graph-trace", action="store_true",
                        help="trace one CUDA-graph replay instead of an eager "
                        "iteration (bubble-free kernel timeline)")
    parser.add_argument("--verify-comm", action="store_true",
                        help="comm_instrument verdict on the first case per mode")
    parser.add_argument("--tag", default="block4")
    parser.add_argument("--weight-cache-dir", type=Path,
                        default=REPO_ROOT / "out" / "moe_weight_cache")
    args = parser.parse_args()

    rank, world = mpi_rank(), mpi_world_size()
    device = torch.device("cuda", _set_device_from_local_rank())
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "30011")
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    _install_wide_gated_rmsnorm_fallback()
    moe_mapping = _build_mapping_from_config(
        ConfigSpec(backend="TRTLLM", parallel_mode="DEP"), world
    )
    AutoTuner.get().setup_distributed_state(moe_mapping)

    # Global token counts across the grid decide the MoE workspace size.
    grid = []
    if "decode" in args.phases:
        grid += [("decode", b) for b in args.decode_batch_sizes]
    if "prefill" in args.phases:
        grid += [("prefill", b) for b in args.prefill_seqs]
    max_global_tokens = max(
        (b if phase == "decode" else b * args.isl) for phase, b in grid
    )
    max_local = max(bench_layer._per_rank_tokens(max_global_tokens, world))

    rows = []
    comm_verdicts = {}
    for mode in args.modes:
        moe_layer, routed = _build_moe_layer(
            mode, args, moe_mapping, max_local, device, max_global_tokens)
        stack = KimiK3Block4Stack(
            mode="cp" if mode in ("cp", "cp_mm") else "tp",
            world_size=world,
            rank=rank,
            device=device,
            moe_layer=moe_layer,
        )
        for phase, size in grid:
            global_tokens = size if phase == "decode" else size * args.isl
            per_rank = bench_layer._per_rank_tokens(global_tokens, world)
            inputs = _make_case_inputs(
                global_tokens=global_tokens,
                rank=rank,
                world_size=world,
                num_experts=NUM_EXPERTS,
                routing_method=routed.routing_method,
                device=device,
                seed=args.seed,
                need_full=(mode in ("tp", "tp_te")),
            )
            # Attention runtime: TP sees the full global batch on every rank;
            # CP sees batch/world whole sequences.
            if mode in ("tp", "tp_te"):
                runtime_batch = size
            else:
                assert size % world == 0, "batch must divide world for CP modes"
                runtime_batch = size // world
            runtime = DemoRuntime(
                attn_model_config=stack.attn_model_config,
                batch_size=runtime_batch,
                isl=args.isl,
                phase=phase,
                device=device,
            )
            hidden = runtime.make_hidden_states(args.seed + size)
            position_ids = runtime.position_ids

            # MLA's registered custom op resolves its metadata through
            # model_extra_attrs (test_kimi_linear.py driving pattern).
            import weakref

            from tensorrt_llm._torch.utils import model_extra_attrs

            attrs = stack.attn_model_config.extra_attrs
            if attrs is None:
                attrs = {}
            attrs["attention_metadata"] = weakref.ref(runtime.attn_metadata)

            def run(stack=stack, hidden=hidden, position_ids=position_ids,
                    runtime=runtime, inputs=inputs, attrs=attrs):
                with model_extra_attrs(attrs):
                    return stack.forward(
                        hidden,
                        position_ids=position_ids,
                        attn_metadata=runtime.attn_metadata,
                        mamba_metadata=runtime.mamba_metadata,
                        router_logits_local=inputs["logits_local"],
                        all_rank_num_tokens=inputs["all_rank_num_tokens"],
                    )

            AutoTuner.get().clear_cache()
            _run_layer_autotune(moe_layer, run)
            if args.verify_comm and (phase, size) == grid[0]:
                with comm_instrument.count_comm_calls(stack) as counts:
                    with torch.inference_mode():
                        run()
                    torch.cuda.synchronize()
                kernels = comm_instrument.profile_comm_kernels(run)
                verdict = (
                    comm_instrument.cp_verdict(counts, kernels)
                    if mode != "tp" else {"passed": None, "note": "tp mode"}
                )
                comm_verdicts[mode] = {
                    "python_call_counts": dict(counts),
                    "verdict": verdict,
                }
            if args.graph_timing:
                timing = _benchmark_graph(run, args.warmup, args.iters)
            else:
                timing = _benchmark(run, args.warmup, args.iters)
            trace_path = None
            if args.graph_trace:
                trace_path = _capture_graph_trace(
                    run,
                    LOCAL_RESULTS / "traces"
                    / f"{args.tag}_{phase}_{mode}_b{size}_isl{args.isl}_w{world}_graph_rank{rank}.json",
                )
            elif args.trace:
                trace_path = _capture_trace(
                    run,
                    LOCAL_RESULTS / "traces"
                    / f"{args.tag}_{phase}_{mode}_b{size}_isl{args.isl}_w{world}_rank{rank}.json",
                )
            gathered = mpi_allgather(timing)
            row = {
                "phase": phase,
                "size": size,
                "isl": args.isl,
                "mode": mode,
                "global_tokens": global_tokens,
                "score_median_ms": max(t["median_ms"] for t in gathered),
                "per_rank": {f"rank{i}": t for i, t in enumerate(gathered)},
                "traces": mpi_allgather(trace_path) if (args.trace or args.graph_trace) else None,
            }
            rows.append(row)
            runtime.shutdown()
            mpi_barrier()
            if rank == 0:
                print(f"[block4] {phase} size={size} {mode}: "
                      f"{row['score_median_ms']:.3f} ms median", flush=True)
        if hasattr(routed, "destroy"):
            routed.destroy()
        del stack, moe_layer, routed
        torch.cuda.empty_cache()
        mpi_barrier()

    receipt = {
        "kind": "block4_e2e",
        "blocks": "KDA,KDA,KDA,MLA (K3 3:1 interleave), shared tp_baseline MoE layer",
        "labels": {
            "tp": "production KimiK3MoE, EP routed + fused finalize-AR",
            "tp_te": "production KimiK3MoE, TP-in-expert routed (TileRT layout), fused finalize-AR",
            "cp": "whole sequences per rank; MoE A2A is the only cross-rank comm",
            "cp_mm": "cp with MegaMoEDeepGemm routed path (comm fused in-kernel)",
        },
        "timing": {"warmup": args.warmup, "iters": args.iters,
                   "launch": "cuda_graph" if args.graph_timing else "eager"},
        "note": (
            "Synthetic attention weights (perf/comm fidelity); decode from "
            "zero-filled caches with num_cached=[isl] — shapes/traffic real, "
            "values synthetic (runtime.py)."
        ),
        "comm_verification": comm_verdicts,
        "rows": rows,
        "provenance": verify_provenance(),
    }
    if rank == 0:
        LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
        out = LOCAL_RESULTS / f"{args.tag}_{'_'.join(args.modes)}_w{world}_isl{args.isl}.json"
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[block4] receipt={out}", flush=True)


if __name__ == "__main__":
    main()
