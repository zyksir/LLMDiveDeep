#!/usr/bin/env python3
"""Kimi-K3 KDA decode bench: TRT-LLM baseline vs b10 CuTeDSL.

Runs inside the trt-dev container (installed tensorrt_llm 1.3.0rc23,
CuTeDSL 4.x), single GPU - collectives are out of scope for the KDA
layer (see kda_trtllm_kimi_k3.py):

  docker exec trt-dev python3 \
      /workspace/diffusion_inference/LLMDiveDeep/kimi_k3_layer/bench_kda_kimi_k3.py

  # torch-profiler traces of every backend at one batch
  ... bench_kda_kimi_k3.py --profile 8

Backends (2 baselines + 2 ours = both fusion ablations):
  trt_fused / trt_unfused   (kda_trtllm_kimi_k3.py)
  b10_fused / b10_overlap   (kda_b10_kimi_k3.py)

Correctness: output, conv state, SSM state and AttnRes buffers of every
backend against trt_fused on seed-identical weights.
Timing: CUDA-graph replay (common.kernel_bench), CSV + PNG.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

_LLMDIR = Path(__file__).resolve().parents[1]


def bootstrap() -> None:
    import tensorrt_llm  # noqa: F401  (real runtime, first!)

    if "kimi_k3_layer" not in sys.modules:
        pkg = types.ModuleType("kimi_k3_layer")
        pkg.__path__ = [str(_LLMDIR / "kimi_k3_layer")]
        sys.modules["kimi_k3_layer"] = pkg
    if str(_LLMDIR) not in sys.path:
        sys.path.insert(0, str(_LLMDIR))


def build_layer(backend: str, batch: int, shard, *, layer_idx: int,
                seed: int):
    from kimi_k3_layer.kda_b10_kimi_k3 import B10_BACKENDS, KimiK3KDAB10
    from kimi_k3_layer.kda_trtllm_kimi_k3 import KimiK3KDA

    cls = KimiK3KDAB10 if backend in B10_BACKENDS else KimiK3KDA
    torch.manual_seed(seed)
    return cls(shard, batch, backend, layer_idx=layer_idx).eval()


def _run_once(backend: str, shard, batch: int, layer_idx: int):
    layer = build_layer(backend, batch, shard,
                        layer_idx=layer_idx, seed=2026)
    with torch.inference_mode():
        output = layer()
    torch.cuda.synchronize()
    result = {
        "output": output.detach().clone(),
        "conv_state": layer.conv_state.detach().clone(),
        "ssm_state": layer.comparable_ssm_state().detach().clone(),
        "prefix_sum": layer.prefix_sum.detach().clone(),
        "block_residual": layer.block_residual.detach().clone(),
    }
    del layer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def check_correctness(backends, shard, layer_idx: int) -> None:
    batch = 4
    reference = _run_once("trt_fused", shard, batch, layer_idx)
    print(f"\nCorrectness: shard={shard.name}, B={batch}, "
          f"layer={layer_idx} (vs trt_fused)")
    print(f"{'backend':>14} {'pass':>5} {'max_abs':>10} {'cosine':>9} "
          f"{'conv':>5} {'ssm_abs':>10} {'attn_res':>8}")
    for backend in backends:
        result = (reference if backend == "trt_fused"
                  else _run_once(backend, shard, batch, layer_idx))
        max_abs = (result["output"].float()
                   - reference["output"].float()).abs().max().item()
        cosine = F.cosine_similarity(
            result["output"].float().flatten(),
            reference["output"].float().flatten(), dim=0).item()
        output_ok = torch.allclose(
            result["output"].float(), reference["output"].float(),
            atol=2e-2, rtol=2e-2)
        conv_ok = torch.equal(result["conv_state"],
                              reference["conv_state"])
        ssm_abs = (result["ssm_state"]
                   - reference["ssm_state"]).abs().max().item()
        attn_res_ok = torch.equal(
            result["prefix_sum"], reference["prefix_sum"]
        ) and torch.equal(
            result["block_residual"], reference["block_residual"])
        passed = (output_ok and conv_ok and ssm_abs <= 2e-2
                  and attn_res_ok)
        print(f"{backend:>14} {str(passed):>5} {max_abs:10.4g} "
              f"{cosine:9.6f} {str(conv_ok):>5} {ssm_abs:10.4g} "
              f"{str(attn_res_ok):>8}")
        if not passed:
            raise AssertionError(
                f"{backend} failed {shard.name} correctness")


def profile_layer(layer, path: Path, backend: str) -> None:
    def run():
        with torch.inference_mode():
            layer()

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            run()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU],
    ) as prof:
        graph.replay()
        torch.cuda.synchronize()
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
    print(f"[profile] {backend}: wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", choices=["tp1", "tp8"],
                        default=["tp8", "tp1"])
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--backends", nargs="+", default=None)
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--profile", type=int, default=0, metavar="B",
        help="export torch-profiler traces of every backend at batch B "
        "(first shard only, results/)")
    parser.add_argument(
        "--csv", type=Path,
        default=Path(__file__).parent / "results"
        / "bench_kda_kimi_k3.csv")
    args = parser.parse_args()

    bootstrap()
    from common.kernel_bench import bench_cuda, plot_rows
    from kimi_k3_layer.config import k3_shard
    from kimi_k3_layer.kda_b10_kimi_k3 import B10_BACKENDS
    from kimi_k3_layer.kda_trtllm_kimi_k3 import TRT_BACKENDS

    backends = args.backends or (*TRT_BACKENDS, *B10_BACKENDS)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print("Scope: TRT AttnRes + exact KimiDeltaAttention decode "
          "projections, conv/KDA/norm, and o_proj; no collectives")
    if not args.skip_correctness:
        for shard_name in args.shards:
            check_correctness(backends, k3_shard(shard_name),
                              args.layer_idx)

    rows = []
    for shard_name in args.shards:
        shard = k3_shard(shard_name)
        for batch in args.batch_sizes:
            for backend in backends:
                gc.collect()
                torch.cuda.empty_cache()
                layer = build_layer(backend, batch, shard,
                                    layer_idx=args.layer_idx,
                                    seed=1000 + batch)

                def run():
                    with torch.inference_mode():
                        return layer()

                latency = bench_cuda(run, args.warmup, args.iters,
                                     args.repeats, use_graph=True)
                rows.append({
                    "shard": shard.name,
                    "tp_size": shard.tp_size,
                    "heads_local": shard.heads_local,
                    "batch": batch,
                    "backend": backend,
                    "layer_idx": args.layer_idx,
                    "latency_us": latency,
                    "tokens_s": batch * 1e6 / latency,
                })
                print(f"{shard.name:>4} B={batch:<3} {backend:>14}: "
                      f"{latency:9.2f} us "
                      f"{rows[-1]['tokens_s']:10.0f} tok/s")
                if args.profile == batch and shard_name == args.shards[0]:
                    profile_layer(
                        layer,
                        args.csv.parent / f"kda_kimi_k3_{shard.name}"
                        f"_b{batch}_{backend}.trace.json",
                        backend)
                del run, layer

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure = plot_rows(
        rows, args.csv.with_suffix(".png"),
        x="batch", y="latency_us", series="backend", panel="shard",
        suptitle="Kimi-K3 AttnRes + KDA decode: TRT-LLM vs b10",
    )
    print(f"Wrote {args.csv}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
