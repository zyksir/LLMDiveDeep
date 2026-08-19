#!/usr/bin/env python3
"""Unified Kimi-K3 KDA decode and prefill benchmark.

Requested token counts at or below ``DECODE_MAX_TOKENS`` exercise batched
decode. Larger counts exercise the one-sequence prefill path. The default
``reference`` and ``b10`` rows use automatic strategy/backend selection;
``--backends`` exposes concrete implementations for verification and ablation.

Decode correctness preserves the original output, conv-state, SSM-state, and
AttnRes comparisons against ``trt_fused``. Prefill preserves the original
TRT-chunk-versus-B10 output cosine comparison.
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

_LLMDIR = Path(__file__).resolve().parents[1]


def bootstrap() -> None:
    import tensorrt_llm  # noqa: F401

    if "kimi_k3_layer" not in sys.modules:
        package = types.ModuleType("kimi_k3_layer")
        package.__path__ = [str(_LLMDIR / "kimi_k3_layer")]
        sys.modules["kimi_k3_layer"] = package
    if str(_LLMDIR) not in sys.path:
        sys.path.insert(0, str(_LLMDIR))


def build_layer(
    implementation: str,
    tokens: int,
    shard,
    *,
    layer_idx: int,
    seed: int,
):
    from kimi_k3_layer.b10_kimi_k3_kda_layer import (
        B10_BACKENDS,
        KimiK3KDA,
        KimiK3KDAB10,
    )

    if implementation in ("reference", "b10"):
        cls = KimiK3KDAB10 if implementation == "b10" else KimiK3KDA
        backend = None
    else:
        cls = KimiK3KDAB10 if implementation in B10_BACKENDS else KimiK3KDA
        backend = implementation
    torch.manual_seed(seed)
    return cls(shard, tokens, backend, layer_idx=layer_idx).eval()


def make_prefill_input(tokens: int, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    hidden = (
        torch.randn(
            tokens,
            hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    ).to(torch.bfloat16)
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    return hidden, cu_seqlens


def run_layer(
    layer,
    hidden: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.inference_mode():
        return layer(hidden, cu_seqlens)


def _run_decode_once(backend: str, shard, batch: int, layer_idx: int):
    layer = build_layer(
        backend, batch, shard, layer_idx=layer_idx, seed=2026
    )
    output = run_layer(layer)
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


def check_decode_correctness(shard, layer_idx: int) -> None:
    from kimi_k3_layer.b10_kimi_k3_kda_layer import B10_BACKENDS, TRT_BACKENDS

    backends = tuple(
        backend
        for backend in (*TRT_BACKENDS, *B10_BACKENDS)
        if not backend.endswith("_prefill")
    )
    batch = 4
    reference = _run_decode_once("trt_fused", shard, batch, layer_idx)
    print(
        f"\nDecode correctness: shard={shard.name}, B={batch}, "
        f"layer={layer_idx} (vs trt_fused)"
    )
    print(
        f"{'backend':>14} {'pass':>5} {'max_abs':>10} {'cosine':>9} "
        f"{'conv':>5} {'ssm_abs':>10} {'attn_res':>8}"
    )
    for backend in backends:
        result = (
            reference
            if backend == "trt_fused"
            else _run_decode_once(backend, shard, batch, layer_idx)
        )
        max_abs = (
            result["output"].float() - reference["output"].float()
        ).abs().max().item()
        cosine = F.cosine_similarity(
            result["output"].float().flatten(),
            reference["output"].float().flatten(),
            dim=0,
        ).item()
        output_ok = torch.allclose(
            result["output"].float(),
            reference["output"].float(),
            atol=2e-2,
            rtol=2e-2,
        )
        conv_ok = torch.allclose(
            result["conv_state"].float(),
            reference["conv_state"].float(),
            atol=2e-2,
            rtol=2e-2,
        )
        ssm_abs = (
            result["ssm_state"] - reference["ssm_state"]
        ).abs().max().item()
        attn_res_ok = torch.equal(
            result["prefix_sum"], reference["prefix_sum"]
        ) and torch.equal(
            result["block_residual"], reference["block_residual"]
        )
        passed = output_ok and conv_ok and ssm_abs <= 2e-2 and attn_res_ok
        print(
            f"{backend:>14} {str(passed):>5} {max_abs:10.4g} "
            f"{cosine:9.6f} {str(conv_ok):>5} {ssm_abs:10.4g} "
            f"{str(attn_res_ok):>8}"
        )
        if not passed:
            raise AssertionError(
                f"{backend} failed {shard.name} decode correctness"
            )


def check_prefill_correctness(shard, tokens: int, layer_idx: int) -> None:
    hidden, cu_seqlens = make_prefill_input(tokens, shard.hidden)
    outputs = {}
    for implementation in ("reference", "b10"):
        layer = build_layer(
            implementation,
            tokens,
            shard,
            layer_idx=layer_idx,
            seed=2026,
        )
        outputs[implementation] = run_layer(
            layer, hidden, cu_seqlens
        ).detach().clone()
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    cosine = F.cosine_similarity(
        outputs["reference"].float().flatten(),
        outputs["b10"].float().flatten(),
        dim=0,
    ).item()
    max_abs = (
        outputs["reference"].float() - outputs["b10"].float()
    ).abs().max().item()
    if not torch.isfinite(outputs["reference"]).all():
        raise AssertionError("reference prefill produced non-finite output")
    if not torch.isfinite(outputs["b10"]).all():
        raise AssertionError("b10 prefill produced non-finite output")
    print(
        f"\nPrefill correctness: shard={shard.name}, S={tokens}, "
        f"reference-vs-b10 cosine={cosine:.6f}, max_abs={max_abs:.4g}"
    )


def time_eager(run, warmup: int, iters: int, repeats: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iters)
    return statistics.median(samples)


def time_graph(run, warmup: int, iters: int, repeats: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iters)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", nargs="+", choices=["tp1", "tp8"], default=["tp8"]
    )
    parser.add_argument(
        "--token-sizes",
        "--batch-sizes",
        dest="token_sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 128, 4096, 8192, 16384],
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=["reference", "b10"],
        default=["reference", "b10"],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="explicit backend override for verification/ablation",
    )
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--prefill-graph",
        action="store_true",
        help="also graph-time the B10 prefill path; reference has host syncs",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent
        / "local_result"
        / "bench_b10_kimi_k3_kda_layer.csv",
    )
    args = parser.parse_args()

    bootstrap()
    from common.kernel_bench import plot_rows
    from kimi_k3_layer.config import k3_shard
    from kimi_k3_layer.b10_kimi_k3_kda_layer import (
        ALL_BACKENDS,
        DECODE_MAX_TOKENS,
    )

    implementations = args.backends or args.implementations
    unknown = set(implementations) - {"reference", "b10", *ALL_BACKENDS}
    if unknown:
        parser.error(f"unknown backend(s): {sorted(unknown)}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"Automatic dispatch: decode <= {DECODE_MAX_TOKENS} tokens; "
        "prefill above it"
    )

    if not args.skip_correctness:
        prefill_sizes = [
            tokens
            for tokens in args.token_sizes
            if tokens > DECODE_MAX_TOKENS
        ]
        for shard_name in args.shards:
            shard = k3_shard(shard_name)
            check_decode_correctness(shard, args.layer_idx)
            if prefill_sizes:
                check_prefill_correctness(
                    shard, prefill_sizes[0], args.layer_idx
                )

    rows = []
    for shard_name in args.shards:
        shard = k3_shard(shard_name)
        for tokens in args.token_sizes:
            hidden_and_cu = (
                make_prefill_input(tokens, shard.hidden)
                if tokens > DECODE_MAX_TOKENS
                else (None, None)
            )
            for implementation in implementations:
                gc.collect()
                torch.cuda.empty_cache()
                layer = build_layer(
                    implementation,
                    tokens,
                    shard,
                    layer_idx=args.layer_idx,
                    seed=1000 + tokens,
                )
                hidden, cu_seqlens = hidden_and_cu

                def run():
                    return run_layer(layer, hidden, cu_seqlens)

                timing = "graph" if layer.strategy == "decode" else "eager"
                timer = time_graph if timing == "graph" else time_eager
                latency = timer(run, args.warmup, args.iters, args.repeats)
                row = {
                    "shard": shard.name,
                    "tp_size": shard.tp_size,
                    "tokens": tokens,
                    "strategy": layer.strategy,
                    "implementation": implementation,
                    "backend": layer.backend,
                    "timing": timing,
                    "layer_idx": args.layer_idx,
                    "latency_us": latency,
                    "tokens_s": tokens * 1e6 / latency,
                }
                rows.append(row)
                print(
                    f"{shard.name:>4} T={tokens:<5} {layer.strategy:>7} "
                    f"{implementation:>14} ({layer.backend}): "
                    f"{latency:9.2f} us {row['tokens_s']:10.0f} tok/s"
                )
                if (
                    args.prefill_graph
                    and layer.strategy == "prefill"
                    and implementation in ("b10", "b10_prefill")
                ):
                    graph_latency = time_graph(
                        run, args.warmup, args.iters, args.repeats
                    )
                    graph_row = dict(row)
                    graph_row["timing"] = "graph"
                    graph_row["latency_us"] = graph_latency
                    graph_row["tokens_s"] = tokens * 1e6 / graph_latency
                    rows.append(graph_row)
                del run, layer

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure = plot_rows(
        rows,
        args.csv.with_suffix(".png"),
        x="tokens",
        y="latency_us",
        series="implementation",
        panel="shard",
        suptitle="Kimi-K3 KDA: automatic decode/prefill dispatch",
    )
    print(f"Wrote {args.csv}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
