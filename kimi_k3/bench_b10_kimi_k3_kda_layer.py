#!/usr/bin/env python3
"""Unified Kimi-K3 KDA decode and prefill benchmark.

Requested token counts at or below ``DECODE_MAX_TOKENS`` exercise batched
decode. Larger counts exercise the one-sequence prefill path. The default
``reference`` and ``b10`` rows use automatic strategy/backend selection;
``--backends`` exposes concrete implementations for verification and ablation.

Decode correctness preserves the original output, conv-state, SSM-state, and
AttnRes comparisons against ``trt_fused``. Prefill preserves the original
TRT-chunk-versus-B10 output cosine comparison.

``--backends trt_block sgl_block`` benchmarks the trace-aligned MTP1 verify
blocks (``TrtKimiK3KdaBlock`` / ``SglKimiK3KdaBlock``). ``--trace-out
<path>`` is the single trace switch: it profiles the selected blocks'
CUDA-graph replays into that gzipped chrome trace (instead of the latency
bench) whose ``TrtKimiK3Kda`` / ``SglKimiK3Kda`` spans diff against the
serving baselines in ``traces/``; run under torchrun with WORLD_SIZE=8 for
the genuine TP8 AR tails.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

_LLMDIR = Path(__file__).resolve().parents[1]
# Anchors the pre(imports+dist) phase of the trace/turnaround timing print.
_PROCESS_T0 = time.perf_counter()

# Trace-alignment blocks: every profiled replay of an MTP1 block — and only
# those — is wrapped in torch.profiler.record_function with the block's
# RECORD_SPAN (plus an [M=...] suffix when several token counts are emitted)
# so mini traces diff against the serving baselines by span name (mirrors
# _BLOCK_SPANS in bench_b10_kimi_k3_moe_layer.py).
MTP_BLOCK_IMPLS = ("trt_block", "sgl_block")


def bootstrap() -> None:
    if "kimi_k3" not in sys.modules:
        package = types.ModuleType("kimi_k3")
        package.__path__ = [str(_LLMDIR / "kimi_k3")]
        sys.modules["kimi_k3"] = package
    # The repo root must PRECEDE any PYTHONPATH entry: containers that put
    # e.g. /workspace/trtllm (which has its own top-level ``common``) on
    # PYTHONPATH would otherwise shadow this repo's ``common`` package.
    if str(_LLMDIR) in sys.path:
        sys.path.remove(str(_LLMDIR))
    sys.path.insert(0, str(_LLMDIR))

    # Pin every JIT/compile cache (applies on import) BEFORE the heavy
    # imports read their env: flashinfer resolves FLASHINFER_WORKSPACE_BASE
    # at import time (tensorrt_llm can pull it in), and the CuTe DSL folds
    # its env knobs into the compile-cache hash.
    import kimi_k3.kernels.jit_cache_env  # noqa: F401

    import tensorrt_llm  # noqa: F401


def build_layer(
    implementation: str,
    tokens: int,
    shard,
    *,
    layer_idx: int,
    seed: int,
):
    from kimi_k3.b10_kimi_k3_kda_layer import (
        B10_BACKENDS,
        KimiK3KDA,
        KimiK3KDAB10,
        SglKimiK3KdaBlock,
        TrtKimiK3KdaBlock,
    )

    if implementation in ("reference", "b10"):
        cls = KimiK3KDAB10 if implementation == "b10" else KimiK3KDA
        backend = None
    elif implementation in MTP_BLOCK_IMPLS:
        cls = (
            TrtKimiK3KdaBlock
            if implementation == "trt_block"
            else SglKimiK3KdaBlock
        )
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
    from kimi_k3.b10_kimi_k3_kda_layer import B10_BACKENDS, TRT_BACKENDS

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


def check_mtp_block_cross(shard, tokens: int, layer_idx: int) -> None:
    """Cross-check the two MTP1 blocks where their math is equivalent.

    Same seed -> identical weights and AttnRes buffers (identical
    construction order). The TRT AttnRes kernel folds the pending o_proj
    ``delta`` into the prefix in-kernel while the sglang stack pre-adds it
    (production folds it into the push-AR), so the SGL block's prefix is set
    to ``trt.prefix_sum + trt.delta`` first; both blocks then aggregate the
    same effective prefix, project with the same weights, and run the same
    fused verify kernel on zeroed states. The comparison unfolds the SGL
    block's residual (its forward returns prefix + reduced o_proj output)
    and compares against the TRT block's o_proj output. Only the o_proj GEMM
    (TGV vs cuBLAS) and the AttnRes kernels differ numerically.
    """
    trt = build_layer("trt_block", tokens, shard, layer_idx=layer_idx, seed=7)
    sgl = build_layer("sgl_block", tokens, shard, layer_idx=layer_idx, seed=7)
    with torch.no_grad():
        sgl.prefix_sum.copy_(trt.prefix_sum + trt.delta)
        for block in (trt, sgl):
            block.ssm_state.zero_()
            block.conv_state.zero_()
        out_trt = trt().detach().float()
        out_sgl = (sgl() - sgl.prefix_sum).detach().float()
    torch.cuda.synchronize()
    cosine = F.cosine_similarity(
        out_trt.flatten(), out_sgl.flatten(), dim=0
    ).item()
    max_abs = (out_trt - out_sgl).abs().max().item()
    print(
        f"\nMTP1 block cross-check: shard={shard.name}, M={tokens}, "
        f"layer={layer_idx}, trt-vs-sgl cosine={cosine:.6f}, "
        f"max_abs={max_abs:.4g}"
    )
    if not torch.isfinite(out_trt).all() or not torch.isfinite(out_sgl).all():
        raise AssertionError("MTP1 block produced non-finite output")
    if cosine < 0.98:
        raise AssertionError(
            f"TRT/SGL MTP1 blocks diverged (cosine={cosine:.6f})"
        )
    del trt, sgl
    gc.collect()
    torch.cuda.empty_cache()


def _emit_trace(args, world: int, rank: int) -> None:
    """One torch-profiler session over the MTP1 KDA blocks, exported as a
    gzipped chrome trace.

    Methodology (matches the serving baselines, which are CUDA-graph decode
    captures, and the MoE bench's --emit-trace): eager warmups first
    (tvm-ffi/nvcc JIT, CuTe DSL compiles, cuBLAS heuristics), ONE forward
    per block captured into its own CUDA graph, several replay warmups, and
    only then the profiler window over graph REPLAYS, each wrapped in the
    block's record_function span. Every profiled replay runs on DIFFERENT
    input data: a pre-generated pool of distinct prefix/delta/conv/SSM-state
    batches is copied into the blocks' captured static buffers before each
    replay, outside the record_function spans.

    Span names are the blocks' base RECORD_SPANs; when more than one token
    count is emitted, each span carries an ``[M=<tokens>]`` suffix so the
    trace stays diffable per shape (e.g. ``TrtKimiK3Kda[M=4]``).

    At world > 1 (one process per GPU, RANK/WORLD_SIZE set) the blocks run
    their genuine TP tails: the TRT block reduces through ``Collectives.
    all_reduce(impl="trt")`` (TRT-LLM oneshot-Lamport pattern 0, the
    baseline's kernel) and the SGL block through the vendored
    ``all_reduce_push_res`` on the CustomAllReduceV2 state it resolves
    lazily via ``kernels.sgl_adapters.comm.get_sgl_ar_state`` (the
    baseline's fused push AR + residual fold). Inputs are
    rank-identical (seeded); rank 0 exports the trace.

    With ``--trace-out none`` (multi-rank turnaround/smoke mode) the same
    loop runs without the profiler and nothing is exported; only the phase
    timing line is printed.
    """
    import gzip
    import tempfile

    import torch.distributed as dist

    from kimi_k3.b10_kimi_k3_kda_layer import MTP_WIDTH
    from kimi_k3.config import k3_shard

    setup_start = time.perf_counter()
    pre_s = setup_start - _PROCESS_T0
    token_counts = list(dict.fromkeys(args.emit_tokens))
    multi_m = len(token_counts) > 1
    shard = k3_shard(args.shards[0])
    requested = args.backends or args.implementations
    implementations = [
        implementation
        for implementation in requested
        if implementation in MTP_BLOCK_IMPLS
    ]
    if not implementations:
        raise SystemExit(
            "--trace-out needs at least one MTP1 block selected via "
            f"--backends {list(MTP_BLOCK_IMPLS)}"
        )

    def _sync_barrier() -> None:
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()

    if not args.skip_correctness and len(implementations) == 2:
        if world == 1:
            check_mtp_block_cross(shard, token_counts[0], args.layer_idx)
        elif rank == 0:
            print(
                "[emit-trace] skipping the single-GPU cross-check at "
                f"world={world} (covered by the world=1 run)",
                flush=True,
            )

    def span_label(block) -> str:
        base = block.RECORD_SPAN
        return f"{base}[M={block.batch}]" if multi_m else base

    blocks = [
        build_layer(
            implementation,
            tokens,
            shard,
            layer_idx=args.layer_idx,
            seed=2026,
        )
        for tokens in token_counts
        for implementation in implementations
    ]

    # Genuine TP tails at world > 1 (see docstring); nothing at world = 1.
    collectives = None
    if world > 1:
        from communication.collective import Collectives

        # The TRT block's tail is an explicit impl="trt" request, so the
        # heavier autotune candidates (flashinfer workspace) are skipped.
        collectives = Collectives(
            dist.group.WORLD,
            max_numel=max(token_counts) * shard.hidden,
            max_hidden=shard.hidden,
            enable_flashinfer=False,
        )
        for block in blocks:
            block.attach_collectives(collectives)
        # The SGL block stands up sglang's CustomAllReduceV2 workspaces
        # itself on first decode (kernels/sgl_adapters/comm.py,
        # get_sgl_ar_state — sglang's k3_ar_fusion._get_state pattern):
        # genuine fused push-AR tail when available, the documented
        # fallback AR tail otherwise. The first decode runs eagerly on
        # every rank in lockstep, before any CUDA graph capture, so the
        # collective build is safe.

    # Distinct inputs per profiled replay: prefix/delta (AttnRes stream),
    # recurrent state and conv window pools regenerated per iteration.
    # Conv pool is generated in the TRT [slot, channel, window] layout and
    # transposed into the SGL block's [slot, window, channel] storage.
    heads, dim = shard.heads_local, shard.head_dim

    def draw_batch(tokens: int, index: int) -> dict:
        requests = tokens // MTP_WIDTH
        generator = torch.Generator(device="cuda").manual_seed(
            10_000 + 100 * tokens + index
        )

        def draw(*shape, scale, dtype):
            return (
                torch.randn(
                    *shape,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float32,
                )
                * scale
            ).to(dtype)

        return {
            "prefix": draw(
                tokens, shard.hidden, scale=0.02, dtype=torch.bfloat16
            ),
            "delta": draw(
                tokens, shard.hidden, scale=0.02, dtype=torch.bfloat16
            ),
            "ssm": draw(
                requests, heads, dim, dim, scale=0.05, dtype=torch.float32
            ),
            "conv": draw(
                requests,
                shard.qkv_dim,
                shard.conv_size - 1,
                scale=0.1,
                dtype=torch.bfloat16,
            ),
        }

    input_pools = {
        tokens: [
            draw_batch(tokens, index)
            for index in range(args.emit_trace_iters)
        ]
        for tokens in token_counts
    }

    def load_inputs(block, batch) -> None:
        block.prefix_sum.copy_(batch["prefix"])
        if hasattr(block, "delta"):
            block.delta.copy_(batch["delta"])
        block.ssm_state.copy_(batch["ssm"])
        if block.conv_state.shape == batch["conv"].shape:
            block.conv_state.copy_(batch["conv"])
        else:  # SGL [slot, window, channel] storage
            block.conv_state.copy_(batch["conv"].transpose(-1, -2))

    def run_all_eager() -> None:
        for block in blocks:
            with torch.no_grad():
                block()

    # Eager warmup: JIT/CuTe compiles, cuBLAS heuristics, workspaces (the
    # TRT AR workspace builds collectively on the first reduce; the loop
    # order is rank-identical, so in-graph collectives capture in lockstep).
    run_all_eager()
    _sync_barrier()
    for _ in range(5):
        run_all_eager()
    _sync_barrier()

    graphs = []
    for block in blocks:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad():
                block()
        graphs.append((block, graph, span_label(block)))
        _sync_barrier()
    for _ in range(3):  # replay warmup, outside the profiler window
        for _, graph, _ in graphs:
            graph.replay()
    _sync_barrier()
    setup_s = time.perf_counter() - setup_start

    emit = args.trace_out.lower() != "none"
    profiled_start = time.perf_counter()
    profiler_ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        if emit
        else contextlib.nullcontext()
    )
    with profiler_ctx as profiler:
        for index in range(args.emit_trace_iters):
            for block, graph, label in graphs:
                # Buffer refresh stays outside the span.
                load_inputs(block, input_pools[block.batch][index])
                with torch.profiler.record_function(label):
                    graph.replay()
        _sync_barrier()
    profiled_s = time.perf_counter() - profiled_start

    export_start = time.perf_counter()
    if emit and rank == 0:
        out = Path(args.trace_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            profiler.export_chrome_trace(tmp.name)
            with open(tmp.name, "rb") as src, gzip.open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    if world > 1:
        dist.barrier()
    export_s = time.perf_counter() - export_start
    if rank == 0:
        if emit:
            print(
                f"[emit-trace] wrote {Path(args.trace_out)} "
                f"(spans={[label for _, _, label in graphs]}, "
                f"M={token_counts}, world={world}, "
                f"replays={args.emit_trace_iters}, "
                f"distinct inputs per replay)"
            )
        print(
            f"[emit-trace] wall: pre(imports+dist)={pre_s:.1f}s "
            f"setup(build+warmup+capture)={setup_s:.1f}s "
            f"{'profiled' if emit else 'replayed(no profiler)'}"
            f"={profiled_s:.1f}s export+gzip={export_s:.1f}s",
            flush=True,
        )
    if collectives is not None:
        collectives.destroy()


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
        choices=["reference", "b10", *MTP_BLOCK_IMPLS],
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
    parser.add_argument(
        "--trace-out",
        default="none",
        help="the single trace switch: a path means 'profile MTP1 block "
        "CUDA-graph replays (selected via --backends trt_block sgl_block) "
        "into this gzipped chrome trace instead of running the latency "
        "bench'; 'none' (default) disables trace emission",
    )
    parser.add_argument(
        "--emit-trace-iters",
        type=int,
        default=24,
        help="profiled graph replays per block (each on distinct inputs)",
    )
    parser.add_argument(
        "--emit-tokens",
        type=int,
        nargs="+",
        default=[2],
        help="total tokens M for the trace (MTP1: 2 per request); with "
        "more than one count, span names gain an [M=...] suffix",
    )
    args = parser.parse_args()
    emit_trace = args.trace_out.lower() != "none"

    bootstrap()
    import torch.distributed as dist

    from common.kernel_bench import plot_rows
    from kimi_k3.config import k3_shard
    from kimi_k3.b10_kimi_k3_kda_layer import (
        ALL_BACKENDS,
        DECODE_MAX_TOKENS,
        MTP_WIDTH,
    )

    # One process per GPU when launched under torchrun/mpirun; the TP tails
    # in the emitted trace then run their genuine AR kernels.
    rank = int(
        os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    )
    world = int(
        os.environ.get(
            "WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        )
    )
    torch.cuda.set_device(
        int(
            os.environ.get(
                "OMPI_COMM_WORLD_LOCAL_RANK",
                os.environ.get("LOCAL_RANK", rank),
            )
        )
        % torch.cuda.device_count()
    )
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29531")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world,
            device_id=torch.device("cuda", torch.cuda.current_device()),
        )

    # Multi-rank runs always take the MTP1 block loop: with a trace path it
    # profiles and exports; with --trace-out none it replays the same loop
    # without the profiler (turnaround / smoke mode).
    if emit_trace or world > 1:
        _emit_trace(args, world, rank)
        if world > 1:
            dist.destroy_process_group()
        return

    implementations = args.backends or args.implementations
    unknown = set(implementations) - {
        "reference",
        "b10",
        *ALL_BACKENDS,
        *MTP_BLOCK_IMPLS,
    }
    if unknown:
        parser.error(f"unknown backend(s): {sorted(unknown)}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"Automatic dispatch: decode <= {DECODE_MAX_TOKENS} tokens; "
        "prefill above it"
    )

    def mtp_block_supported(tokens: int) -> bool:
        return tokens % MTP_WIDTH == 0 and MTP_WIDTH <= tokens <= 8

    if not args.skip_correctness and all(
        implementation in implementations for implementation in MTP_BLOCK_IMPLS
    ):
        for shard_name in args.shards:
            check_mtp_block_cross(
                k3_shard(shard_name), MTP_WIDTH, args.layer_idx
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
                if implementation in MTP_BLOCK_IMPLS and not (
                    mtp_block_supported(tokens)
                ):
                    print(
                        f"{shard.name:>4} T={tokens:<5} {implementation:>14}: "
                        f"skipped (MTP1 blocks cover even M in "
                        f"[{MTP_WIDTH}, 8])"
                    )
                    continue
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
