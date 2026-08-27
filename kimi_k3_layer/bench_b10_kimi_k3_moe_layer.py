#!/usr/bin/env python3
"""TP8 CUDA-graph layer benchmark: reference vs new Kimi-K3 B10 layer."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DECODE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PREFILL_SIZES = (512, 1024, 2048, 4096, 8192, 16384)
GRAPH_TOKEN_BUDGET = 98_304


def _configure_nccl_graph_policy() -> None:
    """Use the serialized NCCL CUDA-graph policy for these benchmarks."""
    # NCCL 2.30+ supports stream ordering off only when graph mixing is off.
    # These benchmarks serialize graph replays and do not need mixed launches.
    os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
    os.environ["NCCL_GRAPH_STREAM_ORDERING"] = "0"


def init_weights(moe, world: int, rank: int, seed: int = 0) -> None:
    """Initialize deterministic global weights, then select this rank's shard."""
    from kimi_k3_layer.config import (
        HIDDEN,
        MOE_INTER,
        MOE_LATENT,
        NUM_EXPERTS,
        SHARED_INTER,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, std=0.02):
        return (
            torch.randn(
                *shape,
                generator=gen,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * std
        )

    with torch.no_grad():
        moe.gate.weight.copy_(randn(NUM_EXPERTS, HIDDEN))
        moe.gate.e_score_correction_bias.copy_(
            randn(NUM_EXPERTS, std=0.02).to(
                moe.gate.e_score_correction_bias.dtype
            )
        )
        moe.fc1_latent_proj.weight.copy_(randn(MOE_LATENT, HIDDEN))
        moe.fc2_latent_proj.weight.copy_(randn(HIDDEN, MOE_LATENT))
        moe.latent_norm.weight.copy_(
            1 + randn(MOE_LATENT).to(moe.latent_norm.weight.dtype)
        )

        shared_inter_local = SHARED_INTER // world
        gate = randn(SHARED_INTER, HIDDEN)
        up = randn(SHARED_INTER, HIDDEN)
        down = randn(HIDDEN, SHARED_INTER)
        rows = slice(
            rank * shared_inter_local,
            (rank + 1) * shared_inter_local,
        )
        moe.shared_experts.gate_up_proj.weight.copy_(
            torch.cat([gate[rows], up[rows]])
        )
        moe.shared_experts.down_proj.weight.copy_(down[:, rows])

        backend = getattr(moe.experts, "backend", moe.experts)
        if type(backend).__name__ == "TRTLLMGenFusedMoE":
            for name, parameter in backend.named_parameters():
                full_shape = (NUM_EXPERTS, *parameter.shape[1:])
                experts_local = parameter.shape[0]
                experts = slice(
                    rank * experts_local,
                    (rank + 1) * experts_local,
                )
                if parameter.dtype == torch.uint8 and "scale" in name:
                    parameter.fill_(124)
                elif parameter.dtype == torch.uint8:
                    # Full-tensor staging on the HOST: at checkpoint width the
                    # int64 randint temp is 73.5 GiB per parameter, which
                    # OOMs any GPU that is not empty (and wastes HBM even
                    # when it is). Rank-identical values still hold -- the
                    # generator seed and draw order are unchanged, only the
                    # staging device moved.
                    import zlib

                    cpu_gen = torch.Generator().manual_seed(
                        int(gen.initial_seed())
                        ^ zlib.crc32(name.encode()))
                    full = torch.randint(
                        0,
                        256,
                        full_shape,
                        generator=cpu_gen,
                        dtype=torch.uint8,
                    )
                    parameter.copy_(full[experts].to(parameter.device))
                else:
                    parameter.zero_()
            if hasattr(backend, "post_load_weights"):
                backend.post_load_weights()
        else:
            w31_full = randn(NUM_EXPERTS, 2 * MOE_INTER, MOE_LATENT)
            w2_full = randn(NUM_EXPERTS, MOE_LATENT, MOE_INTER)
            w31_local = backend.w3_w1_weight.data
            w2_local = backend.w2_weight.data
            experts_local = w31_local.shape[0]
            experts = slice(
                rank * experts_local,
                (rank + 1) * experts_local,
            )
            w31_local.copy_(w31_full[experts])
            w2_local.copy_(w2_full[experts])
    torch.cuda.synchronize()


def _sizes(spec: str) -> tuple[int, ...]:
    if spec == "decode":
        return DECODE_SIZES
    if spec == "prefill":
        return PREFILL_SIZES
    if spec == "all":
        return DECODE_SIZES + PREFILL_SIZES
    return tuple(int(item) for item in spec.split(","))


def _graph_iters(tokens: int, requested: int) -> int:
    return max(5, min(requested, GRAPH_TOKEN_BUDGET // tokens))


def _enum_or_bool(field: str, value: str):
    from kimi_k3_layer.b10_kimi_k3_moe_layer import (
        DecodeFront,
        DecodeTail,
        ExpertBackend,
        PrefillFC1,
        PrefillTail,
        Routing,
    )

    enums = {
        "decode_front": DecodeFront,
        "decode_tail": DecodeTail,
        "routing": Routing,
        "prefill_fc1": PrefillFC1,
        "prefill_tail": PrefillTail,
        "prefill_expert_backend": ExpertBackend,
    }
    if field in enums:
        return enums[field](value)
    if field in {
        "route_on_side_stream",
        "prefill_overlap_shared_branch",
    }:
        if value.lower() not in {"0", "1", "false", "true"}:
            raise ValueError(f"{field} expects true/false")
        return value.lower() in {"1", "true"}
    raise ValueError(f"unknown sweep axis {field!r}")


def _sweep(spec: str):
    """``AXIS=V[,V...]`` groups, ``;``-separated, so ONE run reverts every
    field against the same reference."""
    if not spec:
        return ()
    groups = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "&" in part:
            combo = {}
            for one in part.split("&"):
                axis, sep, value = one.partition("=")
                if not sep:
                    raise ValueError("--sweep expects AXIS=VALUE")
                combo[axis] = _enum_or_bool(axis, value)
            groups.append((None, combo))
            continue
        axis, sep, values = part.partition("=")
        if not sep:
            raise ValueError("--sweep expects AXIS=VALUE[,VALUE...]")
        groups.append((axis, tuple(_enum_or_bool(axis, value)
                                   for value in values.split(","))))
    return tuple(groups)


def _capture(fn, iterations: int, world: int, *, graphpatch: bool = False,
             rank: int = 0):
    """``fn(index)`` is captured once per iteration with its own input
    buffer, so in-graph iterations see DIFFERENT routing (a fixed input
    freezes one expert-load draw and with it one permanent straggler
    rank — see local_debug/arskew_allranks.py).

    graphpatch=True additionally swaps every precomputed-ids
    routingIndicesClusterKernel node for the single-CTA small-batch kernel
    (kimi_k3_layer.kernels.routing_graphpatch; bitwise-validated in
    local_debug/routing_graphpatch_probe.py, +8-12 us/iter at B<=64).
    Score-path routing nodes are left untouched, so baseline-equivalent
    configurations stay exact."""
    import torch.distributed as dist
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    with with_multi_stream(True):
        with autotune():
            fn(0)
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        for index in range(5):
            fn(index % iterations)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True) if graphpatch \
            else torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for index in range(iterations):
                fn(index)
    torch.cuda.synchronize()
    if graphpatch:
        from kimi_k3_layer.kernels.routing_graphpatch import patch_graph

        from kimi_k3_layer.config import NUM_EXPERTS as _num_experts

        patched = patch_graph(
            graph, num_local_experts=_num_experts // max(world, 1))
        graph.instantiate()
        if rank == 0 and patched:
            print(f"  [graphpatch] {patched}/{iterations} routing-indices "
                  "nodes swapped", flush=True)
        elif rank == 0:
            # A silent no-op cost ~7 us/layer once (trt-k3, 2026-08-22);
            # fingerprint the candidates so it can never hide again.
            from kimi_k3_layer.kernels.routing_graphpatch import scan_graph
            nodes = scan_graph(graph)
            print(f"  [graphpatch] 0 nodes swapped — {len(nodes)} cluster "
                  f"nodes found; first: "
                  f"{ {k: v for k, v in nodes[0].items()} if nodes else None}",
                  flush=True)
    if world > 1:
        dist.barrier()
    return graph


def _time_graph(graph, iterations: int, world: int, *,
                inputs: list[torch.Tensor], n_inputs: int,
                seed: int) -> float:
    """Mean over deterministic input sets; each sample is max-over-ranks.
    All per-iteration buffers are refilled per sample (rank-identical:
    seeded generator)."""
    import statistics
    import torch.distributed as dist

    originals = [buffer.clone() for buffer in inputs]
    samples = []
    for index in range(n_inputs):
        generator = torch.Generator(device=inputs[0].device).manual_seed(
            97 + 131 * seed + index)
        for buffer in inputs:
            buffer.copy_(torch.randn(
                buffer.shape,
                generator=generator,
                device=buffer.device,
                dtype=torch.float32,
            ).to(buffer.dtype))
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        latency = start.elapsed_time(stop) * 1000 / iterations
        if world > 1:
            value = torch.tensor(
                [latency], device="cuda", dtype=torch.float64)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            latency = value.item()
        samples.append(latency)

    for buffer, original in zip(inputs, originals):
        buffer.copy_(original)
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    return statistics.mean(samples)


def _profile_graph(graph, path: Path, rank: int, world: int) -> None:
    import torch.distributed as dist

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        graph.replay()
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(path))
        print(f"[profile] wrote {path}", flush=True)
    if world > 1:
        dist.barrier()


def _tie_aware_error(layer, hidden, output, reference):
    from kimi_k3_layer.config import TOP_K

    reference_scale = reference.float().abs().max().clamp_min(1e-9)
    token_error = (
        output.float() - reference.float()).abs().amax(-1) / reference_scale
    raw = token_error.max().item()
    with torch.no_grad():
        scores = torch.sigmoid(layer.gate(hidden).float())
        scores = scores + layer.gate.e_score_correction_bias.float()
        top = torch.topk(scores, TOP_K + 1, dim=1).values
        near_tie = (top[:, TOP_K - 1] - top[:, TOP_K]) < 1e-5
        fp32_ids = torch.sort(torch.topk(scores, TOP_K, dim=1).indices, 1)[0]
        bf16_ids = torch.sort(torch.topk(
            scores.to(torch.bfloat16).float(), TOP_K, dim=1).indices, 1)[0]
        near_tie |= (fp32_ids != bf16_ids).any(1)
    clean = token_error[~near_tie]
    return raw, (clean.max().item() if clean.numel() else 0.0), int(
        near_tie.sum())


def _build_collectives(world: int, sizes, max_decode: int):
    import torch.distributed as dist
    from communication.collective import Collectives
    from kimi_k3_layer.config import HIDDEN, MOE_LATENT

    if world == 1:
        return Collectives(None, 0)
    max_tokens = max(sizes)
    packed = HIDDEN + MOE_LATENT
    collectives = Collectives(
        dist.group.WORLD,
        max_numel=max_tokens * packed,
        max_hidden=packed,
        col_ag_max_rows=max_decode,
        col_ag_max_output_columns=MOE_LATENT,
        fused_ar_max_rows=max_decode,
    )
    # Persist the tuned (op, dim, bucket) -> impl map across processes:
    # with K3_AUTO_MAP_JSON set, the first process on this node profiles
    # and exports; later ones import and skip straight to measuring.
    # Keyed by world size — a TP4 map must never serve a TP8 run. Any
    # cell the imported map misses is still lazily tuned (ensure_tuned).
    map_base = os.environ.get("K3_AUTO_MAP_JSON", "")
    map_path = f"{map_base}.w{world}.json" if map_base else ""
    if map_path and os.path.exists(map_path):
        import json
        with open(map_path) as f:
            collectives.import_auto_map(json.load(f))
        if dist.get_rank() == 0:
            print(f"[collectives.autotune] imported map from {map_path}",
                  flush=True)
        return collectives
    collectives.autotune(
        dims=(HIDDEN, MOE_LATENT, packed),
        max_tokens=min(max_tokens, 256),
        ops=("all_reduce", "all_gather", "allreduce_norm"),
        warmup=2,
        iters=10,
        log=(dist.get_rank() == 0),
    )
    if max_tokens > 256:
        collectives.autotune(
            dims=(HIDDEN, MOE_LATENT),
            max_tokens=max_tokens,
            ops=("all_reduce",),
            warmup=1,
            iters=5,
            skip=("flashinfer", "torch_symm:1shot"),
            clear=False,
            log=(dist.get_rank() == 0),
        )
    if map_path and dist.get_rank() == 0:
        import json
        with open(map_path, "w") as f:
            json.dump(collectives.export_auto_map(), f, indent=1)
        print(f"[collectives.autotune] exported map to {map_path}", flush=True)
    return collectives


def main() -> None:
    _configure_nccl_graph_policy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="all",
                        help="decode, prefill, all, or comma-separated tokens")
    parser.add_argument("--mode", choices=("exp", "deploy"), default="deploy")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--n-inputs", type=int, default=8,
        help="average graph timing over deterministic routing inputs")
    parser.add_argument(
        "--sweep", default="",
        help="EXP only: one AXIS=VALUE[,VALUE...] sweep from measured config")
    parser.add_argument("--include-all-off", action="store_true",
                        help="EXP: graph the exact reference path via all-off")
    parser.add_argument(
        "--profile",
        default="",
        help="comma-separated sizes whose baseline/new graph replays are traced",
    )
    parser.add_argument(
        "--graphpatch", choices=("on", "off"), default="on",
        help="swap the routing-indices cluster kernel for the single-CTA "
             "small-batch kernel in captured decode graphs (opt path only)")
    parser.add_argument("--csv-suffix", default="")
    args = parser.parse_args()
    if args.n_inputs < 1:
        parser.error("--n-inputs must be positive")

    sizes = tuple(sorted(set(_sizes(args.sizes))))
    profile_sizes = set(_sizes(args.profile)) if args.profile else set()
    unknown_profile_sizes = profile_sizes.difference(sizes)
    if unknown_profile_sizes:
        parser.error(
            f"--profile sizes must also appear in --sizes: "
            f"{sorted(unknown_profile_sizes)}"
        )
    sweeps = _sweep(args.sweep)
    if args.mode == "deploy" and (sweeps or args.include_all_off):
        parser.error("DEPLOY has no sweeps; use --mode exp")

    import torch.distributed as dist
    from tensorrt_llm._torch.utils import AuxStreamType

    from kimi_k3_layer.b10_kimi_k3_moe_layer import (
        B10KimiK3MoELayer,
        DECODE_MAX_TOKENS,
        ExperimentConfig,
        KimiK3MoEReference,
        LayerMode,
        k3_model_config,
        measured_config,
    )
    from kimi_k3_layer.config import HIDDEN, MOE_LATENT

    rank = int(os.environ.get(
        "RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world = int(os.environ.get(
        "WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    # multi-node: the device index is the LOCAL rank
    torch.cuda.set_device(int(os.environ.get(
        "OMPI_COMM_WORLD_LOCAL_RANK",
        os.environ.get("LOCAL_RANK", rank))) % torch.cuda.device_count())
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29521")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world,
            device_id=torch.device("cuda", torch.cuda.current_device()),
        )

    # 128-row floor: prefill-only size lists must still construct the
    # col-AG engine (Collectives rejects col_ag_max_rows=0).
    max_decode = max(
        (size for size in sizes if size <= DECODE_MAX_TOKENS),
        default=0) or 128
    collectives = _build_collectives(world, sizes, max_decode)
    config = k3_model_config(rank, world)
    aux = {kind: torch.cuda.Stream() for kind in AuxStreamType}
    # K3_NEW_CLASS selects what the "new" label times: the full B10 layer
    # (default), pure stock (sanity: new == baseline), or the stock-plus
    # ladder (K3_FRONT_STAGES picks the pre-expert stages).
    from kimi_k3_layer.b10_kimi_k3_moe_layer import KimiK3StockPlusFront
    _layer_cls = {
        "stock": KimiK3MoEReference,
        "stockplus": KimiK3StockPlusFront,
    }.get(os.environ.get("K3_NEW_CLASS", ""), B10KimiK3MoELayer)
    layer = _layer_cls(
        config,
        layer_idx=0,
        aux_stream_dict=aux,
        reduce_output=world > 1,
        collectives=collectives,
        mode=LayerMode(args.mode),
    ).cuda()
    init_weights(layer, world, rank)
    if isinstance(layer, B10KimiK3MoELayer):
        layer.init_optimized(max_batch=max_decode, collectives=collectives)

    rows = []
    for tokens in sizes:
        iterations = _graph_iters(tokens, args.iters)
        generator = torch.Generator(device="cuda").manual_seed(tokens)
        # one input buffer per in-graph iteration (rank-identical data)
        inputs = [
            torch.randn(
                tokens, HIDDEN, generator=generator, device="cuda",
                dtype=torch.float32,
            ).to(torch.bfloat16)
            for _ in range(iterations)
        ]

        reference_box = {}

        # Serving-aligned serialization: in production, layer k+1's input
        # depends on layer k's output, so iterations never overlap. The
        # bench's independent buffers let iteration k+1's side-stream work
        # race into iteration k's lamport AR and inflate kernel times.
        # K3_CHAIN_ITERS=1 (default) restores the dependency with a
        # zero-weight residual: exact same values, real data edge.
        _chain = os.environ.get("K3_CHAIN_ITERS", "1") == "1"

        def _chained_input(index, box):
            h = inputs[index]
            prev = box.get("output")
            if _chain and prev is not None and prev.shape == h.shape:
                return torch.add(h, prev.view(h.shape), alpha=0.0)
            return h

        # K3_BENCH_SKEW_NS: rank-dependent busy-wait before each layer
        # call, emulating serving's per-rank attention-time variance —
        # the skew source single-layer benches otherwise lack (a path
        # with 3 sync points pays skew up to 3x/layer, 1 sync pays 1x).
        _skew_ns = int(os.environ.get("K3_BENCH_SKEW_NS", "0")) * rank

        def _inject_skew():
            if _skew_ns:
                from communication.kernels.delay import delay_ns
                delay_ns(_skew_ns)

        def reference_fn(index):
            _inject_skew()
            with torch.no_grad():
                reference_box["output"] = KimiK3MoEReference.forward(
                    layer, _chained_input(index, reference_box))

        reference_graph = _capture(reference_fn, iterations, world)
        reference_us = _time_graph(
            reference_graph,
            iterations,
            world,
            inputs=inputs,
            n_inputs=args.n_inputs,
            seed=tokens,
        )
        reference = reference_box["output"].clone()
        if tokens in profile_sizes:
            _profile_graph(
                reference_graph,
                Path(__file__).parent / "local_results"
                / f"moe_tp{world}_b{tokens}_baseline_graph.trace.json",
                rank,
                world,
            )
        del reference_graph

        variants = [("new", None)]
        if args.include_all_off:
            variants.append(("all_off", ExperimentConfig.all_off()))
        if sweeps:
            base = measured_config(tokens)
            for axis, sweep_values in sweeps:
                if axis is None:      # combined: apply every axis at once
                    label = "+".join(
                        f"{k}={v.value if hasattr(v, 'value') else v}"
                        for k, v in sweep_values.items())
                    variants.append((label, replace(base, **sweep_values)))
                    continue
                variants.extend((
                    f"{axis}={value.value if hasattr(value, 'value') else value}",
                    replace(base, **{axis: value}),
                ) for value in sweep_values)

        if rank == 0:
            print(f"B={tokens:>5} baseline={reference_us:8.2f} us",
                  flush=True)
            rows.append({
                "tokens": tokens,
                "mode": args.mode,
                "config": "baseline",
                "latency_us": reference_us,
                "speedup_pct": 0.0,
                "rel_error": 0.0,
                "tie_excluded_error": 0.0,
                "tie_rows": 0,
                "max_abs_diff": 0.0,
                "rel_max_diff": 0.0,
                "rows_off": 0,
                "rows_total": 0,
            })

        for label, exp_config in variants:
            if args.mode == "exp":
                layer.set_experiment_config(exp_config)
            output_box = {}

            def layer_fn(index):
                _inject_skew()
                with torch.no_grad():
                    output_box["output"] = layer(
                        _chained_input(index, output_box))

            active_config = exp_config or measured_config(tokens)
            uses_column_gather = (
                tokens <= DECODE_MAX_TOKENS
                or active_config.prefill_fc1.value == "sharded")
            if (collectives.has_col_ag and uses_column_gather
                    and tokens <= max_decode):
                collectives.tune_col_ag(torch.empty(
                    tokens, MOE_LATENT // world,
                    device="cuda", dtype=torch.bfloat16))
            graph = _capture(
                layer_fn, iterations, world,
                # the single-CTA routing-indices patch is validated
                # through B=128 only — NOT tied to DECODE_MAX_TOKENS
                graphpatch=(args.graphpatch == "on" and tokens <= 128),
                rank=rank,
            )
            latency = _time_graph(
                graph,
                iterations,
                world,
                inputs=inputs,
                n_inputs=args.n_inputs,
                seed=tokens,
            )
            output = output_box["output"].clone()
            if tokens in profile_sizes and label == "new":
                _profile_graph(
                    graph,
                    Path(__file__).parent / "local_results"
                    / f"moe_tp{world}_b{tokens}_opt_graph.trace.json",
                    rank,
                    world,
                )
            del graph
            if rank == 0:
                # output_box/reference_box hold the LAST captured
                # iteration's result, i.e. inputs[-1] after the restore
                # replay in _time_graph.
                raw, clean, ties = _tie_aware_error(
                    layer, inputs[-1], output, reference)
                d = (output.float() - reference.float()).abs()
                d = d.reshape(-1, d.shape[-1])
                max_abs = d.max().item()
                ref_absmax = reference.float().abs().max().item()
                row_max = d.amax(-1)
                rows_off = int((row_max > 1e-3 * ref_absmax).sum())
                speedup = (reference_us - latency) / reference_us * 100
                which = ""
                if 0 < rows_off <= 12:
                    idx = torch.nonzero(
                        row_max > 1e-3 * ref_absmax).flatten().tolist()
                    which = f" rows={idx}"
                print(f"  {label:<52} {latency:8.2f} us "
                      f"{speedup:+6.1f}% maxdiff={max_abs:.3e} "
                      f"rel={max_abs / max(ref_absmax, 1e-9):.2e} "
                      f"rows_off={rows_off}/{d.shape[0]}{which}", flush=True)
                rows.append({
                    "tokens": tokens,
                    "mode": args.mode,
                    "config": label,
                    "latency_us": latency,
                    "speedup_pct": speedup,
                    "rel_error": raw,
                    "tie_excluded_error": clean,
                    "tie_rows": ties,
                    "max_abs_diff": max_abs,
                    "rel_max_diff": max_abs / max(ref_absmax, 1e-9),
                    "rows_off": rows_off,
                    "rows_total": d.shape[0],
                })
        if world > 1:
            dist.barrier()

    if rank == 0:
        output = Path(__file__).parent / "local_results" / (
            f"bench_moe_layer_tp{world}{args.csv_suffix}.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Merge by token size: partial runs update their rows and keep
        # every other size's rows, so a decode-only refresh no longer
        # clobbers the prefill half of the official table.
        benched = {row["tokens"] for row in rows}
        if output.exists():
            with output.open(newline="") as handle:
                kept = [row for row in csv.DictReader(handle)
                        if int(row["tokens"]) not in benched]
            rows = sorted(
                rows + kept, key=lambda row: int(row["tokens"]))
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {output}")

    collectives.destroy()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
