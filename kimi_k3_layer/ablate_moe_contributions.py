#!/usr/bin/env python3
"""Cumulative strategy-contribution waterfall for the B10 MoE layer.

Report-only. Starts from the exact reference forward, then adds ONE
strategy at a time until the config equals ``measured_config(tokens)``,
so each row's ``step_saved_us`` is the LATENCY SAVED by that strategy
(previous minus new latency; positive = faster) GIVEN the ones
before it (contributions interact; the ladder order is the natural
build order, and the final row is the shipped configuration):

  baseline                      reference forward (graph-timed)
  precomputed_route+packed_AR   opt skeleton: routing precomputed on
                                the host path (run_moe skips its
                                in-kernel scores stage), autotuned
                                packed [latent|shared] AR via
                                Collectives, column-slice norm,
                                zero-copy AR staging
  +fc1_shard(col_AG+quant)      per-rank fc1 column shard, rebuilt by
                                the fused column-AG + MXFP8 quantize
  +fused_front(dual_out)        ONE dual-output GEMM
                                [fc1-shard | shared g/u | gate]
  +radix_routing                Routing.RADIX (SGLang RouteRadixKernel)
  +route_side_stream            routing kernel off the main stream
  +measured_tail                the measured decode tail (fc2 shard
                                <=8 tokens, multimem full-fc2 >=16)
                                == measured_config(tokens)

Prefill sizes (tokens > 128) use their own ladder:
skeleton -> +overlap_shared_branch -> +radix_routing ->
[+fc1_shard(col_AG) -> +fused_AG_quantize when the plan shards] ->
[+native_expert_backend from 8192]. For 129..2048 the shipped plan is
the baseline (all_off); the ladder still runs toward the would-be opt
config to document why baseline ships there.

  mpirun --allow-run-as-root -np 8 python3 \\
      kimi_k3_layer/ablate_moe_contributions.py --sizes 1,8,32,128
"""

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

from kimi_k3_layer.bench_b10_kimi_k3_moe_layer import (  # noqa: E402
    _build_collectives,
    _capture,
    _graph_iters,
    _tie_aware_error,
    _time_graph,
    init_weights,
)


def ladder_for(tokens: int):
    from kimi_k3_layer.b10_kimi_k3_moe_layer import (
        DECODE_MAX_TOKENS,
        DecodeFront,
        DecodeTail,
        ExpertBackend,
        ExperimentConfig,
        PrefillFC1,
        Routing,
        measured_config,
    )

    if tokens <= DECODE_MAX_TOKENS:
        target = measured_config(tokens)
        cfg = ExperimentConfig(
            decode_front=DecodeFront.SEPARATE_FULL_FC1,
            routing=Routing.REFERENCE,
            decode_tail=DecodeTail.PACKED_LATENT_SHARED_REDUCE,
            route_on_side_stream=False,
        )
        steps = [("precomputed_route+packed_AR", cfg)]
        cfg = replace(cfg, decode_front=DecodeFront.SEPARATE_SHARDED_FC1)
        steps.append(("+fc1_shard(col_AG+quant)", cfg))
        cfg = replace(cfg, decode_front=target.decode_front)
        steps.append(("+fused_front(dual_out)", cfg))
        cfg = replace(cfg, routing=target.routing)
        steps.append(("+radix_routing", cfg))
        cfg = replace(cfg, route_on_side_stream=target.route_on_side_stream)
        steps.append(("+route_side_stream", cfg))
        cfg = replace(cfg, decode_tail=target.decode_tail)
        steps.append(("+measured_tail", cfg))
        assert cfg == target, (cfg, target)
        return steps

    # Prefill ladder. For 129..2048 the shipped plan is the baseline
    # (all_off, measured parity); the ladder still runs toward the
    # would-be opt config (prefill_baseline_max_tokens=128) so the table
    # DOCUMENTS why the plan ships baseline there.
    target = measured_config(tokens, prefill_baseline_max_tokens=128)
    # decode_* fields stay at their dataclass defaults: the prefill
    # paths never read them and measured_config leaves them alone too.
    cfg = ExperimentConfig(
        routing=Routing.REFERENCE,
        prefill_fc1=PrefillFC1.FULL,
        prefill_expert_backend=ExpertBackend.FLASHINFER,
        prefill_overlap_shared_branch=False,
    )
    steps = [("precomputed_route+packed_AR", cfg)]
    cfg = replace(cfg, prefill_overlap_shared_branch=True)
    steps.append(("+overlap_shared_branch", cfg))
    cfg = replace(cfg, routing=target.routing)
    steps.append(("+radix_routing", cfg))
    if target.prefill_tail is not cfg.prefill_tail:
        cfg = replace(cfg, prefill_tail=target.prefill_tail)
        steps.append(("+fc2_shard_tail", cfg))
    if target.prefill_fc1 is PrefillFC1.SHARDED:
        cfg = replace(cfg, prefill_fc1=PrefillFC1.SHARDED)
        steps.append(("+fc1_shard(mm_AG+fused_quant)", cfg))
    if target.prefill_expert_backend is ExpertBackend.NATIVE:
        cfg = replace(cfg, prefill_expert_backend=ExpertBackend.NATIVE)
        steps.append(("+native_expert_backend", cfg))
    assert cfg == target, (cfg, target)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1,8,32,128")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--n-inputs", type=int, default=8)
    parser.add_argument("--csv-suffix", default="")
    args = parser.parse_args()

    sizes = tuple(sorted({int(v) for v in args.sizes.split(",")}))

    import torch.distributed as dist
    from tensorrt_llm._torch.utils import AuxStreamType

    from kimi_k3_layer.b10_kimi_k3_moe_layer import (
        B10KimiK3MoELayer,
        DECODE_MAX_TOKENS,
        LayerMode,
        k3_model_config,
    )
    from kimi_k3_layer.config import HIDDEN, MOE_LATENT

    rank = int(os.environ.get(
        "RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world = int(os.environ.get(
        "WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29527")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl", rank=rank, world_size=world,
            device_id=torch.device("cuda", rank))

    max_decode = max(
        (s for s in sizes if s <= DECODE_MAX_TOKENS), default=0)
    # _build_collectives sizes the col-AG engine from the decode max;
    # keep a 128-row floor so prefill-only runs still construct.
    collectives = _build_collectives(world, sizes, max_decode or 128)
    aux = {kind: torch.cuda.Stream() for kind in AuxStreamType}
    layer = B10KimiK3MoELayer(
        k3_model_config(rank, world),
        layer_idx=0,
        aux_stream_dict=aux,
        reduce_output=world > 1,
        collectives=collectives,
        mode=LayerMode.EXP,
    ).cuda()
    init_weights(layer, world, rank)
    layer.init_optimized(max_batch=max_decode or 128,
                         collectives=collectives)

    rows = []
    for tokens in sizes:
        iterations = _graph_iters(tokens, args.iters)
        generator = torch.Generator(device="cuda").manual_seed(tokens)
        inputs = [
            torch.randn(tokens, HIDDEN, generator=generator, device="cuda",
                        dtype=torch.float32).to(torch.bfloat16)
            for _ in range(iterations)
        ]

        reference_box = {}

        def reference_fn(index):
            with torch.no_grad():
                reference_box["output"] = layer.baseline_forward(
                    inputs[index])

        graph = _capture(reference_fn, iterations, world)
        previous_us = reference_us = _time_graph(
            graph, iterations, world,
            inputs=inputs, n_inputs=args.n_inputs, seed=tokens)
        reference = reference_box["output"].clone()
        del graph
        if rank == 0:
            print(f"B={tokens:>4} baseline={reference_us:8.2f} us",
                  flush=True)
            rows.append({"tokens": tokens, "step": "baseline",
                         "latency_us": round(reference_us, 2),
                         "step_saved_us": 0.0, "cumulative_pct": 0.0,
                         "rel_error": 0.0, "tie_excluded_error": 0.0,
                         "tie_rows": 0})

        if collectives.has_col_ag and tokens <= max_decode:
            collectives.tune_col_ag(torch.empty(
                tokens, MOE_LATENT // world,
                device="cuda", dtype=torch.bfloat16))
        for label, cfg in ladder_for(tokens):
            layer.set_experiment_config(cfg)
            output_box = {}

            def layer_fn(index):
                with torch.no_grad():
                    output_box["output"] = layer(inputs[index])

            graph = _capture(layer_fn, iterations, world,
                             graphpatch=tokens <= 128, rank=rank)
            latency = _time_graph(
                graph, iterations, world,
                inputs=inputs, n_inputs=args.n_inputs, seed=tokens)
            output = output_box["output"].clone()
            del graph
            if rank == 0:
                raw, clean, ties = _tie_aware_error(
                    layer, inputs[-1], output, reference)
                delta = previous_us - latency
                cumulative = (reference_us - latency) / reference_us * 100
                print(f"  {label:<28} {latency:8.2f} us "
                      f"saved {delta:+7.2f} us  total {cumulative:+6.1f}% "
                      f"err={raw:.2e} clean={clean:.2e} ties={ties}",
                      flush=True)
                rows.append({
                    "tokens": tokens, "step": label,
                    "latency_us": round(latency, 2),
                    "step_saved_us": round(delta, 2),
                    "cumulative_pct": round(cumulative, 1),
                    "rel_error": raw, "tie_excluded_error": clean,
                    "tie_rows": ties,
                })
            previous_us = latency
        if world > 1:
            dist.barrier()

    if rank == 0:
        out = Path(__file__).parent / "local_results" / (
            f"ablate_moe_contributions_tp{world}{args.csv_suffix}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out}", flush=True)

    collectives.destroy()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
