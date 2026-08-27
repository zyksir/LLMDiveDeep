#!/usr/bin/env python3
"""Coordinate-descent strategy search for the kimi_k3 B10 EXP layer.

Run as:
    mpirun -n 8 --allow-run-as-root \
        python3 kimi_k3/search_moe_strategies.py --sizes 16 --iters 20
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

from kimi_k3_layer.bench_b10_kimi_k3_moe_layer import (
    GRAPH_TOKEN_BUDGET,
    _build_collectives,
    _capture,
    _sizes,
    _tie_aware_error,
    _time_graph,
    init_weights,
)
from kimi_k3_layer.b10_kimi_k3_moe_layer import (
    DECODE_MAX_TOKENS,
    DecodeFront,
    DecodeTail,
    ExperimentConfig,
    ExpertBackend,
    LayerMode,
    PrefillFC1,
    PrefillTail,
    Routing,
    k3_model_config,
    measured_config,
)

DECODE_AXES: list[tuple[str, list]] = [
    ("decode_front", list(DecodeFront)),
    ("routing", list(Routing)),
    ("decode_tail", list(DecodeTail)),
]
PREFILL_AXES: list[tuple[str, list]] = [
    ("routing", list(Routing)),
    ("prefill_fc1", list(PrefillFC1)),
    ("prefill_tail", list(PrefillTail)),
    ("prefill_expert_backend", list(ExpertBackend)),
    ("prefill_overlap_shared_branch", [False, True]),
]

ERR_THRESHOLD = 0.02

# Search start for the prefill "baseline fallback" zone (512–2048 tokens).
_PREFILL_SEARCH_START = ExperimentConfig(
    decode_front=DecodeFront.SEPARATE_FULL_FC1,
    routing=Routing.RADIX,
    prefill_fc1=PrefillFC1.FULL,
    prefill_expert_backend=ExpertBackend.FLASHINFER,
    prefill_overlap_shared_branch=True,
)


def _graph_iters(tokens: int, requested: int) -> int:
    return max(5, min(requested, GRAPH_TOKEN_BUDGET // tokens))


def _cfg_label(cfg: ExperimentConfig) -> str:
    if not cfg.enabled:
        return "all_off"
    return "|".join([
        cfg.decode_front.value, cfg.routing.value, cfg.decode_tail.value,
        cfg.prefill_fc1.value, cfg.prefill_expert_backend.value,
        f"posb={int(cfg.prefill_overlap_shared_branch)}",
    ])


def _val_str(val) -> str:
    return val.value if hasattr(val, "value") else str(val)


def _tune_col_ag(cfg: ExperimentConfig, tokens: int, collectives, world: int,
                 max_decode: int) -> None:
    from kimi_k3_layer.config import MOE_LATENT
    if not getattr(collectives, "has_col_ag", False) or tokens > max_decode:
        return
    uses = (
        not cfg.enabled
        or tokens <= DECODE_MAX_TOKENS
        or (cfg.enabled and cfg.prefill_fc1 is PrefillFC1.SHARDED)
    )
    if uses:
        collectives.tune_col_ag(torch.empty(
            tokens, MOE_LATENT // world, device="cuda", dtype=torch.bfloat16))


def _run_candidate(
    layer, inputs: list[torch.Tensor], cfg: ExperimentConfig,
    collectives, world: int, rank: int,
    iterations: int, n_inputs: int, max_decode: int, seed: int,
    reference: torch.Tensor, baseline_us: float,
    rnd: int, axis: str, val_label: str,
    rows: list[dict],
) -> tuple[float | None, bool]:
    import torch.distributed as dist

    tokens = inputs[0].shape[0]
    layer.set_experiment_config(cfg)
    out_box: dict = {}

    # One input buffer per in-graph iteration, exactly like the bench: a
    # fixed input would freeze one expert-load draw and with it one
    # permanent straggler rank (see bench_b10_kimi_k3_moe_layer._capture).
    def fn(index):
        with torch.no_grad():
            out_box["output"] = layer(inputs[index])

    _tune_col_ag(cfg, tokens, collectives, world, max_decode)

    try:
        graph = _capture(fn, iterations, world)
    except Exception as exc:
        if rank == 0:
            print(f"    [{rnd}] {axis}={val_label} SKIP ({exc})", flush=True)
            rows.append({
                "tokens": tokens, "round": rnd, "axis": axis,
                "value": val_label, "config_label": _cfg_label(cfg),
                "latency_us": None, "speedup_pct": None,
                "rel_error": None, "tie_excluded_error": None,
                "tie_rows": None, "valid": False, "accepted": False,
            })
        return None, False

    latency = _time_graph(
        graph, iterations, world, inputs=inputs, n_inputs=n_inputs, seed=seed)
    output = out_box["output"].clone()
    del graph

    valid = False
    if rank == 0:
        # output/reference hold the LAST captured iteration's result, i.e.
        # inputs[-1] after _time_graph's restore replay (bench convention).
        raw, clean, ties = _tie_aware_error(
            layer, inputs[-1], output, reference)
        valid = clean <= ERR_THRESHOLD
        speedup = (baseline_us - latency) / baseline_us * 100
        print(f"    [{rnd}] {axis}={val_label:<32} {latency:8.2f}us "
              f"{speedup:+6.1f}%  err={raw:.2e} clean={clean:.2e} "
              f"ties={ties}  {'OK' if valid else 'INVALID'}", flush=True)
        rows.append({
            "tokens": tokens, "round": rnd, "axis": axis, "value": val_label,
            "config_label": _cfg_label(cfg), "latency_us": latency,
            "speedup_pct": speedup, "rel_error": raw,
            "tie_excluded_error": clean, "tie_rows": ties,
            "valid": valid, "accepted": False,
        })

    info = torch.tensor([float(valid), latency], device="cuda", dtype=torch.float64)
    if world > 1:
        dist.broadcast(info, src=0)
    return info[1].item(), bool(info[0].item() > 0.5)


def _search_size(
    layer, inputs: list[torch.Tensor], collectives, world: int, rank: int,
    tokens: int, baseline_us: float, reference: torch.Tensor,
    iterations: int, n_inputs: int, max_decode: int, n_rounds: int,
) -> tuple[ExperimentConfig, float, list[dict]]:
    is_decode = tokens <= DECODE_MAX_TOKENS
    axes = DECODE_AXES if is_decode else PREFILL_AXES
    seed = tokens
    rows: list[dict] = []

    def _eval(cfg, rnd, axis, val_label):
        return _run_candidate(
            layer, inputs, cfg, collectives, world, rank,
            iterations, n_inputs, max_decode, seed,
            reference, baseline_us, rnd, axis, val_label, rows)

    # All-off is identical to the reference class; reuse its timing.
    if rank == 0:
        rows.append({
            "tokens": tokens, "round": 0, "axis": "init", "value": "all_off",
            "config_label": "all_off", "latency_us": baseline_us,
            "speedup_pct": 0.0, "rel_error": 0.0, "tie_excluded_error": 0.0,
            "tie_rows": 0, "valid": True, "accepted": False,
        })

    # Measure the search-start config (measured_config if enabled, else sensible default).
    m = measured_config(tokens)
    search_start = m if m.enabled else (
        ExperimentConfig() if is_decode else _PREFILL_SEARCH_START)

    if rank == 0:
        print(f"B={tokens:>5}: measuring search-start ...", flush=True)
    start_us, start_valid = _eval(search_start, 0, "init", "search_start")

    best_enabled_us: float = start_us if start_valid else float("inf")
    best_enabled_cfg: ExperimentConfig | None = search_start if start_valid else None
    current = search_start

    for rnd in range(1, n_rounds + 1):
        if rank == 0:
            label = _cfg_label(best_enabled_cfg) if best_enabled_cfg else "none-valid-yet"
            print(f"B={tokens:>5}: round={rnd} current-best={label}", flush=True)
        changed = False

        for axis, values in axes:
            # Skip axes whose flag has no effect for the current path.

            axis_best_us = best_enabled_us
            axis_best_cfg = best_enabled_cfg

            for val in values:
                # Fused-shared-gate front is only valid for small batches.
                if (axis == "decode_front"
                        and val is DecodeFront.FUSED_FC1_SHARED_GATE_CUTE
                        and tokens > DECODE_MAX_TOKENS):
                    continue

                candidate = replace(current, enabled=True, **{axis: val})
                if candidate == current:
                    continue  # already measured; skip redundant capture

                lat, valid = _eval(candidate, rnd, axis, _val_str(val))
                if lat is None or not valid:
                    continue
                if lat < axis_best_us:
                    axis_best_us = lat
                    axis_best_cfg = candidate

            if axis_best_cfg is not None and axis_best_cfg != best_enabled_cfg:
                best_enabled_us = axis_best_us
                best_enabled_cfg = axis_best_cfg
                current = axis_best_cfg
                changed = True

        if not changed:
            if rank == 0:
                print(f"B={tokens:>5}: stable at round {rnd}", flush=True)
            break

    # Overall winner: best-enabled vs all-off (always valid at baseline cost).
    candidates: list[tuple[float, ExperimentConfig]] = [
        (baseline_us, ExperimentConfig.all_off())]
    if best_enabled_cfg is not None:
        candidates.append((best_enabled_us, best_enabled_cfg))
    final_us, final_cfg = min(candidates, key=lambda x: x[0])

    if rank == 0:
        label = _cfg_label(final_cfg)
        for row in reversed(rows):
            if row["config_label"] == label and not row["accepted"]:
                row["accepted"] = True
                break

    return final_cfg, final_us, rows


def _attribute_size(
    layer, inputs: list[torch.Tensor], collectives, world: int, rank: int,
    tokens: int, baseline_us: float, reference: torch.Tensor,
    iterations: int, n_inputs: int, max_decode: int,
) -> tuple[ExperimentConfig, float, list[dict]]:
    axes = DECODE_AXES if tokens <= DECODE_MAX_TOKENS else PREFILL_AXES
    config = measured_config(tokens)
    if not config.enabled:
        config = (
            ExperimentConfig()
            if tokens <= DECODE_MAX_TOKENS
            else _PREFILL_SEARCH_START
        )
    rows: list[dict] = []

    full_us, full_valid = _run_candidate(
        layer, inputs, config, collectives, world, rank,
        iterations, n_inputs, max_decode, tokens,
        reference, baseline_us, 0, "full_config", "measured", rows,
    )
    if full_us is None or not full_valid:
        raise RuntimeError(f"B={tokens}: measured configuration is invalid")

    for axis, values in axes:
        for value in values:
            if (
                axis == "decode_front"
                and value is DecodeFront.FUSED_FC1_SHARED_GATE_CUTE
                and tokens > DECODE_MAX_TOKENS
            ):
                continue
            candidate = replace(config, enabled=True, **{axis: value})
            if candidate == config:
                if rank == 0:
                    rows.append({
                        "tokens": tokens,
                        "round": 0,
                        "axis": axis,
                        "value": _val_str(value),
                        "config_label": _cfg_label(candidate),
                        "latency_us": full_us,
                        "speedup_pct":
                            (baseline_us - full_us) / baseline_us * 100,
                        "rel_error": 0.0,
                        "tie_excluded_error": 0.0,
                        "tie_rows": 0,
                        "valid": True,
                        "accepted": True,
                    })
                continue
            _run_candidate(
                layer, inputs, candidate, collectives, world, rank,
                iterations, n_inputs, max_decode, tokens,
                reference, baseline_us, 0, axis, _val_str(value), rows,
            )

    if rank == 0:
        for row in rows:
            latency = row["latency_us"]
            row["delta_vs_full_us"] = (
                None if latency is None else latency - full_us
            )
            row["delta_vs_full_pct"] = (
                None if latency is None
                else (latency - full_us) / full_us * 100
            )
    return config, full_us, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coordinate-descent strategy search for kimi_k3 B10 EXP layer")
    parser.add_argument("--sizes", default="decode",
                        help="decode, prefill, all, or comma-separated token counts")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--n-inputs", type=int, default=8,
                        help="average graph timing over this many deterministic inputs")
    parser.add_argument("--rounds", type=int, default=2,
                        help="max coordinate-descent rounds per token size")
    parser.add_argument(
        "--attribution",
        action="store_true",
        help="hold the measured config fixed and sweep every strategy axis",
    )
    parser.add_argument("--csv-suffix", default="",
                        help="appended to output CSV file names")
    parser.add_argument("--csv-out", default="",
                        help="explicit output directory (default: kimi_k3/local_results/)")
    args = parser.parse_args()

    import torch.distributed as dist
    from tensorrt_llm._torch.utils import AuxStreamType
    from kimi_k3_layer.b10_kimi_k3_moe_layer import (
        B10KimiK3MoELayer,
        KimiK3MoEReference,
    )
    from kimi_k3_layer.config import HIDDEN

    sizes = tuple(sorted(set(_sizes(args.sizes))))
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world = int(os.environ.get(
        "WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29522")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl", rank=rank, world_size=world,
            device_id=torch.device("cuda", rank))

    max_decode = max((s for s in sizes if s <= DECODE_MAX_TOKENS), default=0)
    collectives = _build_collectives(world, sizes, max_decode)
    config = k3_model_config(rank, world)
    aux = {kind: torch.cuda.Stream() for kind in AuxStreamType}
    layer = B10KimiK3MoELayer(
        config, layer_idx=0, aux_stream_dict=aux,
        reduce_output=(world > 1), collectives=collectives,
        mode=LayerMode.EXP,
    ).cuda()
    init_weights(layer, world, rank)
    layer.init_optimized(max_batch=max_decode, collectives=collectives)

    all_detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for tokens in sizes:
        iterations = _graph_iters(tokens, args.iters)
        generator = torch.Generator(device="cuda").manual_seed(tokens)
        # one input buffer per in-graph iteration (rank-identical data),
        # exactly as the bench builds them
        inputs = [
            torch.randn(
                tokens, HIDDEN, generator=generator, device="cuda",
                dtype=torch.float32,
            ).to(torch.bfloat16)
            for _ in range(iterations)
        ]

        ref_box: dict = {}

        def ref_fn(index):
            with torch.no_grad():
                ref_box["output"] = KimiK3MoEReference.forward(
                    layer, inputs[index])

        if rank == 0:
            print(f"\n{'='*60}", flush=True)
            print(f"B={tokens:>5}: capturing baseline ...", flush=True)
        ref_graph = _capture(ref_fn, iterations, world)
        baseline_us = _time_graph(
            ref_graph, iterations, world,
            inputs=inputs, n_inputs=args.n_inputs, seed=tokens)
        reference = ref_box["output"].clone()
        del ref_graph
        if rank == 0:
            print(f"B={tokens:>5}: baseline={baseline_us:.2f}us  "
                  f"(all-off reused as-is, zero error)", flush=True)

        if args.attribution:
            best_cfg, best_us, rows = _attribute_size(
                layer, inputs, collectives, world, rank, tokens,
                baseline_us, reference, iterations, args.n_inputs, max_decode)
        else:
            best_cfg, best_us, rows = _search_size(
                layer, inputs, collectives, world, rank, tokens,
                baseline_us, reference, iterations, args.n_inputs,
                max_decode, args.rounds)
        all_detail_rows.extend(rows)

        if rank == 0:
            speedup = (baseline_us - best_us) / baseline_us * 100
            result_label = "MEASURED" if args.attribution else "WINNER"
            print(
                f"\nB={tokens:>5}: {result_label}  {_cfg_label(best_cfg)}",
                flush=True,
            )
            print(f"          {best_us:.2f}us  {speedup:+.1f}% vs baseline", flush=True)
            summary_rows.append({
                "tokens": tokens,
                "best_config": _cfg_label(best_cfg),
                "best_latency_us": best_us,
                "baseline_us": baseline_us,
                "speedup_pct": speedup,
            })

        if world > 1:
            dist.barrier()

    if rank == 0:
        out_dir = (Path(args.csv_out) if args.csv_out
                   else Path(__file__).parent / "local_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_tp{world}{args.csv_suffix}"

        if all_detail_rows:
            stem = "moe_strategy_attribution" if args.attribution \
                else "search_moe_detail"
            detail_path = out_dir / f"{stem}{suffix}.csv"
            with detail_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_detail_rows[0]))
                writer.writeheader()
                writer.writerows(all_detail_rows)
            print(f"\nWrote {detail_path}")

        if summary_rows:
            summary_stem = (
                "moe_strategy_attribution_summary"
                if args.attribution else "search_moe_summary"
            )
            summary_path = out_dir / f"{summary_stem}{suffix}.csv"
            with summary_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"Wrote {summary_path}")

    collectives.destroy()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
