#!/usr/bin/env python3
"""Matched GDN kernel benchmark across FLA, SGLang, vLLM, TRT-LLM, FlashQLA.

What this isolates: the GDN *core* kernels only — the one-token decode
recurrence and the chunk-prefill pipeline — with matched inputs (same Q/K/V,
gates, beta, state layout, sequence boundaries) across frameworks. No
projections, conv, norm, scheduler, CUDA graphs, or engine runtime.

KDA-specific benchmarks live in ``bench_kda_decode.py``,
``bench_kda_prefill.py``, and ``bench_kda_spec_verify.py``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # linear_attn/
sys.path.append(str(Path(__file__).resolve().parents[2]))  # LLMDiveDeep/ (common/)
from gdn.gdn_attention import (
    gdn_recurrent_reference,
    get_gdn_decode_backends,
    get_gdn_prefill_backends,
    make_gdn_inputs,
)
from linear_attention import Shape
from common.kernel_bench import bench_cuda, plot_rows


def _fresh(inputs):
    return replace(inputs, state=inputs.state.clone())


def check_correctness(shape: Shape) -> dict[str, dict[str, float]]:
    base = make_gdn_inputs(2, 3, shape, seed=7)
    ref_out, ref_state = gdn_recurrent_reference(base, shape)
    results = {}
    for name in (
        "fla_gdn_recurrent",
        "sglang_gdn_recurrent",
        "trtllm_gdn_recurrent",
    ):
        inputs = _fresh(base)
        runners, _ = get_gdn_decode_backends(inputs, shape)
        if name not in runners:
            continue
        out, state = runners[name]()
        results[name] = {
            "output_max_abs": (out.float() - ref_out.float()).abs().max().item(),
            "output_cosine": torch.nn.functional.cosine_similarity(
                out.float().flatten(), ref_out.float().flatten(), dim=0
            ).item(),
            "state_max_abs": (state.float() - ref_state.float()).abs().max().item(),
            "state_cosine": torch.nn.functional.cosine_similarity(
                state.float().flatten(), ref_state.float().flatten(), dim=0
            ).item(),
        }
    packed_inputs = make_gdn_inputs(2, 1, shape, seed=11)
    packed_ref_out, packed_ref_state = gdn_recurrent_reference(packed_inputs, shape)
    runners, _ = get_gdn_decode_backends(packed_inputs, shape)
    for name in ("sglang_gdn_packed", "vllm_gdn_recurrent"):
        if name not in runners:
            continue
        out, state = runners[name]()
        results[name] = {
            "output_max_abs": (
                out.float() - packed_ref_out.float()
            ).abs().max().item(),
            "output_cosine": torch.nn.functional.cosine_similarity(
                out.float().flatten(), packed_ref_out.float().flatten(), dim=0
            ).item(),
            "state_max_abs": (
                state.float() - packed_ref_state.float()
            ).abs().max().item(),
            "state_cosine": torch.nn.functional.cosine_similarity(
                state.float().flatten(), packed_ref_state.float().flatten(), dim=0
            ).item(),
        }
    prefill_base = make_gdn_inputs(2, 128, shape, seed=13)
    prefill_ref_out, prefill_ref_state = gdn_recurrent_reference(
        prefill_base, shape
    )
    for name in (
        "fla_gdn_chunk",
        "sglang_gdn_chunk",
        "vllm_gdn_chunk",
        "trtllm_gdn_chunk",
        "flash_qla_gdn_chunk",
    ):
        inputs = _fresh(prefill_base)
        runners, _ = get_gdn_prefill_backends(inputs, shape)
        if name not in runners:
            continue
        result = runners[name]()
        out = result[0].reshape_as(prefill_ref_out)
        state = inputs.state if name == "sglang_gdn_chunk" else result[1]
        results[name] = {
            "output_max_abs": (
                out.float() - prefill_ref_out.float()
            ).abs().max().item(),
            "output_cosine": torch.nn.functional.cosine_similarity(
                out.float().flatten(), prefill_ref_out.float().flatten(), dim=0
            ).item(),
            "state_max_abs": (
                state.float() - prefill_ref_state.float()
            ).abs().max().item(),
            "state_cosine": torch.nn.functional.cosine_similarity(
                state.float().flatten(), prefill_ref_state.float().flatten(), dim=0
            ).item(),
        }
    return results


def run_decode(args, shape: Shape) -> list[dict]:
    rows = []
    print("\nDECODE — matched GDN recurrence")
    print(f"{'backend':>28} {'B':>5} {'latency us':>12} {'tok/s':>12}")
    shown, failed = set(), set()
    for batch in args.batch_sizes:
        inputs = make_gdn_inputs(batch, 1, shape, seed=batch)
        runners, unavailable = get_gdn_decode_backends(inputs, shape)
        for name, reason in unavailable.items():
            if name not in shown:
                print(f"  unavailable {name}: {reason}")
                shown.add(name)
        for name, fn in runners.items():
            if name in failed or (args.backends and name not in args.backends):
                continue
            try:
                latency = bench_cuda(fn, args.warmup, args.iters, args.repeats)
            except Exception as exc:
                print(f"  failed {name}: {type(exc).__name__}: {exc}")
                failed.add(name)
                continue
            row = {
                "mode": "decode",
                "backend": name,
                "batch": batch,
                "seq_len": 1,
                "latency_us": latency,
                "tokens_s": batch * 1e6 / latency,
                "qk_heads": shape.qk_heads,
                "value_heads": shape.value_heads,
                "key_dim": shape.key_dim,
                "value_dim": shape.value_dim,
                "state_dtype": shape.state_dtype,
            }
            rows.append(row)
            print(
                f"{name:>28} {batch:5d} {latency:12.2f} "
                f"{row['tokens_s']:12.0f}"
            )
    return rows


def run_prefill(args, shape: Shape) -> list[dict]:
    rows = []
    print("\nPREFILL — matched GDN chunk kernels")
    print(
        f"{'backend':>28} {'B':>5} {'S':>8} {'latency ms':>12} {'tok/s':>12}"
    )
    shown, failed = set(), set()
    for batch in args.prefill_batch_sizes:
        for seq_len in args.prefill_seq_lens:
            inputs = make_gdn_inputs(batch, seq_len, shape, seed=batch + seq_len)
            runners, unavailable = get_gdn_prefill_backends(inputs, shape)
            for name, reason in unavailable.items():
                if name not in shown:
                    print(f"  unavailable {name}: {reason}")
                    shown.add(name)
            for name, fn in runners.items():
                if name in failed or (args.backends and name not in args.backends):
                    continue
                try:
                    latency = bench_cuda(
                        fn, args.prefill_warmup, args.prefill_iters, args.repeats
                    )
                except Exception as exc:
                    print(f"  failed {name}: {type(exc).__name__}: {exc}")
                    failed.add(name)
                    continue
                tokens = batch * seq_len
                row = {
                    "mode": "prefill",
                    "backend": name,
                    "batch": batch,
                    "seq_len": seq_len,
                    "latency_us": latency,
                    "tokens_s": tokens * 1e6 / latency,
                    "qk_heads": shape.qk_heads,
                    "value_heads": shape.value_heads,
                    "key_dim": shape.key_dim,
                    "value_dim": shape.value_dim,
                    "state_dtype": shape.state_dtype,
                }
                rows.append(row)
                print(
                    f"{name:>28} {batch:5d} {seq_len:8d} "
                    f"{latency / 1000:12.3f} {row['tokens_s']:12.0f}"
                )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("decode", "prefill", "both"),
        default="both",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--prefill-batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--prefill-seq-lens", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--qk-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument(
        "--state-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--prefill-warmup", type=int, default=3)
    parser.add_argument("--prefill-iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backends", nargs="*")
    parser.add_argument(
        "--csv", type=Path, default=(Path(__file__).resolve().parents[2] / "results" / "linear" / "bench_gdn_attention.csv")
    )
    parser.add_argument(
        "--figure",
        type=Path,
        help="output PNG path (default: CSV path with .png suffix)",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    shape = Shape(
        args.qk_heads,
        args.value_heads,
        args.key_dim,
        args.value_dim,
        args.state_dtype,
    )
    if not args.skip_correctness:
        print("CORRECTNESS", check_correctness(shape))
    rows = []
    if args.mode in ("decode", "both"):
        rows.extend(run_decode(args, shape))
    if args.mode in ("prefill", "both"):
        rows.extend(run_prefill(args, shape))
    if rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
        base = args.figure or args.csv.with_suffix(".png")
        decode_rows = [r for r in rows if r["mode"] == "decode"]
        prefill_rows = [r for r in rows if r["mode"] == "prefill"]
        if decode_rows:
            figure = plot_rows(
                decode_rows,
                base.with_name(f"{base.stem}_decode{base.suffix}"),
                x="batch",
                y="latency_us",
                series="backend",
                suptitle="GDN decode: FLA vs SGLang vs vLLM vs TRT-LLM",
            )
            print(f"wrote {figure}")
        if prefill_rows:
            for r in prefill_rows:
                r["panel"] = f"prefill B={r['batch']}"
            figure = plot_rows(
                prefill_rows,
                base.with_name(f"{base.stem}_prefill{base.suffix}"),
                x="seq_len",
                y="tokens_s",
                series="backend",
                panel="panel",
                suptitle="GDN prefill: FLA vs SGLang vs vLLM vs TRT-LLM vs FlashQLA",
            )
            print(f"wrote {figure}")


if __name__ == "__main__":
    main()

