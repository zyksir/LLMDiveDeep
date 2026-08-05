#!/usr/bin/env python3
"""Synthetic KDA kernel benchmark: prefill scaling and decode latency.

What this isolates: SGLang-centric decode/prefill paths (split vs packed
serving fusions) plus a Chrome-compatible profiler trace; kernel-level, no
model weights or server.

Decode deliberately accepts ``--context-lens`` even though KDA does not read
history tokens. Repeating the same state-shaped work at each label makes the
constant-context-cost property directly visible (and catches accidental scans).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # linear_attn/
sys.path.append(str(Path(__file__).resolve().parents[2]))  # LLMDiveDeep/ (common/)
from kda.inputs import make_decode_inputs, make_prefill_inputs
from kda.kda_attention import activate_kda_gate, kda_recurrent_reference
from kda.kda_decode_register import get_kda_decode_backends
from kda.kda_prefill_register import get_kda_sglang_extend_backends
from linear_attention import Shape
from common.kernel_bench import bench_cuda, plot_rows


def check_reference(shape: Shape) -> dict[str, float]:
    """Compare packed SGLang decode with the exact one-step recurrence."""

    data = make_decode_inputs(2, shape, seed=123)
    qf, kf, vf = torch.split(
        data.mixed_qkv,
        [
            shape.qk_heads * shape.key_dim,
            shape.qk_heads * shape.key_dim,
            shape.value_heads * shape.value_dim,
        ],
        dim=-1,
    )
    q = qf.view(2, 1, shape.qk_heads, shape.key_dim)
    k = kf.view(2, 1, shape.qk_heads, shape.key_dim)
    v = vf.view(2, 1, shape.value_heads, shape.value_dim)
    raw_gate = data.raw_gate.view(2, 1, shape.value_heads, shape.key_dim)
    log_decay = activate_kda_gate(raw_gate, data.A_log, data.dt_bias)
    beta = data.beta_logit.sigmoid().view(2, 1, shape.value_heads)
    ref_out, ref_state = kda_recurrent_reference(q, k, v, log_decay, beta)

    packed_state = data.state.clone()
    packed_data = replace(data, state=packed_state)
    runners, unavailable = get_kda_decode_backends(packed_data, shape)
    if "sglang_kda_packed" not in runners:
        raise RuntimeError(
            f"packed Triton KDA unavailable: {unavailable.get('sglang_kda_packed')}"
        )
    got = runners["sglang_kda_packed"]().transpose(0, 1)
    got_state = packed_state.transpose(-1, -2)
    result = {
        "output_max_abs": (got.float() - ref_out.float()).abs().max().item(),
        "output_cosine": torch.nn.functional.cosine_similarity(
            got.float().flatten(), ref_out.float().flatten(), dim=0
        ).item(),
        "state_max_abs": (got_state.float() - ref_state.float()).abs().max().item(),
        "state_cosine": torch.nn.functional.cosine_similarity(
            got_state.float().flatten(), ref_state.float().flatten(), dim=0
        ).item(),
    }
    fla_state = data.state.clone()
    fla_runners, _ = get_kda_decode_backends(replace(data, state=fla_state), shape)
    if "fla_kda_recurrent" in fla_runners:
        fla_out = fla_runners["fla_kda_recurrent"]().transpose(0, 1)
        fla_final = fla_state.transpose(-1, -2)
        result.update(
            fla_output_max_abs=(
                fla_out.float() - ref_out.float()
            ).abs().max().item(),
            fla_output_cosine=torch.nn.functional.cosine_similarity(
                fla_out.float().flatten(),
                ref_out.float().flatten(),
                dim=0,
            ).item(),
            fla_state_max_abs=(
                fla_final.float() - ref_state.float()
            ).abs().max().item(),
            fla_state_cosine=torch.nn.functional.cosine_similarity(
                fla_final.float().flatten(),
                ref_state.float().flatten(),
                dim=0,
            ).item(),
        )
    return result


def run_decode(args, shape: Shape) -> list[dict]:
    rows: list[dict] = []
    print("\nDECODE — one recurrent update; history length is metadata only")
    print(
        f"{'backend':>24} {'B':>5} {'context':>10} {'latency us':>12} "
        f"{'tok/s':>12} {'state MiB':>11}"
    )
    shown_unavailable: set[str] = set()
    failed_backends: set[str] = set()
    for batch in args.batch_sizes:
        for context_len in args.context_lens:
            data = make_decode_inputs(batch, shape, seed=batch + context_len)
            runners, unavailable = get_kda_decode_backends(data, shape)
            for name, reason in unavailable.items():
                if name not in shown_unavailable:
                    print(f"  unavailable {name}: {reason}")
                    shown_unavailable.add(name)
            for backend, fn in runners.items():
                if backend in failed_backends:
                    continue
                if args.backends and backend not in args.backends:
                    continue
                try:
                    latency_us = bench_cuda(fn, args.warmup, args.iters, args.repeats)
                except Exception as exc:
                    key = f"{backend}:{type(exc).__name__}:{exc}"
                    if key not in shown_unavailable:
                        print(f"  failed {backend}: {type(exc).__name__}: {exc}")
                        shown_unavailable.add(key)
                    failed_backends.add(backend)
                    continue
                tokens_s = batch * 1e6 / latency_us
                state_mib = batch * shape.state_bytes_per_request / 2**20
                row = {
                    "mode": "decode",
                    "backend": backend,
                    "batch": batch,
                    "seq_len": 1,
                    "context_len": context_len,
                    "tokens": batch,
                    "latency_us": latency_us,
                    "tokens_s": tokens_s,
                    "state_mib": state_mib,
                    "state_dtype": shape.state_dtype,
                }
                rows.append(row)
                print(
                    f"{backend:>24} {batch:5d} {context_len:10d} "
                    f"{latency_us:12.2f} {tokens_s:12.0f} {state_mib:11.2f}"
                )
    return rows


def run_prefill(args, shape: Shape) -> list[dict]:
    rows: list[dict] = []
    print("\nPREFILL — chunk-parallel KDA")
    print(
        f"{'backend':>24} {'B':>5} {'S':>10} {'latency ms':>12} "
        f"{'tok/s':>12} {'peak MiB':>10}"
    )
    shown_unavailable: set[str] = set()
    failed_backends: set[str] = set()
    for batch in args.prefill_batch_sizes:
        for seq_len in args.prefill_seq_lens:
            data = make_prefill_inputs(batch, seq_len, shape, seed=batch + seq_len)
            runners, unavailable = get_kda_sglang_extend_backends(
                data, lower_bound=args.lower_bound
            )
            for name, reason in unavailable.items():
                if name not in shown_unavailable:
                    print(f"  unavailable {name}: {reason}")
                    shown_unavailable.add(name)
            for backend, fn in runners.items():
                if backend in failed_backends:
                    continue
                if args.backends and backend not in args.backends:
                    continue
                torch.cuda.reset_peak_memory_stats()
                try:
                    latency_us = bench_cuda(fn, args.warmup, args.iters, args.repeats)
                except Exception as exc:
                    key = f"{backend}:{type(exc).__name__}:{exc}"
                    if key not in shown_unavailable:
                        print(f"  failed {backend}: {type(exc).__name__}: {exc}")
                        shown_unavailable.add(key)
                    failed_backends.add(backend)
                    continue
                tokens = batch * seq_len
                tokens_s = tokens * 1e6 / latency_us
                peak_mib = torch.cuda.max_memory_allocated() / 2**20
                row = {
                    "mode": "prefill",
                    "backend": backend,
                    "batch": batch,
                    "seq_len": seq_len,
                    "context_len": seq_len,
                    "tokens": tokens,
                    "latency_us": latency_us,
                    "tokens_s": tokens_s,
                    "state_mib": batch * shape.state_bytes_per_request / 2**20,
                    "state_dtype": shape.state_dtype,
                    "peak_mib": peak_mib,
                }
                rows.append(row)
                print(
                    f"{backend:>24} {batch:5d} {seq_len:10d} "
                    f"{latency_us / 1000:12.3f} {tokens_s:12.0f} {peak_mib:10.1f}"
                )
            del data, runners
            torch.cuda.empty_cache()
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_profile(args, shape: Shape) -> None:
    data = make_decode_inputs(args.profile_batch, shape, seed=7)
    runners, unavailable = get_kda_decode_backends(data, shape)
    backend = args.profile_backend
    if backend not in runners:
        raise RuntimeError(
            f"profile backend {backend!r} unavailable; choices={list(runners)}, "
            f"errors={unavailable}"
        )
    fn = runners[backend]
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        acc_events=True,
    ) as prof:
        for _ in range(args.profile_steps):
            fn()
            prof.step()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.trace))
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=12,
        )
    )
    print(f"\nWrote decode trace: {args.trace}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("decode", "prefill", "all"), default="decode")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument(
        "--context-lens",
        type=int,
        nargs="+",
        default=[4096, 32768, 131072, 1048576],
    )
    parser.add_argument("--prefill-batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--prefill-seq-lens", type=int, nargs="+", default=[512, 2048, 8192]
    )
    parser.add_argument("--qk-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument(
        "--state-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="recurrent-state dtype; SGLang defaults to float32",
    )
    parser.add_argument(
        "--lower-bound",
        type=float,
        default=-5.0,
        help="safe-gate lower bound for prefill; pass 'nan' to use canonical gate",
    )
    parser.add_argument("--backends", nargs="*", default=[])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--csv", type=Path, default=Path("results/bench_linear_attention.csv")
    )
    parser.add_argument(
        "--figure",
        type=Path,
        help="output PNG path (default: CSV path with .png suffix)",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-backend", default="sglang_kda_packed")
    parser.add_argument("--profile-batch", type=int, default=32)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument(
        "--trace", type=Path, default=Path("results/kda_decode_trace.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.lower_bound != args.lower_bound:  # NaN is an argparse-friendly None.
        args.lower_bound = None
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for optimized-kernel benchmarks")
    shape = Shape(
        qk_heads=args.qk_heads,
        value_heads=args.value_heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        state_dtype=args.state_dtype,
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(f"shape: {json.dumps(asdict(shape))}")
    print(
        f"state/request: {shape.state_bytes_per_request / 2**20:.3f} MiB "
        f"({shape.state_dtype})"
    )
    correctness = check_reference(shape)
    print(f"correctness: {json.dumps(correctness, sort_keys=True)}")

    rows = []
    if args.mode in ("decode", "all"):
        rows.extend(run_decode(args, shape))
    if args.mode in ("prefill", "all"):
        rows.extend(run_prefill(args, shape))
    write_csv(args.csv, rows)
    print(f"\nWrote {len(rows)} rows: {args.csv}")
    if rows:
        base = args.figure or args.csv.with_suffix(".png")
        decode_rows = [r for r in rows if r["mode"] == "decode"]
        prefill_rows = [r for r in rows if r["mode"] == "prefill"]
        if decode_rows:
            ctxs = {r["context_len"] for r in decode_rows}
            for r in decode_rows:
                r["line"] = (
                    f"{r['backend']} ctx={r['context_len']}"
                    if len(ctxs) > 1
                    else r["backend"]
                )
            figure = plot_rows(
                decode_rows,
                base.with_name(f"{base.stem}_decode{base.suffix}"),
                x="batch",
                y="latency_us",
                series="line",
                suptitle="KDA decode kernels",
            )
            print(f"Wrote {figure}")
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
                suptitle="KDA chunk prefill kernels",
            )
            print(f"Wrote {figure}")
    if args.profile:
        write_profile(args, shape)


if __name__ == "__main__":
    main()
