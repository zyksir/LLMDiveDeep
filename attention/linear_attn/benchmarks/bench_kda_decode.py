#!/usr/bin/env python3
"""KDA decode: correctness checks, three benchmarks, three figures.

Every row is registered in ``kda/kda_decode_register.py``; this file only
selects rows, sweeps batch size, and plots (x = batch, y = us/iter).

Correctness: every kernel benchmarked below is checked on the SAME full
pipeline — conv4+SiLU -> recurrence -> gated RMSNorm — against the torch
oracle in ``kda_attention.py`` (registered as the ``torch_kda_reference``
row, shown first in the table as the zero-error baseline). Rows whose kernel
fuses less get the missing stages composed from the torch references, so
fused and unfused pipelines must produce the same output (the TRT-LLM fused
kernel hard-codes the safe gate and is checked against the safe-gate oracle).
Each row is compared against the oracle flavor that rounds where its
pipeline rounds the conv output (the b10/SGLang fused kernels keep it fp32
on-chip, the TRT Triton kernel and all composed chains materialize bf16),
so the tolerance can sit at bf16-ulp level. A second table holds
b10 to SGLang directly on the raw kernel outputs. Both tables use the shared
``common.kernel_bench.error_stats`` measure (max_abs + cosine, pass =
allclose) with the hard-coded ATOL/RTOL below; rows above the bar are
reported as warnings, they do not abort the benchmarks.

Benchmark 1 — fully fused layer step, cross-framework.
    The end-to-end decode step of KDA.md §1.1 — conv4+SiLU + recurrence +
    gated RMSNorm — in ONE kernel per layer per step, one line per framework.
    -> results/bench_kda_decode_e2e_fused.{csv,png}

Benchmark 2 — cost of fusion.
    Each stack's fusion ladder on its own panel (bare recurrence ->
    + gated RMSNorm -> + conv + gated RMSNorm): the per-stage fusion cost is
    the gap between adjacent lines. SGLang ships no gated-norm-only kernel,
    so its ladder has two rungs.
    -> results/bench_kda_decode_fusion_cost.{csv,png}

Benchmark 3 — beyond Kimi K3: recurrence kernels at other head counts, no conv.
    The SGLang/TRT fully fused kernels are compiled for the K3 TP8 shard
    (H = 12) only; the b10 kernels are shape-generic (any H, head_dim = 128).
    For every ``--heads`` value other than 12 (default adds 96, K3 without
    TP — the regime every non-K3 KDA deployment lives in) this compares the
    no-conv kernels; correctness reruns at each shape too.
    -> results/bench_kda_decode_h<H>.{csv,png}

``--heads`` is a list and 12 is always included: benchmarks 1 and 2 only
exist there (the fully fused kernels are H = 12-only). K = V = 128
throughout. Kernel-level only: no projections, scheduler, CUDA graphs, or
engine runtime.

Benchmarks 1 and 3 draw a dashed SOL floor: the step's minimum DRAM
traffic (every input read once, every output written once — dominated by
the fp32 state read + write) divided by the device's theoretical peak
bandwidth, see ``decode_step_bytes`` here and ``sol_plot_kwargs`` /
``peak_dram_bytes_per_s`` in ``common.kernel_bench``.

Output: by default each benchmark saves its figure only; ``--table``
prints the (long) result tables and ``--csv`` writes the CSVs too;
``--no-figure`` turns the figures off. Correctness tables always print.

Usage:
  ../.venv/bin/python bench_kda_decode.py
  ../.venv/bin/python bench_kda_decode.py --heads 12 48 96 --batch-sizes 4 64
  ../.venv/bin/python bench_kda_decode.py --table --csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))  # linear_attn/
sys.path.append(str(Path(__file__).resolve().parents[2]))  # LLMDiveDeep/ (common/)
from common.kernel_bench import bench_impls, error_stats, report, sol_plot_kwargs
from kda.inputs import make_conv_norm_inputs, make_decode_inputs
from kda.kda_attention import (
    SAFE_GATE_LOWER_BOUND,
    activate_kda_gate,
    conv4_silu_reference,
    gated_rmsnorm_reference,
    kda_recurrent_reference,
)
from kda.kda_decode_register import (
    DECODE_FUSED_LAYER,
    DECODE_GATED,
    DECODE_SAFE_GATE,
    KDA_DECODE,
)
from linear_attention import Shape

# Rows prefixed ``b10_`` are this repo's private kernels. To produce the
# public version of this bench, delete every line marked "b10 (private)"
# — nothing else changes.

# Benchmark 1: the fully fused layer step, one line per framework.
E2E_FUSED_ROWS = [
    "b10_kda_decode_conv_gated",  # b10 (private)
    "sglang_kda_fused_decode",
    "trtllm_kda_fused_decode",
]

# Benchmark 2: each stack's fusion ladder, one panel per stack.
FUSION_LADDERS = {
    "b10 ladder": [  # b10 (private)
        "b10_kda_decode",  # b10 (private): bare recurrence
        "b10_kda_decode_gated",  # b10 (private): + gated RMSNorm
        "b10_kda_decode_conv_gated",  # b10 (private): + conv4+SiLU
    ],  # b10 (private)
    "SGLang ladder": [
        "sglang_kda_packed",  # bare recurrence
        "sglang_kda_fused_decode",  # + conv4+SiLU + gated RMSNorm
    ],
}

# Benchmark 3: no-conv recurrence kernels at a non-K3 head count. The b10
# kernels have no H=12 specialization (head_dim=128 is the only constraint),
# unlike the SGLang/TRT fully fused kernels.
NO_CONV_ROWS = [
    "b10_kda_decode",  # b10 (private): bare recurrence
    "b10_kda_decode_gated",  # b10 (private): + gated RMSNorm, still one kernel
    "sglang_kda_packed",
    "trtllm_kda_packed",
    "fla_kda_recurrent",
]

# Every kernel that appears in a benchmark at the default (K3) shape; the
# correctness check covers exactly this set, with the eager-torch pipeline
# first as the baseline. Benchmark 3's rows are re-checked at its own shape.
BENCHMARKED_ROWS = list(
    dict.fromkeys(
        E2E_FUSED_ROWS
        + [name for names in FUSION_LADDERS.values() for name in names]
    )
)
CHECKED_ROWS = ["torch_kda_reference", *BENCHMARKED_ROWS]

# Pass criterion for BOTH correctness tables, via the shared
# common.kernel_bench.error_stats: |out - ref| <= ATOL + RTOL * |ref| per
# element. RTOL = two bf16 ulps: the kernels emit bf16 (one ulp = 2^-7 ~
# 7.8e-3 relative), plus one ulp of slack for the reference rounding the
# other way. A bar this tight only works if each row's oracle rounds where
# its pipeline rounds: the fully fused kernels carry the conv output in
# fp32 on-chip while the composed chains materialize it in bf16, so the
# check builds both oracle flavors and compares each row to its own.
# ATOL = 1e-3 covers near-zero elements, where bf16 ulps are far smaller
# than 1e-3 anyway. Rows above the bar do NOT abort the run — the check
# prints the deviation and the benchmarks proceed.
ATOL, RTOL = 1e-3, 2**-6

# Fused kernels that keep the conv output in fp32 on-chip (checked against
# the fp32-conv oracle). The TRT-LLM Triton kernel is fused too but rounds
# the conv output to bf16 internally (measured: within one bf16 ulp of the
# bf16-conv oracle, ~3x rtol off the fp32 one), so it uses the bf16 flavor.
FP32_CONV_ROWS = {"b10_kda_decode_conv_gated", "sglang_kda_fused_decode"}


def decode_step_bytes(batch: int, shape: Shape, conv: bool) -> int:
    """Minimum DRAM traffic of one decode step, in bytes: every input read
    once, every output written once, nothing else. Dominated by the fp32
    recurrent state (read + write); the bf16 activations are the rest.
    ``conv=True`` adds the conv-fused pipeline's extras (conv state
    read + write, output gate z for the norm)."""
    H, K, V = shape.value_heads, shape.key_dim, shape.value_dim
    per_request = (
        2 * shape.state_bytes_per_request  # state read + write
        + shape.qkv_dim * 2  # q/k/v (or pre-conv mixed_qkv, same size)
        + H * K * 2  # raw gate logits
        + H * 2  # beta logit
        + H * V * 2  # output o
    )
    shared = H * 4 + H * K * 4  # A_log + dt_bias (fp32)
    if conv:
        per_request += shape.qkv_dim * 3 * 2 * 2  # conv state read + write
        per_request += H * V * 2  # output-gate z
        shared += shape.qkv_dim * 4 * 2 + V * 4  # conv weight + norm weight
    return batch * per_request + shared


def check_correctness(shape: Shape, names: list[str]) -> None:
    """Every benchmarked kernel on the SAME full pipeline, vs the torch oracle.

    The ``torch_kda_reference`` row (the eager pipeline built from the
    ``kda_attention.py`` oracle functions) is checked first and must come out
    at exactly zero error — it defines the baseline the kernels are held to.
    One matched input set (nonzero state, nonzero conv state); each row's
    kernel runs the stages it fuses, the torch references supply the rest:

    - fused-layer rows (DECODE_FUSED_LAYER): kernel does conv + recurrence +
      gated norm — compared as-is, against the conv flavor the kernel keeps
      internally (fp32 for FP32_CONV_ROWS, bf16 otherwise).
    - gated rows (DECODE_GATED): torch conv -> kernel (recurrence + norm).
    - everything else: torch conv -> kernel (recurrence) -> torch gated norm.
    """
    B, device = 4, "cuda"
    H, K, V = shape.value_heads, shape.key_dim, shape.value_dim
    cn = make_conv_norm_inputs(B, shape, device=device)  # seed-matched with rows

    def fresh_inputs():
        """Rebuilt per row: kernels update state/conv-state pools in place.
        state_seed makes the recurrence start from a nonzero S0."""
        return make_decode_inputs(B, shape, seed=17, state_seed=23, device=device)

    def reference(lower_bound, fp32_conv):
        """Full-pipeline oracle. ``fp32_conv=True`` keeps the conv output in
        fp32 (what the fully fused kernels do on-chip); ``False`` rounds it
        to bf16 at the conv->recurrence boundary (what the composed chains
        do). Each row is compared against its own flavor so the RTOL bar can
        stay at bf16-ulp level."""
        inputs = fresh_inputs()
        x = inputs.mixed_qkv.float() if fp32_conv else inputs.mixed_qkv
        conv_out = conv4_silu_reference(x, cn.conv_weight, cn.conv_state)
        qc, kc, vc = torch.split(
            conv_out, [shape.qk_heads * K, shape.qk_heads * K, H * V], dim=-1
        )
        g = activate_kda_gate(
            inputs.raw_gate.view(B, 1, H, K),
            inputs.A_log,
            inputs.dt_bias,
            lower_bound=lower_bound,
        )
        beta = torch.sigmoid(inputs.beta_logit.float()).view(B, 1, H)
        o, _ = kda_recurrent_reference(
            qc.view(B, 1, shape.qk_heads, K),
            kc.view(B, 1, shape.qk_heads, K),
            vc.view(B, 1, H, V),
            g,
            beta,
            initial_state=inputs.state.transpose(-1, -2),
        )
        return gated_rmsnorm_reference(o[:, 0], cn.z, cn.norm_weight).float()

    oracles = {
        (None, False): reference(None, False),
        (None, True): reference(None, True),
        (SAFE_GATE_LOWER_BOUND, False): reference(SAFE_GATE_LOWER_BOUND, False),
    }

    rows = []
    kernel_outputs: dict[str, torch.Tensor] = {}  # pre-composition, per row
    for name in names:
        inputs = fresh_inputs()
        composed = []
        if name not in DECODE_FUSED_LAYER:
            composed.append("conv")
            inputs.mixed_qkv = conv4_silu_reference(
                inputs.mixed_qkv, cn.conv_weight, cn.conv_state
            )
        built, unavailable = KDA_DECODE.build(inputs, shape, only=[name])
        if name not in built:
            rows.append({"impl": name, "pass": "SKIP", "note": unavailable[name]})
            continue
        out = built[name]()
        out = out[0] if isinstance(out, tuple) else out
        out = out.reshape(B, H, V)
        kernel_outputs[name] = out.float()
        if name not in DECODE_FUSED_LAYER and name not in DECODE_GATED:
            composed.append("gated norm")
            out = gated_rmsnorm_reference(out, cn.z, cn.norm_weight)
        gate = SAFE_GATE_LOWER_BOUND if name in DECODE_SAFE_GATE else None
        fp32_conv = name in FP32_CONV_ROWS
        if name == "torch_kda_reference":
            note = "eager torch pipeline (the oracle itself) — baseline"
        elif composed:
            note = "kernel + torch " + " + ".join(composed)
        else:
            note = (
                f"fully fused kernel ({'fp32' if fp32_conv else 'bf16'}-conv "
                "oracle" + (", safe gate)" if gate is not None else ")")
            )
        rows.append(
            {
                "impl": name,
                **error_stats(out, oracles[(gate, fp32_conv)], atol=ATOL, rtol=RTOL),
                "note": note,
            }
        )
    report(
        rows,
        columns=["impl", "pass", "max_abs", "cosine", "note"],
        title=f"CORRECTNESS: full pipeline (conv + recurrence + gated norm) "
        f"vs torch oracle, H={H} B={B} (atol {ATOL:g}, rtol {RTOL:g})",
    )
    failed = [r["impl"] for r in rows if r.get("pass") is False]

    # Kernel-vs-kernel match: b10 must behave the same as SGLang. Each pair
    # fuses the same stages, so their raw kernel outputs are compared
    # directly — comparing after the torch-composed norm would amplify a
    # last-bit difference on a small pre-norm output into ~4e-3. Same
    # error_stats criterion as above.
    match_rows = []
    for b10_name, other_name in [
        ("b10_kda_decode_conv_gated", "sglang_kda_fused_decode"),
        ("b10_kda_decode", "sglang_kda_packed"),
    ]:
        if b10_name not in kernel_outputs or other_name not in kernel_outputs:
            continue
        stats = error_stats(
            kernel_outputs[b10_name],
            kernel_outputs[other_name],
            atol=ATOL,
            rtol=RTOL,
        )
        match_rows.append({"b10 impl": b10_name, "vs": other_name, **stats})
        if not stats["pass"]:
            failed.append(f"{b10_name} vs {other_name}")
    if match_rows:
        report(
            match_rows,
            columns=["b10 impl", "vs", "pass", "max_abs", "cosine"],
            title=f"MATCH: b10 vs SGLang raw kernel outputs (same fused "
            f"stages; atol {ATOL:g}, rtol {RTOL:g})",
        )

    if failed:
        # Report, don't abort: the tables above carry the actual deviation
        # (max_abs / cosine) — judge whether it is acceptable, the benchmarks
        # still run either way.
        print(
            f"WARNING: above tolerance (atol {ATOL:g}, rtol {RTOL:g}): "
            f"{', '.join(failed)} — see max_abs/cosine above"
        )


def bench_implementations(names: list[str], args, shape: Shape) -> list[dict]:
    """Time the named registered implementations over the batch sweep and
    return their result rows; unavailable ones are reported once and
    skipped."""
    rows: list[dict] = []
    reported: set[str] = set()
    for batch in args.batch_sizes:
        inputs = make_decode_inputs(batch, shape, seed=batch)
        built, unavailable = KDA_DECODE.build(inputs, shape, only=names)
        for name in names:
            if name in unavailable and name not in reported:
                print(f"  unavailable {name}: {unavailable[name]}")
                reported.add(name)
        runners = {name: built[name] for name in names if name in built}
        rows.extend(
            bench_impls(
                runners,
                args.warmup,
                args.iters,
                args.repeats,
                row_extra=lambda name, us: {
                    "batch": batch,
                    "heads": shape.value_heads,
                    "tokens_s": batch * 1e6 / us,
                },
                notes=KDA_DECODE.notes,
            )
        )
    return [r for r in rows if "latency_us" in r]


def benchmark_e2e_fused(args, shape: Shape) -> None:
    rows = bench_implementations(E2E_FUSED_ROWS, args, shape)
    report(
        rows,
        columns=["impl", "batch", "latency_us", "tokens_s", "note"],
        title=f"Benchmark 1: fused layer step (conv + recurrence + gated "
        f"RMSNorm, ONE kernel) H={shape.value_heads} (us/iter)",
        table=args.table,
        csv=args.csv,
        csv_path=args.out_dir / "bench_kda_decode_e2e_fused.csv",
        plot=None
        if not args.figure
        else dict(
            x="batch",
            y="latency_us",
            suptitle=f"KDA fused decode step (conv4+SiLU + recurrence + gated "
            f"RMSNorm, one kernel), H={shape.value_heads}",
            **sol_plot_kwargs(
                lambda b: decode_step_bytes(b, shape, conv=True),
                args.batch_sizes,
            ),
        ),
    )


def benchmark_fusion_cost(args, shape: Shape) -> None:
    rows: list[dict] = []
    for panel, names in FUSION_LADDERS.items():
        panel_rows = bench_implementations(names, args, shape)
        for row in panel_rows:
            row["panel"] = panel
        rows.extend(panel_rows)
    report(
        rows,
        columns=["panel", "impl", "batch", "latency_us", "note"],
        title=f"Benchmark 2: cost of fusion H={shape.value_heads} (us/iter)",
        table=args.table,
        csv=args.csv,
        csv_path=args.out_dir / "bench_kda_decode_fusion_cost.csv",
        plot=None
        if not args.figure
        else dict(
            x="batch",
            y="latency_us",
            panel="panel",
            suptitle=f"KDA decode: cost of fusing conv / gated RMSNorm into "
            f"the decode kernel, H={shape.value_heads}",
        ),
    )


def benchmark_no_conv(args, shape: Shape) -> None:
    H = shape.value_heads
    rows = bench_implementations(NO_CONV_ROWS, args, shape)
    report(
        rows,
        columns=["impl", "batch", "latency_us", "tokens_s", "note"],
        title=f"Benchmark 3: no-conv decode kernels at H={H} — beyond the K3 "
        f"TP8 shard the fused SGLang/TRT kernels are compiled for (us/iter)",
        table=args.table,
        csv=args.csv,
        csv_path=args.out_dir / f"bench_kda_decode_h{H}.csv",
        plot=None
        if not args.figure
        else dict(
            x="batch",
            y="latency_us",
            suptitle=f"KDA decode without conv at H={H} (non-K3 shape; "
            f"b10_kda_decode_gated also fuses the gated RMSNorm)",
            **sol_plot_kwargs(
                lambda b: decode_step_bytes(b, shape, conv=False),
                args.batch_sizes,
            ),
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument(
        "--heads",
        nargs="+",
        type=int,
        default=[12, 96],
        help="head counts to test; 12 (the Kimi K3 TP8 shard, the only shape "
        "the fully fused kernels exist for) is always included and gets "
        "benchmarks 1+2, every other value gets the no-conv benchmark 3",
    )
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=(Path(__file__).resolve().parents[2] / "results" / "kda"))
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--table",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="print the benchmark result tables (long; correctness tables "
        "always print)",
    )
    parser.add_argument(
        "--csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write the result CSVs next to the figures",
    )
    parser.add_argument(
        "--figure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the figures (the default output)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    heads = args.heads if 12 in args.heads else [12, *args.heads]
    for H in heads:
        shape = Shape(H, H, args.key_dim, args.value_dim, "float32")
        print(f"===== H={H} (K=V={args.key_dim}) =====")
        if H == 12:
            if not args.skip_correctness:
                check_correctness(shape, CHECKED_ROWS)
            benchmark_e2e_fused(args, shape)
            benchmark_fusion_cost(args, shape)
        else:
            if not args.skip_correctness:
                check_correctness(shape, ["torch_kda_reference", *NO_CONV_ROWS])
            benchmark_no_conv(args, shape)


if __name__ == "__main__":
    main()

