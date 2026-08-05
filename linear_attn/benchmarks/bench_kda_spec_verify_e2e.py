#!/usr/bin/env python3
"""KDA spec-decode verify: the e2e PIPELINE (conv4+SiLU -> verify recurrence
-> gated RMSNorm), correctness for every implementation, one benchmark +
figure per scheme.

This complements ``bench_kda_spec_verify.py`` (per-kernel latencies of the
scheme design space): here every row is the WHOLE per-layer verify step an
engine launches per iteration — one fused kernel for the b10 rows, the
production chain of 2-3 kernels for the framework rows (timed back-to-back
on staged buffers; production connects them with strided views, so the
kernel sum is the honest e2e latency). Closures, tensors, and the torch
oracle live in ``kda/kda_verify_register.py``.

Correctness (before benchmarking): every implementation runs on the SAME
raw pre-conv inputs (pnat=0, so the ring is empty and both schemes start
from the same state) and must produce
  - the same post-norm output ``o``,
  - the same committed SSM (save_ssm: the last per-token snapshot;
    replay_ssm: the checkpoint folded with the ring records it wrote),
  - side buffers matching its scheme baseline (b10 snapshots vs SGLang's
    intermediate-state snapshots; b10 ring records vs TRT's old_u/old_k/
    old_G) — shown in the MATCH table.
Gate flavors follow the kernels: SGLang's verify activates the canonical
softplus gate in-kernel, TRT and the b10 replay-SSM kernel hard-code the safe
gate; the b10 save-SSM kernel takes a pre-activated gate so it is checked
under BOTH. Each row's oracle also rounds to bf16 exactly where that row's
pipeline materializes tensors between launches (the conv output and the
pre-norm verify output; the fused kernels keep both fp32 on-chip), same
convention as bench_kda_decode. Rows above the tolerance are reported as
warnings, they do not abort the benchmarks.

Benchmark 1 — save_ssm scheme (full-state snapshots: K*V fp32 per token per
    head persisted). Rows: the b10 ONE-kernel fused step, the b10 kernel
    without the conv fused (+ the separate sglang conv launch), and
    SGLang's production three-kernel chain.
    -> results/bench_kda_spec_verify_save_ssm_h<H>.{csv,png}

Benchmark 2 — replay_ssm scheme (solved-update ring: ~1 KB per token
    per head persisted; steady regime, pnat=8). Rows: the b10 ONE-kernel
    fused step, the b10 kernel without the conv fused (+ conv launch),
    TRT-LLM's production three-kernel chain, and the same chain with the
    verify T-loop replaced by the chunk-form (WY) Triton kernel of
    KDA.md Appendix A.
    -> results/bench_kda_spec_verify_replay_ssm_h<H>.{csv,png}

Benchmark 3 — recurrent-loop vs WY-chunk crossover at fixed T=4. It sweeps
    pnat (accepted tokens waiting in the replay ring) for serving batches
    B=1,4,8. One figure contains one panel per batch, so the full crossover
    study produces only one figure per head count.
    -> results/bench_kda_spec_verify_loop_vs_chunk_h<H>.{csv,png}

The two schemes are NOT plotted together: they persist very different bytes
per draft token (65536 vs ~1024 per head) and leave different rollback
side effects, so a single latency figure would compare unlike work. Each
figure instead carries its own dashed SOL floor (minimum DRAM traffic /
theoretical peak bandwidth, see ``verify_step_bytes``), which makes the
schemes' different byte budgets explicit.

Output: by default each benchmark saves its figure only; ``--table`` prints
the result tables, ``--csv`` writes the CSVs, ``--no-figure`` turns figures
off. Correctness tables always print.

Usage:
  ../.venv/bin/python bench_kda_spec_verify_e2e.py
  ../.venv/bin/python bench_kda_spec_verify_e2e.py --heads 96 --batch-sizes 4 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))  # linear_attn/
sys.path.append(str(Path(__file__).resolve().parents[2]))  # LLMDiveDeep/ (common/)
from common.kernel_bench import bench_impls, error_stats, report, sol_plot_kwargs
from kda import kda_verify_register as R

# Rows prefixed ``b10_`` are this repo's private kernels. To produce the
# public version of this bench, delete every line marked "b10 (private)"
# — nothing else changes.

SAFE = R.VERIFY_LOWER_BOUND

NOTES = {
    "b10_save_ssm_conv_gated":
        "ONE kernel: conv4+SiLU + recurrence + T full-state snapshots + "
        "gated RMSNorm (b10 CuTeDSL)",
    "b10_save_ssm_gated + conv":
        "TWO launches: sglang conv, then the b10 kernel without the conv "
        "fused — prices the conv fusion",
    "sglang_save_ssm_chain":
        "THREE launches (production SGLang): causal_conv1d_update + verify "
        "T-loop with snapshots + FusedRMSNormGated",
    "b10_replay_ssm_conv_gated":
        "ONE kernel: conv4+SiLU + cached-replay verify + ring append + "
        "gated RMSNorm (b10 CuTeDSL)",
    "b10_replay_ssm_gated + conv":
        "TWO launches: sglang conv, then the b10 kernel without the conv "
        "fused — prices the conv fusion",
    "trt_replay_chain":
        "THREE launches (production TRT-LLM): conv + cached-replay verify "
        "(Triton) + gated norm",
    "triton_chunk_replay_chain":
        "THREE launches: TRT's chain with the verify T-loop replaced by the "
        "chunk-form (WY) Triton kernel — tensor-core GEMMs, no sequential "
        "loop (KDA.md Appendix A)",
    "b10_replay_ssm_conv_gated_wychunk":
        "ONE generated CuTeDSL kernel: non-contiguous split Q/K/V + conv + "
        "WY chunk verify + gated norm",
    "recurrent loop (b10 fused)":
        "ONE fused CuTeDSL kernel: conv + sequential replay/verify loop + norm",
    "WY chunk (CuTeDSL fused)":
        "ONE generated CuTeDSL kernel: conv + WY chunk replay/verify + norm",
    "WY chunk (Triton chain)":
        "THREE launches: conv + WY chunk replay/verify + norm",
}

# Same pass criterion as bench_kda_decode: |out - ref| <= ATOL + RTOL*|ref|
# per element via the shared error_stats; RTOL = two bf16 ulps (the o and
# ring records are bf16), ATOL covers near-zero elements. The post-norm
# ``o`` gets a larger ATOL: each element is a 128-term q.S contraction on
# bf16 inputs, so its ABSOLUTE error floor (~1e-2 between independent
# implementations, measured) is set by the summand magnitudes and does not
# shrink with the output element — near-zero o elements keep it, while the
# state/snapshot buffers (rank-1 updates, no contraction) hold 4e-3.
# Deviations warn, they do not abort.
ATOL, RTOL = 4e-3, 2**-6
O_ATOL = 2e-2


def _stats_cols(prefix: str, out, ref, atol=ATOL) -> dict:
    s = error_stats(out, ref, atol=atol, rtol=RTOL)
    return {f"{prefix}_max_abs": s["max_abs"], f"_{prefix}_pass": s["pass"]}


def _finish_row(name: str, note: str, parts: dict) -> dict:
    ok = all(v for k, v in parts.items() if k.startswith("_"))
    row = {"impl": name, "pass": ok}
    row.update({k: v for k, v in parts.items() if not k.startswith("_")})
    row["note"] = note
    return row


def check_correctness(draft: int) -> None:
    """Every implementation on the same raw inputs: same o, same committed
    SSM, side buffers matching the scheme baseline. B=4, pnat=0."""
    B = 4
    t = R.verify_conv_tensors(B, draft, pnat=0, seed=0)
    H, K, V = R.H, R.K, R.V

    # oracle flavors: (lower_bound, bf16_conv, bf16_r) — bf16 rounding where
    # each row's pipeline materializes tensors between launches
    FUSED, NOCONV, CHAIN = (False, False), (True, False), (True, True)
    oracles = {
        (SAFE, FUSED): R.e2e_oracle(t, lower_bound=SAFE, bf16_conv=False,
                                    bf16_r=False),
        (SAFE, NOCONV): R.e2e_oracle(t, lower_bound=SAFE, bf16_conv=True,
                                     bf16_r=False),
        (SAFE, CHAIN): R.e2e_oracle(t, lower_bound=SAFE, bf16_conv=True,
                                    bf16_r=True),
        (None, CHAIN): R.e2e_oracle(t, lower_bound=None, bf16_conv=True,
                                    bf16_r=True),
    }

    rows = []

    def add(name, note, flavor, o, s_committed, snap=None):
        o_ref, snap_ref, s_ref = oracles[flavor]
        parts = _stats_cols("o", o, o_ref, atol=O_ATOL)
        parts.update(_stats_cols("ssm", s_committed, s_ref))
        if snap is not None:
            parts.update(_stats_cols("snap", snap, snap_ref))
        rows.append(_finish_row(name, note, parts))

    # --- the oracle itself, as the zero-error baseline row
    o_ref, snap_ref, s_ref = oracles[(SAFE, FUSED)]
    add("torch_reference", "eager torch pipeline (the oracle itself)",
        (SAFE, FUSED), o_ref, s_ref, snap_ref)

    # --- save_ssm: b10 fused kernel, safe gate, all-fp32 oracle
    run, snap = R.b10_save_ssm_conv_gated_closure(t, SAFE)
    o = run().view(B, draft, H, V)
    snap_b10_safe = snap.view(B, draft, H, K, V)
    add("b10_save_ssm_conv_gated", "fully fused (safe gate; conv and "
        "pre-norm r stay fp32 on-chip)", (SAFE, FUSED), o,
        snap_b10_safe[:, -1], snap_b10_safe)

    # --- save_ssm: b10 kernel without conv (bf16 post-conv in, norm fused)
    run, snap = R.b10_save_ssm_gated_closure(t, SAFE)
    o = run().view(B, draft, H, V)
    snap_v = snap.view(B, draft, H, K, V)
    add("b10_save_ssm_gated (noconv)", "kernel on bf16 post-conv q/k/v",
        (SAFE, NOCONV), o, snap_v[:, -1], snap_v)

    # --- save_ssm: SGLang chain, canonical gate, bf16 at both boundaries
    o, snap_sgl = R.sglang_save_ssm_stitched(t)
    snap_sgl = snap_sgl.transpose(-1, -2)  # [B,d,H,V,K] -> [B,d,H,K,V]
    add("sglang_save_ssm_chain", "3-kernel chain (canonical gate, bf16 "
        "boundaries)", (None, CHAIN), o, snap_sgl[:, -1], snap_sgl)

    # --- replay_ssm: b10 fused kernel, safe gate, all-fp32 on-chip
    run, bufs_b10 = R.b10_replay_ssm_conv_gated_closure(t)
    o = run()
    add("b10_replay_ssm_conv_gated", "fully fused (safe gate); SSM = "
        "checkpoint folded with its ring", (SAFE, FUSED), o,
        R.logical_s_from_ring(bufs_b10, draft))

    # --- replay_ssm: generated CuTeDSL chunk/WY kernel, fully fused
    run, bufs_cute_chunk = R.b10_replay_ssm_conv_gated_wychunk_closure(t)
    o = run()
    add("b10_replay_ssm_conv_gated_wychunk", "fully fused chunk/WY kernel on "
        "non-contiguous split Q/K/V; SSM = checkpoint folded with its ring",
        (SAFE, FUSED), o,
        R.logical_s_from_ring(bufs_cute_chunk, draft))

    # --- replay_ssm: b10 kernel without conv
    run, bufs = R.b10_replay_ssm_conv_gated_closure(
        t, qkv=(t["q"], t["k"], t["v"]))
    o = run()
    add("b10_replay_ssm_gated (noconv)", "kernel on bf16 post-conv q/k/v",
        (SAFE, NOCONV), o, R.logical_s_from_ring(bufs, draft))

    # --- replay_ssm: TRT chain, safe gate, bf16 at both boundaries
    o, bufs_trt = R.trt_replay_stitched(t)
    add("trt_replay_chain", "3-kernel chain (safe gate, bf16 boundaries)",
        (SAFE, CHAIN), o, R.logical_s_from_ring(bufs_trt, draft))

    # --- replay_ssm: chunk-form Triton kernel in the same 3-kernel chain
    o, bufs_chunk = R.triton_chunk_stitched(t)
    add("triton_chunk_replay_chain", "TRT chain w/ chunk-form verify (WY "
        "GEMMs, no T-loop)", (SAFE, CHAIN), o,
        R.logical_s_from_ring(bufs_chunk, draft))

    report(
        rows,
        columns=["impl", "pass", "o_max_abs", "ssm_max_abs", "snap_max_abs",
                 "note"],
        title=f"CORRECTNESS: e2e pipeline (conv + verify + gated norm) vs "
        f"torch oracle — same o / committed SSM / snapshots, H={H} B={B} "
        f"T={draft} pnat=0 (rtol {RTOL:g}; atol {O_ATOL:g} for o, {ATOL:g} "
        f"for state buffers)",
    )
    failed = [r["impl"] for r in rows if r.get("pass") is False]

    # --- MATCH: b10 outputs and side buffers directly against the scheme
    # baselines. The b10 save-SSM kernel reruns under the CANONICAL gate so
    # it is directly comparable with SGLang's chain. This is a cross-flavor
    # comparison (fp32 on-chip vs bf16 kernel boundaries), so each side sits
    # up to ~2 ulps from a shared value: rtol is 2x the oracle checks'.
    match_rtol = 2 * RTOL
    match_rows = []
    run, snap = R.b10_save_ssm_conv_gated_closure(t, None)
    o_b10 = run().view(B, draft, H, V)
    o_sgl, snap_sgl_vk = R.sglang_save_ssm_stitched(t)
    snap_b10 = snap.view(B, draft, H, K, V)
    for what, a, b in [
        ("o", o_b10, o_sgl),
        ("snapshots", snap_b10, snap_sgl_vk.transpose(-1, -2)),
    ]:
        s = error_stats(a, b, atol=O_ATOL if what == "o" else ATOL,
                        rtol=match_rtol)
        match_rows.append({"b10 impl": "b10_save_ssm_conv_gated",
                           "vs": "sglang_save_ssm_chain", "buffer": what, **s})
    for what in ("old_u", "old_k", "old_G"):
        s = error_stats(bufs_b10[what], bufs_trt[what], atol=ATOL,
                        rtol=match_rtol)
        match_rows.append({"b10 impl": "b10_replay_ssm_conv_gated",
                           "vs": "trt_replay_chain", "buffer": what, **s})
    s = error_stats(bufs_b10["S"], t["S"], atol=0.0, rtol=0.0)
    match_rows.append({"b10 impl": "b10_replay_ssm_conv_gated",
                       "vs": "checkpoint untouched (no overflow)",
                       "buffer": "S", **s})
    for r in match_rows:
        if r.get("pass") is False:
            failed.append(f"{r['b10 impl']} vs {r['vs']} ({r['buffer']})")
    report(
        match_rows,
        columns=["b10 impl", "vs", "buffer", "pass", "max_abs", "cosine"],
        title=f"MATCH: b10 vs the scheme baseline, same gate flavor (rtol "
        f"{match_rtol:g} — cross-flavor: fp32 on-chip vs bf16 kernel "
        f"boundaries; atol {O_ATOL:g} for o, {ATOL:g} otherwise)",
    )

    if failed:
        # Report, don't abort: judge max_abs/cosine above, benchmarks run
        # either way.
        print(
            f"WARNING: above tolerance: {', '.join(failed)} — see the "
            f"tables above"
        )


def verify_step_bytes(batch: int, draft: int, pnat: int, snapshots: bool) -> int:
    """Minimum DRAM traffic of one e2e verify step, in bytes: every input
    read once, every output written once. Both schemes read the fp32
    checkpoint but do not write it here (save_ssm commits after sampling,
    replay's steady regime only appends to the ring); they differ in what
    they persist per draft token — T full fp32 states (``snapshots=True``)
    vs T solved-update ring records plus ``pnat`` record reads for the
    replay."""
    H, K, V = R.H, R.K, R.V
    ring_record = 2 * V + 2 * K + 4 * K  # u bf16 + k bf16 + G fp32, per head
    per_request = (
        H * K * V * 4  # checkpoint S read
        + draft * 3 * H * K * 2  # pre-conv mixed_qkv (bf16)
        + 3 * H * K * (R.CONV_WIDTH - 1) * 2 * 2  # conv state read + write
        + draft * H * K * 4  # raw gate logits (fp32)
        + draft * H * 4  # beta logits (fp32)
        + draft * H * V * 2  # output gate z (bf16)
        + draft * H * V * 2  # post-norm output o (bf16)
    )
    if snapshots:
        per_request += draft * H * K * V * 4  # T full-state snapshots (fp32)
    else:
        per_request += (draft + pnat) * H * ring_record  # append + replay reads
    shared = (
        3 * H * K * R.CONV_WIDTH * 2  # conv weight
        + H * 4 + H * K * 4 + V * 4  # A_log + dt_bias + norm weight
    )
    return batch * per_request + shared


def _chain2(first, second):
    """Two launches timed back-to-back as one e2e row."""
    def run():
        first()
        return second()
    return run


def bench_scheme(row_builders, args, draft: int, pnat: int) -> list[dict]:
    """Sweep the scheme's e2e rows over the batch sizes. ``row_builders``
    maps row name -> builder(t) returning the runner closure."""
    rows: list[dict] = []
    for batch in args.batch_sizes:
        t = R.verify_conv_tensors(batch, draft, pnat, seed=batch + draft)
        runners = {name: build(t) for name, build in row_builders.items()}
        rows.extend(
            bench_impls(
                runners,
                args.warmup,
                args.iters,
                args.repeats,
                row_extra=lambda name, us: {
                    "batch": batch,
                    "draft": draft,
                    "heads": R.H,
                    "tokens_s": batch * draft * 1e6 / us,
                },
                notes=NOTES,
            )
        )
        torch.cuda.empty_cache()
    return [r for r in rows if "latency_us" in r]


def benchmark_save_ssm(args, draft: int) -> None:
    H = R.H
    builders = {
        "b10_save_ssm_conv_gated":  # b10 (private)
            lambda t: R.b10_save_ssm_conv_gated_closure(t, SAFE)[0],  # b10 (private)
        "b10_save_ssm_gated + conv":  # b10 (private)
            lambda t: _chain2(R.sglang_conv_closure(t)[0],  # b10 (private)
                              R.b10_save_ssm_gated_closure(t, SAFE)[0]),  # b10 (private)
        "sglang_save_ssm_chain":
            lambda t: R.sglang_save_ssm_chain_closure(t)[0],
    }
    rows = bench_scheme(builders, args, draft, pnat=0)
    report(
        rows,
        columns=["impl", "batch", "latency_us", "tokens_s", "note"],
        title=f"Benchmark 1: save_ssm e2e step (conv + verify + T full-state "
        f"snapshots + gated norm) H={H} T={draft} (us/iter)",
        table=args.table,
        csv=args.csv,
        csv_path=args.out_dir / f"bench_kda_spec_verify_save_ssm_h{H}.csv",
        plot=None
        if not args.figure
        else dict(
            x="batch",
            y="latency_us",
            suptitle=f"KDA spec-verify e2e, save_ssm scheme (conv4+SiLU + "
            f"verify + snapshots + gated RMSNorm), H={H} T={draft}",
            **sol_plot_kwargs(
                lambda b: verify_step_bytes(b, draft, 0, snapshots=True),
                args.batch_sizes,
            ),
        ),
    )


def benchmark_replay_ssm(args, draft: int) -> None:
    H = R.H
    builders = {
        "b10_replay_ssm_conv_gated":  # b10 (private)
            lambda t: R.b10_replay_ssm_conv_gated_closure(t)[0],  # b10 (private)
        "b10_replay_ssm_gated + conv":  # b10 (private)
            lambda t: _chain2(  # b10 (private)
                R.sglang_conv_closure(t)[0],  # b10 (private)
                R.b10_replay_ssm_conv_gated_closure(  # b10 (private)
                    t, qkv=(t["q"], t["k"], t["v"]))[0]),  # b10 (private)
        "trt_replay_chain":
            lambda t: R.trt_replay_chain_closure(t)[0],
        "triton_chunk_replay_chain":
            lambda t: R.triton_chunk_chain_closure(t)[0],
        "b10_replay_ssm_conv_gated_wychunk":
            lambda t: R.b10_replay_ssm_conv_gated_wychunk_closure(t)[0],
    }
    rows = bench_scheme(builders, args, draft, pnat=R.VERIFY_PNAT)
    report(
        rows,
        columns=["impl", "batch", "latency_us", "tokens_s", "note"],
        title=f"Benchmark 2: replay_ssm e2e step (conv + cached-replay "
        f"verify + ring append + gated norm) H={H} T={draft} "
        f"pnat={R.VERIFY_PNAT} (us/iter)",
        table=args.table,
        csv=args.csv,
        csv_path=args.out_dir / f"bench_kda_spec_verify_replay_ssm_h{H}.csv",
        plot=None
        if not args.figure
        else dict(
            x="batch",
            y="latency_us",
            suptitle=f"KDA spec-verify e2e, replay_ssm scheme "
            f"(conv4+SiLU + verify + ring + gated RMSNorm), H={H} T={draft}",
            **sol_plot_kwargs(
                lambda b: verify_step_bytes(
                    b, draft, R.VERIFY_PNAT, snapshots=False
                ),
                args.batch_sizes,
            ),
        ),
    )


def benchmark_loop_vs_chunk(args, draft: int) -> None:
    """Sweep replay length at small serving batches.

    This is an e2e comparison: the recurrent path is the current one-kernel
    fused b10 implementation, while the WY path is the connected
    conv + chunk-verify + norm chain. It therefore answers which complete
    pipeline wins today, including the chunk path's launch overhead.
    """
    H = R.H
    rows: list[dict] = []
    for batch in args.crossover_batch_sizes:
        for pnat in args.replay_lengths:
            if pnat + draft > R.VERIFY_HIST:
                raise ValueError(
                    f"pnat={pnat} + T={draft} exceeds replay ring capacity "
                    f"{R.VERIFY_HIST}"
                )
            t = R.verify_conv_tensors(
                batch, draft, pnat, seed=10_000 + 100 * batch + pnat
            )
            runners = {
                "recurrent loop (b10 fused)":
                    R.b10_replay_ssm_conv_gated_closure(t)[0],
                "WY chunk (CuTeDSL fused)":
                    R.b10_replay_ssm_conv_gated_wychunk_closure(t)[0],
                "WY chunk (Triton chain)":
                    R.triton_chunk_chain_closure(t)[0],
            }
            rows.extend(
                bench_impls(
                    runners,
                    args.warmup,
                    args.iters,
                    args.repeats,
                    row_extra=lambda name, us, b=batch, p=pnat: {
                        "batch": b,
                        "batch_panel": f"B={b}",
                        "pnat": p,
                        "draft": draft,
                        "heads": H,
                        "tokens_s": b * draft * 1e6 / us,
                    },
                    notes=NOTES,
                )
            )
            torch.cuda.empty_cache()

    rows = [r for r in rows if "latency_us" in r]
    report(
        rows,
        columns=["impl", "batch", "pnat", "latency_us", "tokens_s", "note"],
        title=f"Benchmark 3: recurrent loop vs WY chunk e2e crossover, "
        f"H={H} T={draft} (us/iter)",
        table=args.table,
        csv=args.csv,
        csv_path=args.out_dir / f"bench_kda_spec_verify_loop_vs_chunk_h{H}.csv",
        plot=None
        if not args.figure
        else dict(
            x="pnat",
            y="latency_us",
            panel="batch_panel",
            logx=False,
            suptitle=f"KDA replay/verify crossover: recurrent loop vs WY "
            f"chunk, H={H} T={draft}",
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 4, 16, 64],
                        help="spec-decode serving batches are small")
    parser.add_argument("--draft", type=int, default=4,
                        help="verify window T = 1+gamma tokens per request")
    parser.add_argument("--replay-lengths", nargs="+", type=int,
                        default=[0, 1, 2, 4, 8, 12],
                        help="pnat values for the loop-vs-chunk crossover")
    parser.add_argument("--crossover-batch-sizes", nargs="+", type=int,
                        default=[1, 4, 8],
                        help="small serving batches for the crossover figure")
    parser.add_argument("--heads", nargs="+", type=int, default=[12, 96],
                        help="per-rank head counts; 12 = Kimi K3 TP8 shard, "
                        "96 = TP1/PP shard")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--crossover-only", action="store_true",
        help="run only Benchmark 3 (loop-vs-chunk pnat sweep)")
    parser.add_argument(
        "--table", action=argparse.BooleanOptionalAction, default=False,
        help="print the benchmark result tables (long; correctness tables "
        "always print)")
    parser.add_argument(
        "--csv", action=argparse.BooleanOptionalAction, default=False,
        help="write the result CSVs next to the figures")
    parser.add_argument(
        "--figure", action=argparse.BooleanOptionalAction, default=True,
        help="save the figures (the default output)")
    return parser.parse_args()


def main():
    args = parse_args()
    for heads in args.heads:
        R.set_heads(heads)
        print(f"===== H={heads} (K=V=128) =====")
        if not args.skip_correctness and not args.crossover_only:
            check_correctness(args.draft)
        if not args.crossover_only:
            benchmark_save_ssm(args, args.draft)
            benchmark_replay_ssm(args, args.draft)
        benchmark_loop_vs_chunk(args, args.draft)


if __name__ == "__main__":
    main()
