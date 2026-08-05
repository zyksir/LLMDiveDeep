#!/usr/bin/env python3
"""Per-kernel benchmark of the KDA speculative-decode (MTP verify) design space.

WHAT THIS IS
------------
Spec decode makes the target model process T = 1+gamma tokens per request
(bonus token + gamma drafts), then sampling accepts only a prefix, known AFTER
the forward. Every scheme must therefore (a) produce outputs for T tokens,
(b) leave the canonical recurrent state recoverable to any accepted prefix,
(c) never re-forward accepted tokens. KDA.md section 2 develops the taxonomy;
this script times THE INDIVIDUAL KERNELS each scheme launches, on identical
inputs (same B, T window from ``verify_tensors``; K=V=128; head counts swept
via ``--heads``, default 12 = Kimi K3 TP8 shard and 96 = TP1/PP shard).

THIS IS NOT A SCHEME-VS-SCHEME RACE. The schemes trade memory for latency and
produce different side effects (full-state snapshots vs raw-input rings vs
solved-update rings, a current vs lagging checkpoint), so summing rows across
schemes and crowning a winner is not meaningful. The per-kernel rows ARE
meaningful individually: each is a self-contained Triton kernel contract, and
each is a future CuTeDSLGen target — the goal is one generated CuTeDSL kernel
per row that beats that row.

Coverage note: vLLM has no KDA verify kernel to time (its spec-decode state
rollback ships for GDN only; Kimi's KDA layer is not wired into it), so the
three schemes below span every framework that ships one.

THE THREE SCHEMES (names used everywhere in this file; KDA.md section 2)
-------------------------------------------------------------------------
  save_ssm            schema 1 — cache full states, commit the accepted one
  replay_ssm          schema 2 — cache per-token records, replay accepted prefix
    replay_ssm_split    2.1 — cache raw inputs, fold after sampling
    replay_ssm_fused    2.2 — cache solved updates, replay inside verify

Given the same history tokens + draft tokens and full acceptance, all schemes
must leave the same committed SSM under Kimi's safe gate (oracle / b10 / fold /
TRT fused). SGLang Triton's verify is softplus-only — timed for latency, not
checked for correctness.

KERNEL INVENTORY (one kernel per row; what runs when, what it stores)
-------------------------------------------------------------------------
save_ssm — snapshot verify + eager commit (SGLang main; TRT-LLM default).
ONE compute kernel:
  [verify_save_ssm]  recurrent T-loop verify
                     (`fused_sigmoid_gating_delta_rule_update`,
                     disable_state_update=True) that also writes the FULL
                     post-token state after every step: K*V = 16384 fp32
                     per token per head.
  [commit (copy)]    NOT a compute kernel: after sampling, copy
                     snapshot[accept_len-1] into the canonical slot
                     (timed with torch here; frameworks fuse the scatter).

replay_ssm_split — raw-input ring + exact fold (SGLang PR #32541 "ReplaySSM").
TWO kernels:
  [verify_store]     verify T-loop + ring store of raw v, pre-norm k (bf16),
                     activated gate g_k (fp32), beta (fp32):
                     2V+2K+4K+4 = 1028 bytes per token per head vs 65536 for
                     a snapshot. In the PR this is ONE fused Triton kernel;
                     that kernel is not vendored here, so the bench times its
                     two halves separately — verify_store[verify] (the SGLang
                     kernel without snapshot writes) and verify_store[ring]
                     (four strided torch copies). Their sum is an upper bound
                     on the fused kernel.
  [fold_replay]      after sampling: `commit_kda_replayssm_spec` (vendored,
                     Triton) replays the accepted prefix into the checkpoint
                     in place, bit-exact with the decode kernel.

replay_ssm_fused — cached-update ring, replay merged into verify (TRT-LLM
`use_replay_state_update`; contract details in KDA.md section 2.5).
ONE kernel, timed in its two launch REGIMES (same code, different branch):
  [trt_replay_ssm (steady)]
                     `fused_recurrent_gated_delta_rule_cached_replay_update`:
                     computes outputs against checkpoint + ring jointly,
                     appends solved (u bf16, k_norm bf16, G fp32) records =
                     2V+2K+4K = 1024 bytes; checkpoint is NOT written.
  [trt_replay_ssm (overflow)]
                     the SAME kernel on the launch where the 16-deep ring
                     would overflow: its fold branch additionally writes the
                     checkpoint, so that launch is slower. It occurs on
                     ~accept/HIST of launches, so the amortized per-launch
                     cost sits between the two regime timings.

WHY replay_ssm_split NEEDS TWO KERNELS BUT save_ssm / replay_ssm_fused NEED ONE
------------------------------------------------------------------------------
The fold must run AFTER sampling (acceptance known), the verify BEFORE — two
points on the timeline. Merging replay_ssm_split's fold into the NEXT verify
launch would put a sequential raw-input replay (re-do L2 norm + delta-solve
per token) on every launch's critical path; making that merged replay cheap is
exactly what caching solved update vectors does — i.e. the one-kernel version
of input replay IS replay_ssm_fused. What replay_ssm_fused gives up for it: the
bit-exactness argument (split's fold clones the decode kernel's op order;
fused recombines in a different order) and an always-current checkpoint
(fused's lags up to HIST tokens, which any state reader — e.g. prefix cache
dumps — must know about).

GENERATED KERNELS
-----------------
b10 CuTeDSL kernels live under ``kda/b10/`` and are imported directly:
``b10_kda_replay_ssm_cutedsl.kda_replay_ssm`` and
``b10_kda_save_ssm_cutedsl.kda_save_ssm``. Closures and
tensor builders live in ``kda/kda_verify_register.py``. ``--check`` validates
each schema separately in ``check_correctness``. Latency always times b10 next
to the Triton row it must beat.

Two features (plus optional hist/cost probes):

1. ``check_correctness``  — ``--check``: each schema vs its oracle (+ same-SSM)
2. ``check_performance`` / ``check_performance_across_shapes`` — default:
   per-kernel latency, swept over ``--shapes``

USAGE
-----
  python bench_kda_spec_verify.py                     # latency across shapes
  python bench_kda_spec_verify.py --check             # correctness only (+ then latency)
  python bench_kda_spec_verify.py --shapes 1x4 4x4    # latency at selected shapes
  python bench_kda_spec_verify.py --hist-cost         # pnat=1 vs 10 @ B=1,4,16
Accepted-prefix length for post-sampling kernels defaults to T (worst case).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))  # linear_attn/
sys.path.append(str(Path(__file__).resolve().parents[2]))  # LLMDiveDeep/ (common/)
from common.kernel_bench import bench_impls, print_table
from kda import kda_verify_register as V
from kda.kda_verify_register import (
    REPLAY_SSM,
    REPLAY_SSM_SPLIT,
    SAVE_SSM,
    STORED_BYTES_PER_TOKEN_HEAD,
    VERIFY_HIST,
    VERIFY_PNAT,
    commit_gather_closure,
    fold_replay_closure,
    replay_ssm_closure,
    replay_ssm_conv_closure,
    save_ssm_conv_closure,
    ring_buffers,
    ring_store_closure,
    set_heads,
    sglang_verify_closure,
    snapshot_oracle,
    save_ssm_closure,
    trt_closure,
    verify_reference,
    verify_tensors,
)

# Spec-decode serving batches are small (concurrent requests under MTP verify),
# not prefill-scale. Keep B in {1, 4, 16}.
DEFAULT_SHAPES = [(1, 4), (4, 2), (4, 4), (4, 8), (16, 4)]
HIST_COST_SHAPES = [(1, 4), (4, 4), (16, 4)]  # default grid for --hist-cost
DEFAULT_HEADS = [12, 96]


def _check_row(name: str, out, ref, atol: float, rtol: float) -> dict:
    """One correctness row: max abs + pass/fail against the row's own reference."""
    err = (out.float() - ref.float()).abs().max().item()
    return {
        "kernel": name,
        "pass": err <= atol + rtol * ref.float().abs().max().item(),
        "max_abs": err,
    }


def _safe_exec(rows: list[dict], name: str, fn):
    """Run ``fn()``; on failure append an error row and return None."""
    try:
        return fn()
    except Exception as exc:
        rows.append({"kernel": name, "error": f"{type(exc).__name__}: {exc}"})
        return None


def _logical_s_from_ring(bufs, n: int):
    """Logical SSM after accepting ``n`` new ring records (fused does not write S).

    Reads the active half of (old_u, old_k, old_G) written by a pnat=0 verify
    and folds them into the lagging checkpoint — same unrolled form as
    ``verify_tensors``'s history cook.
    """
    bi = int(bufs["buf_idx"][0].item()) if "buf_idx" in bufs else 0
    s = bufs["S"].transpose(-1, -2).float().clone()  # [B,H,K,V]
    G_last = bufs["old_G"][:, bi, :, :, n - 1]
    s = s * torch.exp(G_last)[..., None]
    for step in range(n):
        dec = torch.exp(G_last - bufs["old_G"][:, bi, :, :, step])
        s = s + (bufs["old_k"][:, bi, step].float() * dec)[..., None] * bufs[
            "old_u"
        ][:, bi, step].float()[:, :, None, :]
    return s


def _check_bag(rows: list[dict], kind: str, bag: dict, ref, atol: float, rtol: float):
    """Every entry in ``bag`` must match ``ref`` (same o / same final S / same snaps)."""
    for name, val in bag.items():
        rows.append(_check_row(f"{kind}: {name}", val, ref, atol, rtol))


def check_correctness():
    """Feature 1: collect each kernel's ``o`` and committed ``S``, then check equality.

    Same history + draft + accept=T => every safe-gate (Kimi) kernel produces the
    same ``o`` and final ``S``. ``save_ssm`` also checks per-token middle snapshots.
    SGLang Triton's verify is softplus-only — skipped here; still timed in latency.
    """
    batch, draft = 4, 4
    rows: list[dict] = []
    atol = rtol = 5e-3
    committed_o: dict[str, torch.Tensor] = {}
    committed_s: dict[str, torch.Tensor] = {}
    committed_snap: dict[str, torch.Tensor] = {}  # save_ssm only: [B,T,H,K,V]

    # Shared inputs (empty history): committed S after accept=T = advance S0 by T.
    t = verify_tensors(batch, draft, pnat=0, seed=0)
    o_ref, S_ref = verify_reference(t)  # S_ref: [B,H,K,V]

    # =========================================================================
    # Schema 1: save_ssm — cache full states, commit the accepted snapshot
    # =========================================================================
    o, snap = snapshot_oracle(t)
    committed_o["save_ssm: oracle"] = o
    committed_s["save_ssm: oracle"] = snap[:, -1].contiguous()
    committed_snap["save_ssm: oracle"] = snap

    def _run_b10_save():
        from kda.b10.b10_kda_save_ssm_cutedsl import kda_save_ssm

        run, snap_buf = save_ssm_closure(kda_save_ssm, t)
        o_b10 = run().view(batch, draft, V.H, V.V)
        snap_b10 = snap_buf.view(batch, draft, V.H, V.K, V.V)
        return o_b10, snap_b10

    out = _safe_exec(rows, "save_ssm: b10", _run_b10_save)
    if out is not None:
        o_b10, snap_b10 = out
        committed_o["save_ssm: b10"] = o_b10
        committed_s["save_ssm: b10"] = snap_b10[:, -1].contiguous()
        committed_snap["save_ssm: b10"] = snap_b10

    # =========================================================================
    # Schema 2.1: replay_ssm_split — real fold commits S (safe-gate ring)
    # =========================================================================
    def _run_split_fold():
        rings = ring_buffers(batch, draft)
        ring_store_closure(t, rings)()
        fold, ckpt = fold_replay_closure(t, rings, accept_len=draft)
        fold()
        return ckpt.transpose(-1, -2).contiguous()

    out = _safe_exec(rows, "replay_ssm_split: fold", _run_split_fold)
    if out is not None:
        committed_s["replay_ssm_split: fold"] = out  # real in-place commit

    # =========================================================================
    # Schema 2.2: replay_ssm_fused — b10 vs TRT across pnat regimes
    # =========================================================================
    # Both store solved (u, k̃, G) into the ring and optionally fold hist into S.
    # For each pnat: o vs oracle; b10 vs TRT on o + S + ring (old_u/old_k/old_G).
    def _run_fused_pnats():
        from kda.b10.b10_kda_replay_ssm_cutedsl import kda_replay_ssm

        pnats = (0, 1, 2, 4, 8, 12, VERIFY_HIST - draft + 1)
        bag_o: dict[str, torch.Tensor] = {}
        bag_s: dict[str, torch.Tensor] = {}
        trt_ok = True
        for pnat in pnats:
            # pnat=0 reuses shared ``t`` (seed=0) so same_o / same_s bags align.
            t_f = t if pnat == 0 else verify_tensors(
                batch, draft, pnat, seed=100 + pnat
            )
            o_ref_f, _ = verify_reference(t_f)
            run_b10, bufs_b10 = replay_ssm_closure(kda_replay_ssm, t_f)
            o_b10 = run_b10()
            tag = f"pnat={pnat}"
            rows.append(
                _check_row(f"replay_ssm_fused: b10 {tag} o", o_b10, o_ref_f, atol, rtol)
            )

            o_trt = bufs_trt = None
            if trt_ok:
                try:
                    run_trt, bufs_trt = trt_closure(t_f)
                    o_trt = run_trt()
                except Exception as exc:
                    rows.append({
                        "kernel": "replay_ssm_fused: trt",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    trt_ok = False

            if o_trt is not None and bufs_trt is not None:
                rows.append(
                    _check_row(
                        f"replay_ssm_fused: trt {tag} o", o_trt, o_ref_f, atol, rtol
                    )
                )
                rows.append(
                    _check_row(
                        f"replay_ssm_fused: b10==trt {tag} o", o_b10, o_trt, atol, rtol
                    )
                )
                for name in ("S", "old_u", "old_k", "old_G"):
                    rows.append(
                        _check_row(
                            f"replay_ssm_fused: b10==trt {tag} {name}",
                            bufs_b10[name],
                            bufs_trt[name],
                            atol,
                            rtol,
                        )
                    )

            if pnat + draft > VERIFY_HIST:
                rows.append(
                    _check_row(
                        f"replay_ssm_fused: {tag} folded hist S",
                        bufs_b10["S"],
                        t_f["s_logical"].transpose(-1, -2),
                        atol,
                        rtol,
                    )
                )
            else:
                rows.append(
                    _check_row(
                        f"replay_ssm_fused: {tag} S untouched",
                        bufs_b10["S"],
                        t_f["S"],
                        0.0,
                        0.0,
                    )
                )
            if pnat == 0:
                bag_o["replay_ssm_fused: b10"] = o_b10
                bag_s["replay_ssm_fused: b10"] = _logical_s_from_ring(bufs_b10, draft)
                if o_trt is not None and bufs_trt is not None:
                    bag_o["replay_ssm_fused: trt"] = o_trt
                    bag_s["replay_ssm_fused: trt"] = _logical_s_from_ring(
                        bufs_trt, draft
                    )
        return bag_o, bag_s

    out = _safe_exec(rows, "replay_ssm_fused: b10 vs trt", _run_fused_pnats)
    if out is not None:
        bag_o, bag_s = out
        committed_o.update(bag_o)
        committed_s.update(bag_s)

    # =========================================================================
    # Same o / same final S / same middle snapshots (safe-gate family)
    # =========================================================================
    _check_bag(rows, "same_o", committed_o, o_ref, atol, rtol)
    _check_bag(rows, "same_s", committed_s, S_ref, atol, rtol)
    if committed_snap:
        _check_bag(
            rows, "same_snap", committed_snap, committed_snap["save_ssm: oracle"],
            atol, rtol,
        )

    print_table(
        rows,
        columns=["kernel", "pass", "max_abs", "error"],
        title="CORRECTNESS (B=4 T=4; same draft+history => same o / same S)",
    )


# ---------------------------------------------------------------------------
# Feature 2: performance (kernel latency)
# ---------------------------------------------------------------------------


def check_performance(batch: int, draft: int, accept_len: int):
    """Time every scheme's kernels at ONE (B, T) shape. Prints a latency table."""
    accept_len = min(accept_len, draft)
    t = verify_tensors(batch, draft, VERIFY_PNAT, seed=batch + draft)
    t_ovf = verify_tensors(batch, draft, VERIFY_HIST - draft + 1, seed=batch + draft)

    runners: dict = {}
    meta: dict[str, dict] = {}

    def add(scheme, kernel, fn, note=""):
        runners[kernel] = fn
        stored = (
            f"{STORED_BYTES_PER_TOKEN_HEAD[scheme]} B"
            if "verify" in kernel or kernel.startswith("b10_")
            else "-"
        )
        meta[kernel] = {"scheme": scheme, "stored/token/head": stored, "note": note}

    # --- save_ssm: one kernel + a post-sampling copy
    try:
        run, snap, state = sglang_verify_closure(t, snapshots=True)
        add(SAVE_SSM, "verify_save_ssm", run, "one Triton kernel; writes T full states")
        add(SAVE_SSM, "commit (copy)", commit_gather_closure(snap, state, accept_len),
            f"not a compute kernel; torch gather, accept={accept_len}")
    except Exception:
        pass

    # --- replay_ssm_split: two kernels; verify_store is ONE fused kernel in the PR,
    # timed here as its two halves (fused kernel not vendored)
    try:
        run, _, _ = sglang_verify_closure(t, snapshots=False)
        add(REPLAY_SSM_SPLIT, "verify_store[verify]", run,
            "verify half of the PR's fused kernel")
    except Exception:
        pass
    rings = ring_buffers(batch, draft)
    add(REPLAY_SSM_SPLIT, "verify_store[ring]", ring_store_closure(t, rings),
        "ring half; 4 strided torch copies; sum of halves = upper bound")
    try:
        run, _ = fold_replay_closure(t, rings, accept_len)
        add(REPLAY_SSM_SPLIT, "fold_replay", run,
            f"Triton; post-sampling, accept={accept_len}")
    except Exception:
        pass

    # --- replay_ssm_fused: ONE kernel, two launch regimes of the same code
    try:
        run, _ = trt_closure(t)
        add(REPLAY_SSM, "trt_replay_ssm (steady)", run,
            f"ring append only, pnat={VERIFY_PNAT}")
        run, _ = trt_closure(t_ovf)
        add(REPLAY_SSM, "trt_replay_ssm (overflow)", run,
            f"SAME kernel, pnat={VERIFY_HIST - draft + 1}: fold branch also writes S")
    except Exception:
        pass

    # --- b10 CuTeDSL kernels — always timed next to the Triton rows they must
    # beat. Their drivers launch on a cached raw stream, so CUDA-graph capture
    # records an empty graph; use_graph=False forces eager CUDA-event timing.
    def add_b10(scheme, name, run, note):
        run.use_graph = False
        add(scheme, name, run, note)

    try:
        from kda.b10.b10_kda_save_ssm_cutedsl import kda_save_ssm
        run, _ = save_ssm_closure(kda_save_ssm, t)
        add_b10(SAVE_SSM, "b10_save_ssm", run, "b10 CuTeDSL save_ssm kernel")
    except Exception:
        pass
    try:
        from kda.b10.b10_kda_save_ssm_conv_cutedsl import kda_save_ssm_conv
        run, _, _ = save_ssm_conv_closure(kda_save_ssm_conv, t)
        add_b10(SAVE_SSM, "b10_save_ssm_conv", run,
                "b10 save_ssm + fused conv4+SiLU input stage (replaces the "
                "separate conv launch; also writes conv_win)")
    except Exception:
        pass
    try:
        from kda.b10.b10_kda_replay_ssm_cutedsl import kda_replay_ssm
        run, _ = replay_ssm_closure(kda_replay_ssm, t)
        add_b10(REPLAY_SSM, "b10_replay_ssm", run,
                "b10 CuTeDSL replay_ssm_fused kernel")
        run, _ = replay_ssm_closure(kda_replay_ssm, t_ovf)
        add_b10(REPLAY_SSM, "b10_replay_ssm (overflow)", run,
                "b10 CuTeDSL replay_ssm_fused, folds S")
    except Exception:
        pass
    try:
        from kda.b10.b10_kda_replay_ssm_conv_cutedsl import (
            kda_replay_ssm_conv,
        )
        run, _ = replay_ssm_conv_closure(kda_replay_ssm_conv, t)
        add_b10(REPLAY_SSM, "b10_replay_ssm_conv", run,
                "b10 replay_ssm + fused conv input stage "
                "(conv_state read-only; also writes conv_win)")
    except Exception:
        pass

    rows = bench_impls(
        runners, warmup=5, iters=50, repeats=5,
        row_extra=lambda name, us: meta[name],
    )
    for row in rows:  # row_extra only runs on success; failed rows need meta too
        if "error" in row:
            row.update(meta[row["impl"]])
    print_table(
        rows,
        columns=["scheme", "impl", "latency_us", "stored/token/head", "note", "error"],
        title=f"LATENCY H={V.H} B={batch} T={draft}  (accept_len={accept_len}; us/iter; "
        "rows are NOT summable across schemes — see module docstring)",
    )


def check_performance_across_shapes(
    shapes: list[tuple[int, int]], accept_len: int | None = None
) -> None:
    """Sweep :func:`check_performance` over a list of (B, T) shapes."""
    for batch, draft in shapes:
        check_performance(batch, draft, accept_len or draft)


def run_cached_replay_cost_breakdown(batch: int, draft: int):
    """Answer "where does trt_replay_ssm's time go?" by sweeping the
    ring fill (pnat = tokens waiting in the ring): pnat=0 does no replay work
    and reads no ring bytes, so its delta over the SGLang verify T-loop (same
    state bytes, no ring) is pure kernel structure; the pnat=0 -> 2 jump is
    the replay-branch entry; pnat=2 -> 12 is the actual ring traffic.
    Findings and roofline in KDA.md section 2.6."""
    t = verify_tensors(batch, draft, VERIFY_PNAT, seed=1)
    run, _, _ = sglang_verify_closure(t, snapshots=False)
    runners = {"sglang verify T-loop (no stores)": run}
    for pnat in (0, 2, 4, 8, 12):
        t = verify_tensors(batch, draft, pnat, seed=pnat + 1)
        runners[f"trt trt_replay_ssm pnat={pnat}"], _ = trt_closure(t)
    rows = bench_impls(runners, warmup=5, iters=50, repeats=5)
    print_table(
        rows,
        title=f"WHERE trt_replay_ssm SPENDS ITS TIME — ring-fill sweep, "
        f"B={batch} T={draft} (us; see KDA.md 4.5)",
    )


def run_hist_replay_cost(batch: int, draft: int):
    """Cost of replaying history: same verify window T, pnat=1 vs pnat=10.

    ``pnat`` is the number of accepted-but-unfolded tokens already in the
    ring that the kernel must replay before processing the new T tokens.
    Reports absolute kernel latency at each pnat, plus
    Delta(pnat=10 − pnat=1) = marginal cost of ~9 extra hist replays
    (steady regime: no overflow fold, since 10+T <= HIST=16 for T<=4).
    Times both TRT Triton and b10 CuTeDSL on identical tensors.
    """
    runners: dict = {}
    # keep both in the steady (no-fold) regime: pnat + T <= HIST
    for pnat in (1, 10):
        assert pnat + draft <= VERIFY_HIST, (
            f"pnat={pnat} + T={draft} would overflow HIST={VERIFY_HIST}; "
            "pick a smaller T for a clean hist-replay delta"
        )
        t = verify_tensors(batch, draft, pnat, seed=100 + pnat)
        runners[f"trt  pnat={pnat}"], _ = trt_closure(t)
        try:
            from kda.b10.b10_kda_replay_ssm_cutedsl import kda_replay_ssm
            runners[f"b10  pnat={pnat}"], _ = replay_ssm_closure(kda_replay_ssm, t)
        except Exception as exc:
            runners[f"b10  pnat={pnat}"] = lambda e=exc: (_ for _ in ()).throw(e)
    raw = bench_impls(runners, warmup=5, iters=50, repeats=5)
    by = {r["impl"]: r for r in raw}
    # one summary row per kernel: abs latency at each pnat + delta
    summary: list[dict] = []
    for tag in ("trt", "b10"):
        a, b = by.get(f"{tag}  pnat=1"), by.get(f"{tag}  pnat=10")
        row: dict = {"impl": tag}
        if a and "error" in a:
            row["error"] = a["error"]
        elif b and "error" in b:
            row["error"] = b["error"]
        elif a and b and "latency_us" in a and "latency_us" in b:
            d = b["latency_us"] - a["latency_us"]
            row.update({
                "abs_us_pnat1": a["latency_us"],
                "abs_us_pnat10": b["latency_us"],
                "delta_us": d,
                "us_per_hist": d / 9.0,
                "ratio": b["latency_us"] / max(a["latency_us"], 1e-9),
            })
        else:
            row["error"] = "missing measurement"
        summary.append(row)
    print_table(
        summary,
        columns=[
            "impl", "abs_us_pnat1", "abs_us_pnat10",
            "delta_us", "us_per_hist", "ratio", "error",
        ],
        title=f"HIST REPLAY COST — abs kernel latency + pnat=1→10 delta, "
        f"H={V.H} B={batch} T={draft} (steady, no fold; us/iter)",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shapes", nargs="*", default=None,
                        help="BxT pairs, e.g. 1x4 4x4 16x4 (default: "
                        "spec-decode grid B in {1,4,16}; --hist-cost defaults "
                        "to 1x4 4x4 16x4)")
    parser.add_argument("--heads", nargs="+", type=int, default=DEFAULT_HEADS,
                        help="per-rank head counts to sweep; defaults are Kimi "
                        "K3's TP8 shard (12) and TP1/PP shard (96)")
    parser.add_argument("--cost-breakdown", action="store_true",
                        help="answer 'where does trt_replay_ssm's time "
                        "go?' by sweeping the ring fill (pnat), isolating "
                        "kernel structure vs replay-branch vs ring bytes")
    parser.add_argument("--hist-cost", action="store_true",
                        help="measure hist-replay cost: pnat=1 vs pnat=10 "
                        "(same T window, steady regime) for TRT and b10; "
                        "default shapes B=1,4,16")
    parser.add_argument("--accept-len", type=int, default=None,
                        help="accepted prefix for commit/fold kernels (default T)")
    parser.add_argument("--check", action="store_true",
                        help="feature 1: check_correctness (per-schema + same-SSM)")
    args = parser.parse_args()
    if args.shapes:
        shapes = [tuple(map(int, s.split("x"))) for s in args.shapes]
    elif args.hist_cost:
        shapes = list(HIST_COST_SHAPES)
    else:
        shapes = list(DEFAULT_SHAPES)
    for heads in args.heads:
        set_heads(heads)
        print(f"\n===== H={heads} (K=V=128) =====")
        if args.check:
            check_correctness()
        if args.hist_cost:
            for batch, draft in shapes:
                run_hist_replay_cost(batch, draft)
            continue
        if args.cost_breakdown:
            for batch, draft in shapes:
                run_cached_replay_cost_breakdown(batch, draft)
            continue
        # default: latency across the shape grid
        check_performance_across_shapes(shapes, args.accept_len)


if __name__ == "__main__":
    main()
