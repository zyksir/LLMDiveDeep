#!/usr/bin/env python3
"""KDA chunk-prefill benchmark: matched fused-raw-gate kernels across ALL
known implementations.

Coverage (see each registration's ``note`` in ``kda/kda_prefill_register.py`` —
printed in the report — for provenance and which rows are forks of the same code):

- FLA upstream Triton chunk pipeline (`fla_kda_chunk`, canonical gate) and
  its safe-gate variant (`fla_kda_safe_triton`, the matched FlashKDA baseline)
- SGLang / vLLM / TRT-LLM: all three vendor forks of the same FLA pipeline
  (different gate/beta fusions — noted per row)
- FlashKDA (MoonshotAI CUTLASS; also what FLA auto-dispatches to and what
  SGLang's `flashkda` backend wraps)
- INT21 KDA-B200 (`flashkda_ptx_int21`): plain CUDA/PTX rewrite of FlashKDA,
  ~1.5x faster, valid only up to 262144 total tokens (int32 offsets)
- TRT-LLM in-house CuTe DSL Blackwell prefill (`trtllm_kda_cute`, the `cute`
  backend of the `trid/kda-cute-int21-snapshots` branch; that branch's
  `triton`/`flashkda`/`int21` backends wrap the same kernels as
  `trtllm_kda_chunk` / `flash_kda` / `flashkda_ptx_int21` above)
- this repo's b10 Triton port of FlashKDA's algorithm (c16/c64) and the
  seq<=1024 short-prefill CuTeDSL probe

Kernel-level only — no projections, conv, norm, scheduler, CUDA graphs, or
engine runtime.

Framework (``common/kernel_bench.py``): every implementation adapts the SAME
canonical inputs in its untimed builder; only the run closure is timed;
correctness compares normalized outputs on identical inputs; reports are
dict rows printed by the shared table printer.

Modes:

- ``prefill`` (default): one in-process sweep over ``--batch-sizes`` x
  ``--seq-lens``. Known int32-overflow crashers (FLA / SGLang chunk kernels on
  huge shapes) are pre-skipped: an illegal access would poison the CUDA
  context and kill every later kernel in the process.
- ``stages``: the chunk pipeline *split* into gate cumsum / WY prep /
  state carry / output leaf-kernel timings (SGLang kernels, same inputs).
- ``sweep``: the full evaluation grid (B in {1..64} x S in {4k, 64k, 200k})
  with per-shape subprocess isolation. Re-invokes this script in ``prefill``
  mode.

b10 kernels live under ``kda/b10/`` and are registered as ordinary backends
in ``kda/kda_prefill_register.py``; each CuTeDSL entry point asserts the baked
head_dim=128 contract.

Usage:
  ../.venv/bin/python bench_kda_prefill.py                       # default sweep
  ../.venv/bin/python bench_kda_prefill.py --mode stages
  ../.venv/bin/python bench_kda_prefill.py --mode sweep          # full grid
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))  # linear_attn/
sys.path.append(str(Path(__file__).resolve().parents[2]))  # LLMDiveDeep/ (common/)
from common.kernel_bench import (
    bench_impls,
    check_impls,
    plot_rows,
    print_table,
    write_csv,
)
from kda.inputs import make_prefill_inputs
from kda.kda_attention import activate_kda_gate, kda_recurrent_reference
from kda.kda_prefill_register import (
    KDA_PREFILL,
    get_kda_prefill_backends,
)
from linear_attention import Shape

# --mode sweep grid (win condition: beat flash_kda at every B > 1)
SWEEP_BATCHES = [1, 2, 4, 8, 16, 32, 64]
SWEEP_SEQ_LENS = [4096, 65536, 204800]
# built/benched together in one subprocess (no known crashes; flashkda-ptx
# raises cleanly at build time past its 262144-token envelope, no IMA)
SWEEP_SAFE_BACKENDS = [
    "flashinfer_cake_kda",
    "flash_kda",
    "flashkda_ptx_int21",
    "b10_flashkda_triton_c16",
    "b10_flashkda_triton_c64",
    "b10_kda_chunk_prefill",
]
# each gets its own subprocess per shape: both IMA on big shapes (int32
# offsets overflow — FLA past ~1.6M tokens, SGLang's [B,NT,H,K,V] fp32
# intermediate past ~2^31 elements) and a poisoned context kills neighbours
SWEEP_ISOLATED_BACKENDS = ["fla_kda_chunk", "sglang_kda_chunk"]

def baseline_is_safe(name: str, batch: int, seq: int, heads: int = 96) -> bool:
    """Pre-skip kernels that IMA on big shapes — illegal access poisons the
    whole CUDA context, so these cannot be caught at run time.
    Boundaries measured empirically on B200.

    flashkda_ptx_int21: Despite the Python-level check of 262144 total tokens,
    the kernel causes IMA at much lower token counts for large head dimensions.
    Measured limits (H=96, K=V=128):
      - B=14 T=4096 (57344 tokens): SAFE
      - B=15 T=4096 (61440 tokens): IMA
    Root cause: the kernel uses narrower-than-int32 index arithmetic for
    large heads × tokens × dim products. Conservative safe limit: 57344 for
    H>=96. For H<=12 (smaller per-head working set) the limit is much higher
    and the original 262144-token check holds."""
    tokens = batch * seq
    if name == "fla_kda_chunk":
        return tokens <= 1 << 20
    if name == "sglang_kda_chunk":
        return tokens <= 1 << 20 and not (seq >= 204800 and batch >= 4)
    if name == "flashkda_ptx_int21":
        from kda.kda_prefill_register import _FLASHKDA_PTX_MAX_TOKENS
        if tokens > _FLASHKDA_PTX_MAX_TOKENS:
            return False
        # Empirical IMA boundary for large head dimensions
        if heads >= 96:
            return tokens <= 57344   # B=14, T=4096 verified safe; B=15 crashes
    return True


# kernels using the canonical softplus gate vs the safe (bounded sigmoid)
# gate — each group is checked against its own exact-recurrence reference
CANONICAL_GATE_CHECKED = (
    "fla_kda_chunk",
    "sglang_kda_chunk",
    "vllm_kda_chunk",
    "b10_kda_chunk_prefill",
)
SAFE_GATE_CHECKED = (
    "fi_recurrent_kda",
    "fla_kda_safe_triton",
    "flash_kda",
    "flashkda_ptx_int21",
    "trtllm_kda_chunk",
    "trtllm_kda_cute",
    "flashinfer_cake_kda",
)


def check_correctness(shape: Shape) -> None:
    """Every implementation against the exact fp32 token recurrence, on the
    SAME canonical inputs (fresh copies per impl: vLLM writes its output into
    `v` in place, SGLang/TRT-LLM update the indexed state pool in place)."""
    from kda.kda_attention import SAFE_GATE_LOWER_BOUND as _SAFE_GATE_LOWER_BOUND

    seed_inputs = make_prefill_inputs(2, 128, shape, seed=19)
    unpack = lambda tensor: tensor.view(2, 128, *tensor.shape[2:])

    def reference(lower_bound: float | None):
        out, state = kda_recurrent_reference(
            unpack(seed_inputs.q),
            unpack(seed_inputs.k),
            unpack(seed_inputs.v),
            activate_kda_gate(
                unpack(seed_inputs.raw_gate),
                seed_inputs.A_log,
                seed_inputs.dt_bias,
                lower_bound=lower_bound,
            ),
            unpack(seed_inputs.beta),
        )
        return out, state.transpose(-1, -2)

    states: dict[str, torch.Tensor] = {}

    def make_fresh_input_runner(name: str):
        """Runner that rebuilds the (seed-identical) inputs on every call, so
        one backend's in-place mutation (vLLM writes into `v`, SGLang/TRT-LLM
        update the indexed state pool) cannot leak into the next."""

        def run():
            # Warm Triton autotune caches on throwaway inputs first:
            # autotuning re-runs the kernel per config, and some chunk
            # kernels mutate `v`/state in place, so a cold first call
            # returns corrupted output.
            for _ in range(2):
                inputs = make_prefill_inputs(2, 128, shape, seed=19)
                built, unavailable = KDA_PREFILL.build(inputs, shape, only=[name])
                if name not in built:
                    raise RuntimeError(unavailable[name])
                result = built[name]()
            if name in ("sglang_kda_chunk", "trtllm_kda_chunk", "trtllm_kda_cute"):
                states[name] = inputs.state
            elif name == "flashinfer_cake_kda":
                # CAKE returns (out, state) where state is BF16 [B,H,V,K];
                # ref state is FP32 [B,H,V,K]; transpose to [B,H,K,V] not needed
                # (CAKE already returns [B, H, V, K] which is the ref layout)
                states[name] = result[1]
            elif name == "b10_kda_chunk_prefill":
                # returns S_T as [B,H,K,V]; the reference state is [B,H,V,K]
                states[name] = result[1].transpose(-1, -2)
            else:
                states[name] = result[1]
            return result[0] if isinstance(result, tuple) else result

        return run

    all_rows: list[dict] = []
    for gate, group, lower_bound in (
        ("canonical", CANONICAL_GATE_CHECKED, None),
        ("safe", SAFE_GATE_CHECKED, _SAFE_GATE_LOWER_BOUND),
    ):
        ref_out, ref_state = reference(lower_bound)
        rows = check_impls(
            {
                n: make_fresh_input_runner(n)
                for n in group
                if n in KDA_PREFILL.builders
            },
            ref_out,
            lambda name, out: out,
            atol=5e-2,
            rtol=5e-2,
            notes=KDA_PREFILL.notes,
        )
        for row in rows:
            row["gate"] = gate
            state = states.get(row["impl"])
            if state is not None and "error" not in row:
                row["state_max_abs"] = (
                    (state.float() - ref_state.float()).abs().max().item()
                )
        all_rows.extend(rows)
    print_table(
        all_rows,
        columns=[
            "impl", "gate", "pass", "max_abs", "cosine", "state_max_abs",
            "error", "note",
        ],
        title="CORRECTNESS vs exact fp32 recurrence (B=2 S=128, matched-gate "
        "reference per row)",
    )


def run_prefill(args, shape: Shape) -> list[dict]:
    all_rows: list[dict] = []
    failed: set[str] = set()
    shown: set[str] = set()
    for batch in args.batch_sizes:
        for seq_len in args.seq_lens:
            try:
                inputs = make_prefill_inputs(
                    batch, seq_len, shape, seed=batch + seq_len
                )
            except torch.OutOfMemoryError:
                print(f"  skip B={batch} S={seq_len}: inputs exceed GPU memory")
                torch.cuda.empty_cache()
                continue
            wanted = args.backends or list(KDA_PREFILL.builders)
            safe = [n for n in wanted if baseline_is_safe(n, batch, seq_len, heads=shape.value_heads)]
            for name in set(wanted) - set(safe):
                print(
                    f"  skip {name} B={batch} S={seq_len}: known IMA (int32 overflow)"
                )
            runners, unavailable = get_kda_prefill_backends(
                inputs, shape, only=safe
            )
            # don't retry an impl that already failed on a previous shape
            runners = {n: f for n, f in runners.items() if n not in failed}
            for name, reason in unavailable.items():
                if name not in shown:
                    print(f"  unavailable {name}: {reason}")
                    shown.add(name)

            tokens = batch * seq_len

            def row_extra(name: str, latency_us: float) -> dict:
                return {
                    "mode": "kda_prefill",
                    "heads": shape.value_heads,
                    "batch": batch,
                    "seq_len": seq_len,
                    "tokens_s": tokens * 1e6 / latency_us,
                    "key_dim": shape.key_dim,
                    "value_dim": shape.value_dim,
                    "state_dtype": shape.state_dtype,
                }

            rows = bench_impls(
                runners,
                args.warmup,
                args.iters,
                args.repeats,
                row_extra=row_extra,
                notes=KDA_PREFILL.notes,
            )
            # OOM is shape-specific (retry the impl on smaller shapes);
            # any other failure disables the impl for the rest of the sweep
            failed.update(
                r["impl"]
                for r in rows
                if r.get("error") and "OutOfMemoryError" not in r["error"]
            )
            print_table(
                rows,
                columns=["impl", "latency_us", "tokens_s", "error", "note"],
                title=f"KDA PREFILL H={shape.value_heads} B={batch} S={seq_len} (us/iter)",
            )
            all_rows.extend(r for r in rows if "latency_us" in r)
            del inputs, runners
            torch.cuda.empty_cache()
    return all_rows


def run_stages(args, shape: Shape) -> list[dict]:
    """Split SGLang's chunk KDA into gate / WY / state / output timings."""
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from sglang.srt.layers.attention.fla.chunk_intra import chunk_kda_fwd_intra
    from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
    from sglang.srt.layers.attention.fla.kda import (
        chunk_gla_fwd_o_gk,
        kda_gate_chunk_cumsum,
    )

    all_rows: list[dict] = []
    chunk_size = 64
    for batch in args.batch_sizes:
        for seq_len in args.seq_lens:
            inputs = make_prefill_inputs(batch, seq_len, shape, seed=batch + seq_len)
            q, k, v = inputs.q, inputs.k, inputs.v
            scale = shape.key_dim**-0.5
            chunk_indices = prepare_chunk_indices(inputs.cu_seqlens, chunk_size)

            def gate():
                return kda_gate_chunk_cumsum(
                    inputs.raw_gate,
                    A_log=inputs.A_log,
                    chunk_size=chunk_size,
                    dt_bias=inputs.dt_bias,
                    cu_seqlens=inputs.cu_seqlens,
                    chunk_indices=chunk_indices,
                )

            # Materialize intermediates once so later stages are timed alone.
            g = gate()
            _NT = chunk_indices.shape[0]
            _small = batch * _NT * shape.qk_heads <= 256

            def wy():
                return chunk_kda_fwd_intra(
                    q=q,
                    k=k,
                    v=v,
                    gk=g,
                    beta=inputs.beta,
                    scale=scale,
                    cu_seqlens=inputs.cu_seqlens,
                    chunk_size=chunk_size,
                    chunk_indices=chunk_indices,
                    fuse_diagonal=_small,
                    fuse_recompute=_small,
                )

            w, u, _, kg, Aqk, _ = wy()
            state0 = inputs.state.clone()

            def state():
                inputs.state.copy_(state0)
                return chunk_gated_delta_rule_fwd_h(
                    k=kg,
                    w=w,
                    u=u,
                    gk=g,
                    initial_state=inputs.state,
                    initial_state_indices=inputs.state_indices,
                    cu_seqlens=inputs.cu_seqlens,
                    chunk_indices=chunk_indices,
                )

            h, v_new = state()
            o_buf = torch.empty_like(v)

            def output():
                return chunk_gla_fwd_o_gk(
                    q=q,
                    v=v_new,
                    g=g,
                    A=Aqk,
                    h=h,
                    o=o_buf,
                    scale=scale,
                    chunk_size=chunk_size,
                    cu_seqlens=inputs.cu_seqlens,
                    chunk_indices=chunk_indices,
                )

            tokens = batch * seq_len

            def row_extra(name: str, latency_us: float) -> dict:
                return {
                    "mode": "kda_stages",
                    "heads": shape.value_heads,
                    "batch": batch,
                    "seq_len": seq_len,
                    "tokens_s": tokens * 1e6 / latency_us,
                    "key_dim": shape.key_dim,
                    "value_dim": shape.value_dim,
                    "state_dtype": shape.state_dtype,
                }

            rows = bench_impls(
                {
                    "gate_cumsum": gate,
                    "wy_prep": wy,
                    "state_carry": state,
                    "output": output,
                },
                args.warmup,
                args.iters,
                args.repeats,
                row_extra=row_extra,
            )
            print_table(
                rows,
                columns=["impl", "latency_us", "tokens_s", "error"],
                title=f"KDA STAGES H={shape.value_heads} B={batch} S={seq_len} "
                "(SGLang leaf kernels, us)",
            )
            all_rows.extend(r for r in rows if "latency_us" in r)
    return all_rows


def run_sweep_group(
    heads: int, batch: int, seq: int, backends: list[str], out_csv: Path
) -> list[dict]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode", "prefill",
        "--skip-correctness",
        "--heads", str(heads),
        "--batch-sizes", str(batch),
        "--seq-lens", str(seq),
        "--backends", *backends,
        "--csv", str(out_csv),
        "--warmup", "2", "--iters", "5", "--repeats", "3",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT H={heads} B={batch} S={seq} {backends}")
        return []
    if proc.returncode != 0 and not out_csv.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:]
        print(
            f"  CRASH H={heads} B={batch} S={seq} {backends}: "
            f"{tail[0] if tail else '?'}"
        )
    for line in proc.stdout.splitlines():
        if "skip" in line or "unavailable" in line:
            print(f"  [H={heads} B={batch} S={seq}]{line}")
    if not out_csv.exists():
        return []
    with out_csv.open() as fh:
        return list(csv.DictReader(fh))


def run_sweep(args) -> None:
    """Full-grid sweep with per-shape subprocess isolation (see module doc)."""
    batches = args.batch_sizes or SWEEP_BATCHES
    seq_lens = args.seq_lens or SWEEP_SEQ_LENS
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for heads in args.heads:
            for batch in batches:
                for seq in seq_lens:
                    groups = [SWEEP_SAFE_BACKENDS] + [
                        [b] for b in SWEEP_ISOLATED_BACKENDS
                    ]
                    for group in groups:
                        out = tmpdir / f"h{heads}_b{batch}_s{seq}_{group[0]}.csv"
                        got = run_sweep_group(heads, batch, seq, group, out)
                        rows.extend(got)
                        for row in got:
                            print(
                                f"{row['impl']:>28} H={heads:3d} {batch:5d} "
                                f"{seq:8d} "
                                f"{float(row['latency_us']) / 1000:12.3f} ms"
                            )

    if not rows:
        print("no results")
        return
    if args.append and args.csv.exists():
        new_keys = {
            (r["impl"], r.get("heads"), r["batch"], r["seq_len"]) for r in rows
        }
        with args.csv.open() as fh:
            old = [
                r
                for r in csv.DictReader(fh)
                if (
                    r.get("impl", r.get("backend")),
                    r.get("heads"),
                    r["batch"],
                    r["seq_len"],
                )
                not in new_keys
            ]
        rows = old + rows
        rows.sort(
            key=lambda r: (
                int(r.get("heads") or 0),
                int(r["batch"]),
                int(r["seq_len"]),
                r.get("impl", r.get("backend", "")),
            )
        )
    write_csv(rows, args.csv)
    print(f"\nwrote {args.csv}")

    for row in rows:  # plot_rows expects numeric fields
        for key in ("batch", "seq_len"):
            row[key] = int(row[key])
        for key in ("latency_us", "tokens_s"):
            row[key] = float(row[key])
        if row.get("heads"):
            row["heads"] = int(row["heads"])
    for row in rows:
        row.setdefault("impl", row.get("backend"))
        row["panel"] = f"prefill H={row.get('heads', '')} B={row['batch']}"
    figure = plot_rows(
        rows,
        args.figure or args.csv.with_suffix(".png"),
        x="seq_len",
        y="tokens_s",
        series="impl",
        panel="panel",
        suptitle="KDA prefill: FLA vs FlashKDA vs PTX vs SGLang vs Triton port",
    )
    print(f"wrote {figure}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("prefill", "stages", "sweep"), default="prefill")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=None)
    parser.add_argument(
        "--heads",
        nargs="+",
        type=int,
        default=[12, 96],
        help="per-rank head counts to sweep; defaults are Kimi K3's TP8 "
        "shard (12) and TP1/PP shard (96)",
    )
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument(
        "--state-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backends", nargs="*")
    parser.add_argument(
        "--csv", type=Path, default=Path("results/bench_kda_prefill.csv")
    )
    parser.add_argument(
        "--figure",
        type=Path,
        help="output PNG path (default: CSV path with .png suffix)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="sweep mode: merge with existing CSV rows (new rows win per shape)",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "sweep":
        run_sweep(args)
        return
    if args.batch_sizes is None:
        args.batch_sizes = [1, 4]
    if args.seq_lens is None:
        args.seq_lens = [512, 2048, 8192]
    rows: list[dict] = []
    for heads in args.heads:
        shape = Shape(
            heads,
            heads,
            args.key_dim,
            args.value_dim,
            args.state_dtype,
        )
        print(f"\n===== H={heads} (K=V={args.key_dim}) =====")
        if not args.skip_correctness:
            check_correctness(shape)
        if args.mode == "prefill":
            rows.extend(run_prefill(args, shape))
        else:
            rows.extend(run_stages(args, shape))
    if rows:
        write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")
        # prefill mode plots throughput; stages mode plots per-stage latency
        y = "tokens_s" if args.mode == "prefill" else "latency_us"
        for row in rows:
            row["panel"] = f"{args.mode} H={row['heads']} B={row['batch']}"
        figure = plot_rows(
            rows,
            args.figure or args.csv.with_suffix(".png"),
            x="seq_len",
            y=y,
            series="impl",
            panel="panel",
            suptitle="KDA prefill: FLA vs SGLang vs vLLM vs TRT-LLM vs FlashKDA vs PTX",
        )
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
