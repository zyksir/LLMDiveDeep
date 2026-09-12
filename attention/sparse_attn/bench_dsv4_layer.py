"""Benchmark DeepSeek-V4 CSA vs HCA attention layers.

Times the standalone blocks in `dsv4_layer.py`. The two questions:

    Q1. Decode (T=1): how does CSA (indexer + sparse over top-k compressed
        entries) compare with HCA (dense over all S/128 compressed entries)
        as context S grows?
    Q2. Prefill: same comparison with the sliding-window branch included,
        at a width that fits in one GPU (`tiny` by default).

Neither path loads weights or a server. Decode history is a synthetic
compressed cache of the right length (`floor(S / m)` rows); that is the
tensor volume a real decode step would read.

    python bench_dsv4_layer.py                  # decode + prefill, flash+tiny
    python bench_dsv4_layer.py --decode         # Q1 only
    python bench_dsv4_layer.py --prefill --spec tiny
    python bench_dsv4_layer.py --check          # shapes / causal / finite
    python bench_dsv4_layer.py --spec pro --S 4096 16384 65536
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_LLMDIVEDEEP = os.path.dirname(_HERE)
if _LLMDIVEDEEP not in sys.path:
    sys.path.insert(0, _LLMDIVEDEEP)

from common.kernel_bench import (  # noqa: E402
    bench_cuda,
    print_section,
    print_table,
    report,
)
from dsv4_layer import SPECS, Dsv4Spec, LayerKind, build_layer  # noqa: E402


DECODE_KINDS: tuple[LayerKind, ...] = ("csa", "hca", "swa")
PREFILL_KINDS: tuple[LayerKind, ...] = ("csa", "hca", "swa", "dense")
DEFAULT_DECODE_S = (4096, 16384, 65536, 131072)
DEFAULT_PREFILL_S = (512, 1024, 2048, 4096)


def _ms(fn, warmup: int, iters: int) -> float:
    return bench_cuda(fn, warmup=warmup, iters=iters, repeats=3) / 1000.0


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _hidden(spec: Dsv4Spec, batch: int, seq: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        batch, seq, spec.hidden_size, generator=generator, device=device, dtype=torch.bfloat16
    )


def check_layers(spec: Dsv4Spec, seq_len: int, device: torch.device) -> list[dict]:
    """Shape / causal / finite checks. Not a CSA↔HCA value comparison."""
    rows = []
    hidden = _hidden(spec, 1, seq_len, device, seed=0)
    token = _hidden(spec, 1, 1, device, seed=1)
    for kind in PREFILL_KINDS:
        layer = build_layer(kind, spec, device=device)
        row: dict = {"kind": kind, "mode": "prefill"}
        try:
            out = layer(hidden)
            finite = torch.isfinite(out.hidden_states.float()).all().item()
            ok_shape = tuple(out.hidden_states.shape) == (1, seq_len, spec.hidden_size)
            row["shape_ok"] = ok_shape
            row["finite"] = bool(finite)
            row["n_compressed"] = out.n_compressed
            row["n_core_keys"] = out.n_core_keys
            if kind == "csa":
                expect = seq_len // spec.compress_rate_csa
                row["comp_ok"] = out.n_compressed == expect
                if out.topk_indices is not None and out.topk_indices.numel():
                    pos = torch.arange(seq_len, device=device).view(1, seq_len, 1)
                    future = (out.topk_indices >= 0) & (
                        out.topk_indices >= (pos + 1) // spec.compress_rate_csa
                    )
                    row["causal_ok"] = not bool(future.any().item())
                else:
                    row["causal_ok"] = True
            elif kind == "hca":
                row["comp_ok"] = out.n_compressed == seq_len // spec.compress_rate_hca
                row["causal_ok"] = True
            else:
                row["comp_ok"] = out.n_compressed == 0
                row["causal_ok"] = True
            row["pass"] = (
                row["shape_ok"] and row["finite"] and row["comp_ok"] and row["causal_ok"]
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["pass"] = False
        rows.append(row)

        if kind == "dense":
            continue
        row_d: dict = {"kind": kind, "mode": "decode"}
        try:
            cache = layer.build_decode_cache(seq_len, batch=1, device=device, seed=2)
            out = layer.decode(token, cache)
            finite = torch.isfinite(out.hidden_states.float()).all().item()
            row_d["shape_ok"] = tuple(out.hidden_states.shape) == (1, 1, spec.hidden_size)
            row_d["finite"] = bool(finite)
            row_d["n_compressed"] = out.n_compressed
            row_d["n_core_keys"] = out.n_core_keys
            if kind == "csa":
                row_d["comp_ok"] = out.n_compressed == seq_len // spec.compress_rate_csa
                row_d["causal_ok"] = True
                if out.topk_indices is not None and out.topk_indices.numel():
                    thresh = (seq_len + 1) // spec.compress_rate_csa
                    bad = (out.topk_indices >= 0) & (out.topk_indices >= thresh)
                    row_d["causal_ok"] = not bool(bad.any().item())
            elif kind == "hca":
                row_d["comp_ok"] = out.n_compressed == seq_len // spec.compress_rate_hca
                row_d["causal_ok"] = True
            else:
                row_d["comp_ok"] = out.n_compressed == 0
                row_d["causal_ok"] = True
            row_d["pass"] = (
                row_d["shape_ok"]
                and row_d["finite"]
                and row_d["comp_ok"]
                and row_d["causal_ok"]
            )
        except Exception as exc:  # noqa: BLE001
            row_d["error"] = f"{type(exc).__name__}: {exc}"
            row_d["pass"] = False
        rows.append(row_d)
    return rows


def bench_decode(
    spec: Dsv4Spec,
    seq_lens: list[int],
    batch: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> list[dict]:
    print_section(
        f"decode T=1  spec={spec.name}  "
        f"CSA m={spec.compress_rate_csa} topk={spec.index_topk}  "
        f"HCA m'={spec.compress_rate_hca}  SWA={spec.sliding_window}"
    )
    print(
        "  CSA core keys ≈ SWA + min(topk, S/m); "
        "HCA core keys ≈ SWA + S/m'; SWA core keys = window."
    )
    rows = []
    for seq_len in seq_lens:
        token = _hidden(spec, batch, 1, device, seed=seq_len)
        by_kind: dict[str, float] = {}
        for kind in DECODE_KINDS:
            layer = build_layer(kind, spec, device=device)
            cache = layer.build_decode_cache(seq_len, batch=batch, device=device)
            out = layer.decode(token, cache)
            timed = _ms(lambda: layer.decode(token, cache), warmup, iters)
            by_kind[kind] = timed
            indexer_ms = float("nan")
            if kind == "csa":
                q_lora = layer.q_a_norm(layer.q_a_proj(token))
                pos = torch.full((batch, 1), seq_len, device=device, dtype=torch.long)
                indexer_ms = _ms(
                    lambda: layer.indexer(
                        token, q_lora, compressed_kv=cache.indexer_kv, position_ids=pos
                    ),
                    warmup,
                    iters,
                )
            rows.append(
                {
                    "S": seq_len,
                    "kind": kind,
                    "ms": timed,
                    "indexer_ms": indexer_ms,
                    "n_compressed": out.n_compressed,
                    "n_core_keys": out.n_core_keys,
                }
            )
            del layer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        swa_ms = by_kind["swa"]
        for row in rows[-len(DECODE_KINDS) :]:
            row["vs_swa"] = row["ms"] / swa_ms if swa_ms > 0 else float("nan")
    return rows


def bench_prefill(
    spec: Dsv4Spec,
    seq_lens: list[int],
    batch: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> list[dict]:
    print_section(
        f"prefill T=S  spec={spec.name}  "
        f"heads={spec.num_heads} d={spec.head_dim} hidden={spec.hidden_size}"
    )
    rows = []
    for seq_len in seq_lens:
        hidden = _hidden(spec, batch, seq_len, device, seed=seq_len)
        by_kind: dict[str, float] = {}
        start = len(rows)
        for kind in PREFILL_KINDS:
            layer = build_layer(kind, spec, device=device)
            try:
                out = layer(hidden)
                timed = _ms(lambda: layer(hidden), warmup, iters)
            except RuntimeError as exc:
                rows.append(
                    {
                        "S": seq_len,
                        "kind": kind,
                        "error": str(exc).split("\n", 1)[0],
                    }
                )
                del layer
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            by_kind[kind] = timed
            compress_ms = float("nan")
            indexer_ms = float("nan")
            if kind == "csa":
                compress_ms = _ms(lambda: layer.csa_compressor(hidden), warmup, iters)
                q_lora = layer.q_a_norm(layer.q_a_proj(hidden))
                indexer_ms = _ms(lambda: layer.indexer(hidden, q_lora), warmup, iters)
            elif kind == "hca":
                compress_ms = _ms(lambda: layer.hca_compressor(hidden), warmup, iters)
            rows.append(
                {
                    "S": seq_len,
                    "kind": kind,
                    "ms": timed,
                    "compress_ms": compress_ms,
                    "indexer_ms": indexer_ms,
                    "n_compressed": out.n_compressed,
                    "n_core_keys": out.n_core_keys,
                }
            )
            del layer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dense_ms = by_kind.get("dense")
        if dense_ms:
            for row in rows[start:]:
                if "ms" in row:
                    row["vs_dense"] = dense_ms / row["ms"] if row["ms"] else float("nan")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--decode", action="store_true", help="Q1: T=1 decode sweep")
    parser.add_argument("--prefill", action="store_true", help="Q2: T=S prefill sweep")
    parser.add_argument("--check", action="store_true", help="shape / causal / finite only")
    parser.add_argument("--all", action="store_true", help="decode + prefill")
    parser.add_argument("--spec", choices=list(SPECS), default=None)
    parser.add_argument("--decode-spec", choices=list(SPECS), default="flash")
    parser.add_argument("--prefill-spec", choices=list(SPECS), default="tiny")
    parser.add_argument("--S", type=int, nargs="+", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if not (args.decode or args.prefill or args.check or args.all):
        args.all = True
    if args.all:
        args.decode = args.prefill = True
    if args.spec is not None:
        args.decode_spec = args.prefill_spec = args.spec

    device = _device()
    print("environment:")
    print(f"  {'torch':>20} : {torch.__version__}")
    print(f"  {'device':>20} : {device}")
    if device.type == "cuda":
        print(f"  {'gpu':>20} : {torch.cuda.get_device_name(0)}")
    print(f"  {'decode spec':>20} : {SPECS[args.decode_spec].name}")
    print(f"  {'prefill spec':>20} : {SPECS[args.prefill_spec].name}")
    if (args.decode or args.prefill) and device.type != "cuda":
        raise SystemExit("decode/prefill timing requires CUDA; use --check on CPU")

    all_rows: list[dict] = []

    if args.check:
        spec = SPECS[args.prefill_spec]
        seq_len = args.S[0] if args.S else spec.compress_rate_hca * 2
        print_section(f"checks  spec={spec.name}  S={seq_len}")
        rows = check_layers(spec, seq_len, device)
        print_table(rows)
        all_rows.extend(rows)
        if not all(row.get("pass") for row in rows):
            failed = [row for row in rows if not row.get("pass")]
            raise SystemExit(f"checks failed: {failed}")

    if args.decode:
        spec = SPECS[args.decode_spec]
        seq_lens = args.S or list(DEFAULT_DECODE_S)
        rows = bench_decode(spec, seq_lens, args.batch, args.warmup, args.iters, device)
        print_table(rows)
        all_rows.extend({"mode": "decode", **row} for row in rows)

    if args.prefill:
        spec = SPECS[args.prefill_spec]
        seq_lens = args.S or list(DEFAULT_PREFILL_S)
        rows = bench_prefill(spec, seq_lens, args.batch, args.warmup, args.iters, device)
        print_table(rows)
        all_rows.extend({"mode": "prefill", **row} for row in rows)

    if args.csv:
        report(all_rows, table=False, csv_path=args.csv)


if __name__ == "__main__":
    main()
