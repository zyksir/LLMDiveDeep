"""Matched forward-kernel benchmark. Requires --run; never loads a model."""

import argparse

import torch

from .backends import BACKENDS, BackendUnavailable, prepare_attention
from .benchmark_utils import (
    environment,
    runtime_device,
    time_forward,
    timing_arguments,
    write_rows,
)
from .dense_attention import matmul_attention


def main(windowed=False):
    parser = argparse.ArgumentParser(description=__doc__)
    timing_arguments(parser, "sliding_window" if windowed else "dense")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=["torch_eager", "torch_sdpa_math", "torch_sdpa_flash", "fa4"],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 2048])
    parser.add_argument(
        "--q-len",
        type=int,
        default=None,
        help="default: Q=KV; set 1 for decode-shaped kernel input",
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--noncausal", action="store_true")
    parser.add_argument(
        "--max-reference-elements",
        type=int,
        default=16_777_216,
        help="bound eager score matrices and the correctness oracle",
    )
    if windowed:
        parser.add_argument(
            "--window",
            type=int,
            default=128,
            help="number of visible tokens INCLUDING self",
        )
    args = parser.parse_args()
    if args.list:
        print("\n".join(BACKENDS))
        return
    if not args.run:
        parser.print_help()
        return
    if min(args.batch, args.heads, args.kv_heads, args.head_dim, *args.seq_lens) < 1:
        parser.error("shape dimensions must be positive")
    if args.heads % args.kv_heads or args.max_reference_elements < 1:
        parser.error(
            "heads must be divisible by kv-heads; oracle budget must be positive"
        )
    window = args.window if windowed else None
    if windowed and (args.noncausal or window < 1):
        parser.error("this window tutorial is causal and requires window >= 1")
    if args.q_len is not None and (
        args.q_len < 1 or any(args.q_len > n for n in args.seq_lens)
    ):
        parser.error("q-len must be positive and <= every KV length")
    device = runtime_device(args)
    dtype = getattr(torch, args.dtype)
    metadata = environment(device)
    rows, failed = [], False
    for nk in args.seq_lens:
        nq = args.q_len or nk
        q = torch.randn(
            args.batch, nq, args.heads, args.head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            args.batch, nk, args.kv_heads, args.head_dim, device=device, dtype=dtype
        )
        v = torch.randn_like(k)
        for name in args.backends:
            row = dict(
                backend=name,
                batch=args.batch,
                q_len=nq,
                kv_len=nk,
                heads=args.heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                dtype=args.dtype,
                causal=not args.noncausal,
                window=window,
                seed=args.seed,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                status="",
                us="",
                max_abs_error="",
                error="",
                **metadata,
            )
            try:
                if (
                    name == "torch_eager"
                    and args.batch * args.heads * nq * nk > args.max_reference_elements
                ):
                    raise BackendUnavailable(
                        "eager score matrix exceeds --max-reference-elements"
                    )
                ncheck = min(
                    nq, 8, args.max_reference_elements // (args.batch * args.heads * nk)
                )
                if ncheck < 1:
                    raise BackendUnavailable(
                        "even one oracle query exceeds the reference budget"
                    )
                prepared = prepare_attention(
                    name, q, k, v, causal=not args.noncausal, window=window
                )
                row.update(
                    implementation=prepared.implementation, boundary=prepared.boundary
                )
                with torch.inference_mode():
                    actual = prepared.run()  # compile/warm up outside timing
                    expected = matmul_attention(
                        q[:, -ncheck:], k, v, causal=not args.noncausal, window=window
                    )
                    if actual.shape != q.shape:
                        raise AssertionError(
                            f"wrong output shape: {actual.shape} != {q.shape}"
                        )
                    if not torch.isfinite(actual).all():
                        raise AssertionError("nonfinite output")
                    atol = (
                        0.03
                        if dtype == torch.bfloat16
                        else 0.003
                        if dtype == torch.float16
                        else 0.0001
                    )
                    torch.testing.assert_close(
                        actual[:, -ncheck:], expected, atol=atol, rtol=atol
                    )
                    row["max_abs_error"] = (
                        (actual[:, -ncheck:].float() - expected.float())
                        .abs()
                        .max()
                        .item()
                    )
                row["us"] = time_forward(
                    prepared.run,
                    device,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
                row["status"] = "PASS"
            except (BackendUnavailable, ImportError) as exc:
                row.update(status="SKIP", error=str(exc))
            except Exception as exc:
                row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
                failed = True
            rows.append(row)
            print(
                f"{name:22} Q={nq} KV={nk}: {row['status']} {row['us']} {row['error']}"
            )
    write_rows(rows, args.csv)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
