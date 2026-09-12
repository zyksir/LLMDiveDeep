"""Compare sparse ATTENTION CORES on identical caches and selected IDs.

Compression, indexing, top-k, paging, and mask construction are setup, NOT
timed. This is not a model benchmark or a claim of total CSA pipeline speed.
"""

import argparse

import torch

from .backends import BackendUnavailable
from .benchmark_utils import (
    environment,
    runtime_device,
    time_forward,
    timing_arguments,
    write_rows,
)
from .sparse_attn.backends import BACKENDS, prepare_sparse
from .sparse_attn.compressed_sparse_attention import (
    combined_selection,
    dense_selected_attention,
    gated_compress,
    indexer_topk,
    selection_mask,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    timing_arguments(parser, "compressed_sparse")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=["torch_gather", "sglang_flashmla", "flashmla", "flashinfer_dsv4"],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--ratio", type=int, choices=(1, 2, 4, 128), default=4)
    parser.add_argument("--selection", choices=("topk", "all"), default="topk")
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--no-sink", action="store_true")
    parser.add_argument("--max-reference-elements", type=int, default=16_777_216)
    args = parser.parse_args()
    if args.list:
        print("\n".join(BACKENDS))
        return
    if not args.run:
        parser.print_help()
        return
    if (
        min(
            args.batch,
            args.seq_len,
            args.q_len,
            args.heads,
            args.head_dim,
            args.topk,
            args.window,
            args.max_reference_elements,
        )
        < 1
    ):
        parser.error(
            "shape dimensions, topk, window and oracle budget must be positive"
        )
    if args.q_len > args.seq_len or args.seq_len < args.ratio:
        parser.error("require Q <= T and at least one complete compression group")
    device = runtime_device(args)
    dtype = getattr(torch, args.dtype)
    b, t, nq, h, d = args.batch, args.seq_len, args.q_len, args.heads, args.head_dim
    # Synthetic kernel inputs; no checkpoint, projection, or model loaded.
    q = torch.randn(b, nq, h, d, device=device, dtype=dtype)
    raw = torch.randn(b, t, d, device=device, dtype=dtype)
    width = d * (2 if args.ratio == 4 else 1)
    values = torch.randn(b, t, width, device=device, dtype=dtype)
    logits = torch.randn_like(values)
    compressed = gated_compress(values, logits, args.ratio, overlap=args.ratio == 4).to(
        dtype
    )
    del values, logits
    n = compressed.shape[1]
    positions = torch.arange(t - nq, t, device=device)
    if args.selection == "all":
        global_ids = torch.arange(n, device=device, dtype=torch.int32)[
            None, None
        ].expand(b, nq, n)
    else:
        # Small unquantized indexer for setup; not the model's calibrated indexer.
        qi = torch.randn(b, nq, 4, 32, device=device, dtype=dtype)
        ki = torch.randn(b, n, 32, device=device, dtype=dtype)
        wi = torch.randn(b, nq, 4, device=device) / (32 * 4) ** 0.5
        global_ids = indexer_topk(
            qi, ki, wi, positions, ratio=args.ratio, topk=args.topk
        )
        del qi, ki, wi
    ids = combined_selection(
        t, positions, global_ids, window=args.window, ratio=args.ratio
    )
    sink = None if args.no_sink else torch.zeros(h, device=device)
    ncheck = min(nq, 8, args.max_reference_elements // (b * h * (t + n)))
    if ncheck < 1:
        parser.error("one oracle query exceeds --max-reference-elements")
    cache = torch.cat((raw, compressed), dim=1)
    oracle_mask = selection_mask(ids[:, -ncheck:], t + n)
    with torch.inference_mode():
        expected = dense_selected_attention(
            q[:, -ncheck:], cache, oracle_mask, sink=sink
        )
    del cache, oracle_mask
    metadata = environment(device)
    rows, failed = [], False
    for name in args.backends:
        row = dict(
            backend=name,
            batch=b,
            q_len=nq,
            kv_len=t,
            heads=h,
            head_dim=d,
            ratio=args.ratio,
            selection=args.selection,
            selected_capacity=ids.shape[-1],
            window=args.window,
            sink=not args.no_sink,
            dtype=args.dtype,
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
                name == "torch_dense_mask"
                and b * h * nq * (t + n) > args.max_reference_elements
            ):
                raise BackendUnavailable(
                    "dense score matrix exceeds --max-reference-elements"
                )
            prepared = prepare_sparse(
                name, q, raw, compressed, ids, window=args.window, sink=sink
            )
            row.update(
                implementation=prepared.implementation, boundary=prepared.boundary
            )
            with torch.inference_mode():
                actual = prepared.run()
                if actual.shape != q.shape or not torch.isfinite(actual).all():
                    raise AssertionError("wrong output shape or nonfinite values")
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
                    (actual[:, -ncheck:].float() - expected.float()).abs().max().item()
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
        print(f"{name:22} {row['status']} {row['us']} {row['error']}")
    write_rows(rows, args.csv)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
