"""Small CPU trace and analytical cache/work counts; no checkpoint download."""

import argparse

import torch

from .config import global_cache_bytes, index_positions_per_decode, layer_spec
from .csa2_reference import (
    CSA2Router,
    gated_compress,
    joint_sparse_attention,
    window_indices,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--query", type=int, default=None, help="zero-based position to trace"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--schedule", action="store_true")
    args = parser.parse_args()
    if args.seq_len < 1 or args.seq_len > 4096:
        parser.error("this toy trace supports sequence lengths 1..4096")
    query = args.seq_len - 1 if args.query is None else args.query
    if not 0 <= query < args.seq_len:
        parser.error("query must be inside the sequence")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if args.schedule:
        print("layer stage   mode     ratio KV-owner index-owner")
        for i in range(40):
            s = layer_spec(i)
            print(
                f"{i:5} {s.stage:7} {s.mode:7} {s.ratio:5} {str(s.kv_owner):>8} {str(s.index_owner):>11}"
            )
    # Toy dimensions expose the same schedule without allocating the model.
    b, h, d, hi, di = 1, 2, 8, 2, 4
    positions = torch.arange(args.seq_len)
    router = CSA2Router(positions, topk=3, candidate_blocks=2, block_size=4)
    local_ids = window_indices(positions, window=4, batch_size=b)
    print(f"\nToy trace at query {query}: window=4, top-k=3, candidate pool<=8")
    for i in range(40):
        spec = layer_spec(i)
        kwargs = {}
        if spec.mode == "Full":
            v, gates = torch.randn(b, args.seq_len, d), torch.randn(b, args.seq_len, d)
            latent = gated_compress(v, gates, spec.ratio)
            latent = latent / (latent.square().mean(-1, keepdim=True) + 1e-6).sqrt()
            # Stand-in projected keys; this demo omits RoPE and quantization.
            kwargs.update(main_kv=latent, index_k=latent @ torch.randn(d, di))
        if spec.mode in ("Full", "Reindex"):
            kwargs.update(
                q=torch.randn(b, args.seq_len, hi, di),
                weights=torch.randn(b, args.seq_len, hi) / (hi * di) ** 0.5,
            )
        routed = router.route(i, **kwargs)
        q = torch.randn(b, args.seq_len, h, d)
        local_kv = torch.randn(b, args.seq_len, d)
        empty = torch.empty(b, 0, d)
        empty_ids = torch.empty(b, args.seq_len, 0, dtype=torch.int32)
        output = joint_sparse_attention(
            q,
            local_kv,
            routed.main_kv if routed else empty,
            local_ids,
            routed.indices if routed else empty_ids,
            torch.zeros(h),
        )
        ids = routed.indices[0, query].tolist() if routed else []
        print(
            f"L{i:02} {spec.mode:7} global={ids} output_norm={output[0, query].norm():.4f}"
        )
        if i == 20:
            print(
                "    shared decoder candidates:", router.candidates[0, query].tolist()
            )
    print("\nReleased-shape analytical counts (not benchmark results):")
    print("tokens     global-MiB   index-positions/decode")
    for n in (4096, 65536, 1048576):
        print(
            f"{n:7} {global_cache_bytes(n) / 2**20:14.4f} {index_positions_per_decode(n):24,}"
        )


if __name__ == "__main__":
    main()
