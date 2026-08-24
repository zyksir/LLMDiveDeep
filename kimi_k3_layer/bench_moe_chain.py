#!/usr/bin/env python3
"""Four-layer chained MoE bench: serving-realistic skew + overlap room.

Single-layer benches under-model serving twice: (1) rank skew from
imbalanced experts propagates ACROSS layers (layer i's slow rank enters
layer i+1 late), (2) the attention residue between MoE blocks offers
overlap opportunities a lone layer cannot express. This bench chains
N (default 4) MoE layers with an attention-residue proxy between them:

    a  = rms_norm(h);  attn = a @ W_attn_i  (distinct weights/position)
    h  = h + attn                            (attention residual)
    m  = rms_norm(h);  moe = layer_i(m)
    h  = h + moe                             (MoE residual)

Modes (--arm):
    stock      KimiK3StockMoE chain (production path)
    b10        B10 deploy chain
    b10_fused  B10 chain with the MoE residual+norm FUSED into the tail
               AR (comm.allreduce_norm(residual=...)): the "overlap the
               attention residue with the all-reduce" idea — the next
               position's input norm and the residual add ride the AR
               kernel instead of following it.

Expert skew: set K3_BENCH_EXPERT_IMBALANCE=2.0 (see
b10_kimi_k3_moe_layer._bench_doctor_logits) for the serving-measured
imbalance. Distinct per-layer weights also defeat single-layer L2
residency.

mpirun -n 8 --allow-run-as-root python3 kimi_k3_layer/bench_moe_chain.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--sizes", default="8")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--arms", default="stock,b10,b10_fused")
    args = parser.parse_args()

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29556")
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank,
                            world_size=world,
                            device_id=torch.device("cuda", rank))

    from tensorrt_llm._torch.utils import AuxStreamType
    from kimi_k3_layer.bench_b10_kimi_k3_moe_layer import (
        _build_collectives, init_weights)
    from kimi_k3_layer.b10_kimi_k3_moe_layer import (
        B10KimiK3MoELayer, KimiK3StockMoE, LayerMode, k3_model_config)
    from kimi_k3_layer.config import HIDDEN, MOE_LATENT

    sizes = tuple(int(s) for s in args.sizes.split(","))
    collectives = _build_collectives(world, sizes, max(sizes))
    config = k3_model_config(rank, world)
    # rank-IDENTICAL: the proxy weights feed TP collectives; rank-dependent
    # weights would desynchronize the activations (bug caught 2026-08-24).
    gen = torch.Generator(device="cuda").manual_seed(11)

    def build_chain(cls):
        chain = []
        for i in range(args.layers):
            aux = {k: torch.cuda.Stream() for k in AuxStreamType}
            layer = cls(config, layer_idx=0, aux_stream_dict=aux,
                        reduce_output=world > 1, collectives=collectives,
                        mode=LayerMode("deploy")).cuda()
            init_weights(layer, world, rank, seed=100 + i)
            layer.init_optimized(max_batch=max(sizes),
                                 collectives=collectives)
            chain.append(layer)
        return chain

    # attention-residue proxy weights: distinct per position, sized to
    # the KDA in_proj byte cost (~90 MB -> ~19 us serving-honest)
    attn_w = [
        (torch.randn(HIDDEN, HIDDEN, generator=gen,
                     device="cuda") * 0.02).bfloat16().t().contiguous()
        for _ in range(args.layers)
    ]
    norm_w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
    eps = 1e-5

    arms = {}
    if "stock" in args.arms:
        arms["stock"] = (build_chain(KimiK3StockMoE), False)
    if "b10" in args.arms.replace("b10_fused", ""):
        arms["b10"] = (build_chain(B10KimiK3MoELayer), False)
    if "b10_fused" in args.arms:
        arms["b10_fused"] = (build_chain(B10KimiK3MoELayer), True)

    for tokens in sizes:
        gen_in = torch.Generator(device="cuda").manual_seed(tokens)
        h0 = torch.randn(tokens, HIDDEN, generator=gen_in,
                         device="cuda").bfloat16()
        results = {}
        for name, (chain, fuse_residual) in arms.items():
            def block(h):
                # a = input-normed h, maintained across positions so the
                # fused arm can source it from the previous tail AR.
                a = torch.nn.functional.rms_norm(
                    h.float(), (HIDDEN,), norm_w.float(), eps).bfloat16()
                for i, layer in enumerate(chain):
                    h = h + a @ attn_w[i]
                    m = torch.nn.functional.rms_norm(
                        h.float(), (HIDDEN,), norm_w.float(), eps
                    ).bfloat16()
                    if fuse_residual:
                        # the overlap idea: the MoE output AR carries the
                        # residual add AND the next block's input norm in
                        # its fused epilogue (trt allreduce_norm), so the
                        # residue work rides the AR instead of following
                        # it as two elementwise kernels.
                        a, h = layer(m, _fused_residual=(h, norm_w))
                        a = a.view(-1, HIDDEN)
                        h = h.view(-1, HIDDEN)
                    else:
                        h = h + layer(m)
                        a = torch.nn.functional.rms_norm(
                            h.float(), (HIDDEN,), norm_w.float(), eps
                        ).bfloat16()
                return h

            with torch.no_grad():
                out = block(h0)
            assert torch.isfinite(out).all().item(), name
            if name == "stock":
                ref_out = out.float().clone()
                b10_out = None
            elif rank == 0:
                if name == "b10":
                    b10_out = out.float().clone()
                if name == "b10_fused" and b10_out is not None:
                    r2 = ((out.float() - b10_out).abs().max()
                          / (b10_out.abs().max() + 1e-6)).item()
                    print(f"  [b10_fused] vs b10 rel={r2:.4f}"
                          f"{'' if r2 < 0.05 else '  FUSED-MISMATCH'}",
                          flush=True)
                rel = ((out.float() - ref_out).abs().max()
                       / (ref_out.abs().max() + 1e-6)).item()
                flag = "" if rel < 0.05 else "  ARM-MISMATCH"
                print(f"  [{name}] vs stock rel={rel:.4f}{flag}",
                      flush=True)
            for _ in range(3):
                with torch.no_grad():
                    block(h0)
            torch.cuda.synchronize()
            dist.barrier()
            from tensorrt_llm._torch.autotuner import autotune
            from tensorrt_llm._torch.modules.multi_stream_utils import (
                with_multi_stream)
            graph = torch.cuda.CUDAGraph()
            with with_multi_stream(True):
                with autotune():
                    with torch.no_grad():
                        block(h0)
                torch.cuda.synchronize()
                dist.barrier()
                with torch.cuda.graph(graph):
                    for _ in range(args.iters):
                        with torch.no_grad():
                            h = block(h0)
            torch.cuda.synchronize()
            dist.barrier()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            for _ in range(2):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            t0.record()
            for _ in range(5):
                graph.replay()
            t1.record()
            t1.synchronize()
            us = t0.elapsed_time(t1) * 1000 / (5 * args.iters * args.layers)
            v = torch.tensor([us], device="cuda", dtype=torch.float64)
            dist.all_reduce(v, op=dist.ReduceOp.MAX)
            results[name] = v.item()
            del graph
        if rank == 0:
            base = results.get("stock")
            row = f"B={tokens:>4} chain({args.layers})"
            for name, us in results.items():
                gain = f" ({100*(base-us)/base:+.1f}%)" if base and name != "stock" else ""
                row += f"  {name}={us:7.2f}us/lyr{gain}"
            print(row, flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
