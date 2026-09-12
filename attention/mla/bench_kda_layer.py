#!/usr/bin/env python3
"""Standalone K3 KDA (linear-attention) layer decode bench, TP8.

Constructs the PRODUCTION KimiDeltaAttention module (same flags as
serving) with duck-typed metadata/cache stubs, dummy weights, and times
CUDA-graphed decode forwards. Alignment target: the serving trace's
KDA-layer kernel inventory (mla_layer/RESEARCH.md).

Run: mpirun -n 8 --allow-run-as-root python3 mla_layer/bench_kda_layer.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1,4,8,16")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--profile", default="")
    parser.add_argument(
        "--weight-sets", type=int, default=8,
        help="rotate N weight replicas inside the graph so memory-bound "
             "kernels run L2-cold like 92-layer serving (1 = warm)")
    args = parser.parse_args()

    # serving parity: AUTO resolves to NCCL-symk in this harness, but
    # serving runs the trt fused AR — pin ONESHOT (aligned 2026-08-24).
    os.environ.setdefault("BENCH_ALLREDUCE_STRATEGY", "ONESHOT")
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    if world > 1:
        import torch.distributed as dist
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29551")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl", rank=rank, world_size=world,
            device_id=torch.device("cuda", rank))

    from kimi_k3.b10_kimi_k3_moe_layer import k3_model_config
    from tensorrt_llm._torch.modules.mamba.kda_mixer import KimiDeltaAttention

    if os.environ.get("K3_KDA_Z", "0") == "1":
        # grid.z value-tile parallel decode kernel (bench-only
        # monkeypatch; the module-level name in kda_mixer is rebound).
        import tensorrt_llm._torch.modules.mamba.kda_mixer as _mixer
        from mla_layer.kernels.fused_kda_decode_z import (
            fused_kda_decode as _kda_z)
        _mixer.fused_kda_decode = _kda_z
        if rank == 0:
            print("[bench] K3_KDA_Z=1: grid.z kda decode kernel patched",
                  flush=True)

    config = k3_model_config(rank, world)
    pcfg = config.pretrained_config
    # the MoE bench's pretrained stub lacks the attention-side fields:
    # graft the REAL text_config values from the HF json.
    import json as _json
    _tc = _json.load(open("/node-storage/var/kimi-k3-config/config.json")
                     )["text_config"]
    for k, v in _tc.items():
        if not hasattr(pcfg, k):
            setattr(pcfg, k, v)
    gen = torch.Generator(device="cuda").manual_seed(7 + rank)

    def build_kda(layer_idx: int):
        m = KimiDeltaAttention(config, layer_idx=layer_idx).cuda()
        with torch.no_grad():
            for name, p in m.named_parameters():
                if p.dtype.is_floating_point:
                    if "norm" in name:
                        p.fill_(1.0)
                    else:
                        p.normal_(0.0, 0.02, generator=gen)
        return m

    # weight-set rotation defeats single-layer L2 residency (the 126 MB
    # L2 keeps one layer's 115 MB of KDA weights warm; serving streams
    # 69 different KDA layers per step).
    kdas = [build_kda(1) for _ in range(max(1, args.weight_sets))]
    kda = kdas[0]

    la = pcfg.linear_attn_config
    n_heads_local = la["num_heads"] // world
    head_dim = la["head_dim"]
    conv_k = la["short_conv_kernel_size"]
    proj_local = n_heads_local * head_dim
    MAX_SLOTS = 64

    conv_states = torch.zeros(MAX_SLOTS, 3 * proj_local, conv_k - 1,
                              device="cuda", dtype=torch.bfloat16)
    ssm_states = torch.zeros(MAX_SLOTS, n_heads_local, head_dim, head_dim,
                             device="cuda", dtype=torch.float32)

    cache_mgr = SimpleNamespace(
        get_conv_states=lambda idx: conv_states,
        get_ssm_states=lambda idx: ssm_states,
        is_speculative=lambda: False,
        use_replay_state_update=False,
        get_replay_state_update_metadata=lambda: None,
        get_prefill_snapshot_write_plan=lambda: None,
        get_prefill_ssm_snapshot_states=lambda idx: None,
        mamba_layer_cache=lambda idx: None,
        get_mamba_ssm_rand_seed=lambda: torch.zeros(
            2, dtype=torch.int64, device="cuda"),
    )

    def meta_for(batch: int):
        attn_md = SimpleNamespace(
            num_contexts=0,
            seq_lens=torch.full((batch,), 128, dtype=torch.int32),
            num_ctx_tokens=0,
            num_tokens=batch,
            kv_cache_manager=cache_mgr,
        )
        mamba_md = SimpleNamespace(
            state_indices=torch.arange(batch, dtype=torch.int32,
                                       device="cuda"),
            has_initial_states=torch.ones(batch, dtype=torch.bool,
                                          device="cuda"),
            num_new_prefill_states=0,
            new_prefill_state_indices=torch.empty(
                0, dtype=torch.int32, device="cuda"),
        )
        return attn_md, mamba_md

    sizes = tuple(int(s) for s in args.sizes.split(","))
    profile_sizes = set(int(s) for s in args.profile.split(",")) \
        if args.profile else set()
    for batch in sizes:
        attn_md, mamba_md = meta_for(batch)
        gen_in = torch.Generator(device="cuda").manual_seed(batch)
        h = torch.randn(batch, pcfg.hidden_size, generator=gen_in,
                        device="cuda").bfloat16()

        calls = [0]

        def fn():
            m = kdas[calls[0] % len(kdas)]
            calls[0] += 1
            with torch.no_grad():
                return m(h, attn_md, mamba_md)

        out = fn()  # smoke + shape
        assert out.shape == (batch, pcfg.hidden_size), out.shape
        assert torch.isfinite(out).all().item(), "non-finite"
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.iters):
                fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        start.record()
        for _ in range(5):
            graph.replay()
        stop.record()
        stop.synchronize()
        us = start.elapsed_time(stop) * 1000 / (5 * args.iters)
        val = torch.tensor([us], device="cuda", dtype=torch.float64)
        if world > 1:
            dist.all_reduce(val, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"KDA decode B={batch:>3}  {val.item():7.2f} us/layer "
                  f"(serving KDA-side ≈ 47 us/lyr at bs8 incl AR)",
                  flush=True)
        if batch in profile_sizes:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as prof:
                graph.replay()
                torch.cuda.synchronize()
            path = Path(__file__).resolve().parents[1] / "results" / "legacy" / (
                f"kda_tp{world}_b{batch}_graph.trace.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            if rank == 0:
                prof.export_chrome_trace(str(path))
                print(f"[profile] wrote {path}", flush=True)
    if world > 1:
        dist.barrier()


if __name__ == "__main__":
    main()

