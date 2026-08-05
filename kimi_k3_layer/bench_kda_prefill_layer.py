#!/usr/bin/env python3
"""Kimi-K3 KDA PREFILL, layer level: production baseline vs b10 opt.

Single GPU, K3 TP8 shard (12 heads, head_dim 128, hidden 7168), one
sequence of S tokens (B=1 varlen layout, cu_seqlens=[0, S]).

baseline  the production (1.3.0rc19 @ 496bc01f8e) prefill flow and its
          two BLOCKING D2H syncs:
            (1) new-state reset via CUDA-bool boolean indexing
                (`state_indices[~has_initial_states]` -> internal
                nonzero() -> D2H copy of the match count);
            (2) the Triton chunk pipeline's `prepare_chunk_indices`
                (`.tolist()` on cu_seqlens-derived lengths; fires per
                distinct cu_seqlens - emulated per call here by
                clearing its @tensor_cache, which is what a serving
                engine sees across varying batch shapes).
          Kernels: in_proj GEMM -> causal_conv1d_fn -> TRT-vendored
          FLA `chunk_kda_with_fused_gate` (safe gate, fused l2norm)
          -> gated RMSNorm -> o_proj.

opt       same projections/conv, but: sync-free state reset
          (precomputed int32 index list, like checkout HEAD), and the
          b10 CuTeDSLGen chunk-prefill champion
          (`kda_chunk_prefill`, kda/b10) with the safe-gate log-decay
          activation + q/k l2norm computed as separate (timed) torch
          ops. No host sync anywhere -> also CUDA-graph capturable.

Timing is EAGER wall-clock over --iters calls (events around the
loop): the baseline's host syncs are part of its real cost and forbid
graph capture. For reference the opt path is also timed as a captured
graph (column `opt_graph`).

  docker exec trt-dev bash -c "cd .../LLMDiveDeep && \
     CUDA_VISIBLE_DEVICES=4 python3 kimi_k3_layer/bench_kda_prefill_layer.py"
"""

from __future__ import annotations

import argparse
import statistics
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

_LLMDIR = Path(__file__).resolve().parents[1]


def bootstrap() -> None:
    import tensorrt_llm  # noqa: F401

    if "kimi_k3_layer" not in sys.modules:
        pkg = types.ModuleType("kimi_k3_layer")
        pkg.__path__ = [str(_LLMDIR / "kimi_k3_layer")]
        sys.modules["kimi_k3_layer"] = pkg
    for p in (str(_LLMDIR), str(_LLMDIR / "linear_attn")):
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=[4096, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    bootstrap()
    # the wheel's fla package predates the KDA chunk pipeline - graft
    # the checkout's modules (leaf-first; absolute imports resolve via
    # sys.modules) exactly like the decode bench grafts its kernels
    from kimi_k3_layer.kda_trtllm_kimi_k3 import _graft

    _graft("tensorrt_llm._torch.modules.stochastic_rounding",
           "_torch/modules/stochastic_rounding.py")
    for leaf in ("utils", "op", "index", "l2norm", "cumsum",
                 "solve_tril", "chunk_delta_h", "chunk_o",
                 "chunk_scaled_dot_kkt", "wy_fast", "chunk_kda"):
        mod = _graft(f"tensorrt_llm._torch.modules.fla.{leaf}",
                     f"_torch/modules/fla/{leaf}.py")
    chunk_kda_with_fused_gate = mod.chunk_kda_with_fused_gate
    l2norm_fwd = sys.modules[
        "tensorrt_llm._torch.modules.fla.l2norm"].l2norm_fwd
    from tensorrt_llm._torch.modules.mamba.causal_conv1d import (
        causal_conv1d_fn,
    )

    from kda.b10.b10_kda_chunk_prefill_cutedsl import kda_chunk_prefill
    from kimi_k3_layer.config import GATE_LOWER_BOUND, RMS_EPS, k3_shard
    from kimi_k3_layer.kda_trtllm_kimi_k3 import KimiK3KDA

    shard = k3_shard("tp8")
    H, D = shard.heads_local, shard.head_dim  # 12, 128
    proj = shard.proj_dim  # 1536

    # weights/modules from the existing decode layer (batch dim unused
    # by the pieces we call)
    torch.manual_seed(2026)
    layer = KimiK3KDA(shard, 1, "trt_fused").cuda().eval()
    w_in = layer.in_proj_qkvgfab.weight.data  # [6288, 7168]
    pad = layer.in_proj_padding
    split = layer.in_proj_split_sizes  # [4608, 1536, 128, 12]
    conv_w = layer.conv_weight.data  # [4608, 4]
    A_log, dt_bias = layer.A_log.data, layer.dt_bias.data
    f_b_w = layer.f_b_proj.weight.data  # [1536, 128]
    o_w = layer.o_proj.weight.data
    norm = layer.o_norm  # gated RMSNorm (sigmoid gate)

    # state pools (one slot), conv state pool
    state = torch.zeros(1, H, D, D, device="cuda", dtype=torch.float32)
    conv_states = torch.zeros(1, shard.qkv_dim, shard.conv_size - 1,
                              device="cuda", dtype=torch.bfloat16)
    state_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    has_init = torch.zeros(1, device="cuda", dtype=torch.bool)

    def act_safe_gate(raw_g: torch.Tensor) -> torch.Tensor:
        """TRT-LLM safe gate: lb * sigmoid(exp(A_log)*(g + bias))."""
        g = (raw_g.float() + dt_bias) * torch.exp(A_log.float())[
            None, :].repeat_interleave(D, 1)
        return GATE_LOWER_BOUND * torch.sigmoid(g)

    def common_front(h, S, cu):
        projected = h @ w_in.T
        if pad:
            projected = projected[:, :-pad]
        qkv, z, f_a, beta_l = torch.split(projected, split, dim=-1)
        conv_out = causal_conv1d_fn(
            qkv.transpose(0, 1).contiguous(),
            conv_w,
            activation="silu",
            conv_states=conv_states,
            has_initial_state=has_init,
            cache_indices=state_indices,
            query_start_loc=cu,
        ).transpose(0, 1)
        raw_gate = f_a @ f_b_w.T  # [S, 1536]
        return conv_out, z, raw_gate, beta_l

    def tail(core_out, z, S):
        normed = norm(core_out.view(1, S, H, D),
                      z.view(1, S, H, D))
        return normed.reshape(S, proj) @ o_w.T

    def run_baseline(h, S, cu):
        # (1) production state reset: boolean indexing -> nonzero() ->
        # blocking D2H of the match count, EVERY prefill batch
        idx = state_indices[~has_init]
        state[idx] = 0.0
        conv_states[idx] = 0.0
        # (2) the Triton chunk pipeline computes chunk indices with
        # .tolist() (blocking D2H); its @tensor_cache keys on tensor
        # IDENTITY, so a fresh cu_seqlens object per call - what every
        # serving step hands the kernel - pays the sync every time
        cu = cu.clone()
        conv_out, z, raw_gate, beta_l = common_front(h, S, cu)
        q, k, v = (t.view(1, S, H, D) for t in
                   conv_out.split(proj, dim=-1))
        o, _ = chunk_kda_with_fused_gate(
            q=q, k=k, v=v,
            raw_g=raw_gate.view(1, S, H, D),
            raw_beta=beta_l.view(1, S, H).float(),
            A_log=A_log, g_bias=dt_bias,
            scale=D ** -0.5,
            initial_state=state,
            initial_state_indices=state_indices,
            inplace_indexed_state_update=True,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
        )
        return tail(o.view(S, H, D), z, S)

    reset_idx = torch.zeros(1, device="cuda", dtype=torch.long)

    def run_opt(h, S, cu):
        # sync-free reset (checkout-HEAD form: precomputed index list)
        state.index_fill_(0, reset_idx, 0.0)
        conv_states.index_fill_(0, reset_idx, 0.0)
        conv_out, z, raw_gate, beta_l = common_front(h, S, cu)
        q, k, v = (t.view(1, S, H, D) for t in
                   conv_out.split(proj, dim=-1))
        # fused Triton l2norm (same kernel family the TRT pipeline
        # fuses); the D^-0.5 scale is linear in q, fold it here so no
        # output pass is needed
        q = l2norm_fwd(q.contiguous()).mul_(D ** -0.5)
        k = l2norm_fwd(k.contiguous())
        g = act_safe_gate(raw_gate).view(1, S, H, D)
        beta = torch.sigmoid(beta_l.float()).view(1, S, H).contiguous()
        o, _ = kda_chunk_prefill(q, k, v, g, beta,
                                 state.zero_())
        return tail(o.view(S, H, D), z, S)

    def time_eager(fn, h, S, cu):
        for _ in range(args.warmup):
            fn(h, S, cu)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.iters):
                fn(h, S, cu)
            e.record()
            e.synchronize()
            samples.append(s.elapsed_time(e) * 1000 / args.iters)
        return statistics.median(samples)

    def time_graph_opt(fn, h, S, cu):
        """The sync-free opt path is CUDA-graph capturable (the
        baseline is NOT - its D2H syncs forbid capture). Production
        engines replay per-shape graphs; this column is the opt's
        honest cost in that mode."""
        for _ in range(args.warmup):
            fn(h, S, cu)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(args.iters):
                fn(h, S, cu)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            e.synchronize()
            samples.append(s.elapsed_time(e) * 1000 / args.iters)
        return statistics.median(samples)

    print(f"{'S':>7} {'base_us':>10} {'opt_us':>10} {'gain':>7} "
          f"{'opt_graph':>10} {'gain_g':>7} "
          f"{'cosine':>8}  (K3 tp8 shard, B=1)")
    for S in args.seq_lens:
        gen = torch.Generator(device="cuda").manual_seed(S)
        h = (torch.randn(S, shard.hidden, generator=gen, device="cuda",
                         dtype=torch.float32) * 0.02).to(torch.bfloat16)
        cu = torch.tensor([0, S], device="cuda", dtype=torch.int32)

        out_b = run_baseline(h, S, cu)
        out_o = run_opt(h, S, cu)
        cos = F.cosine_similarity(
            out_b.float().flatten(), out_o.float().flatten(), dim=0
        ).item()

        t_b = time_eager(run_baseline, h, S, cu)
        t_o = time_eager(run_opt, h, S, cu)
        t_g = time_graph_opt(run_opt, h, S, cu)
        print(f"{S:>7} {t_b:>10.1f} {t_o:>10.1f} "
              f"{(t_b - t_o) / t_b * 100:>+6.1f}% "
              f"{t_g:>10.1f} {(t_b - t_g) / t_b * 100:>+6.1f}% "
              f"{cos:>8.5f}",
              flush=True)


if __name__ == "__main__":
    main()
