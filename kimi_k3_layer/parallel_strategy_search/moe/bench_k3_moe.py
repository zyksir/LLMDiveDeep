#!/usr/bin/env python3
"""PRODUCTION K3 MoE layer bench: TP vs CP via the real KimiK3MoE module.

Uses the unchanged production module (modeling_kimi_k3.KimiK3MoE, a
NemotronHMOE with SiTU experts, the 3584 latent expert space, and the fused
moe_finalize_allreduce_rmsnorm_concat epilogue). The two modes differ ONLY in
the mapping flag, exactly as production:

- ``tp``  Mapping(enable_attention_dp=False): tokens replicated, comm=None,
          each rank runs its 112 local experts on all tokens, the fused
          finalize+allreduce(+latent rmsnorm+shared concat) completes the sum
          (decode T<=128 -> fused kernel; larger -> cat+AR fallback, as prod).
- ``cp``  Mapping(enable_attention_dp=True): tokens split per rank,
          NVLinkOneSided A2A dispatch/combine, no allreduce.

Routing: ENABLE_PERFECT_ROUTER=1 (balanced; keeps the real dsv3 gate GEMM in
the trace). All experts carry IDENTICAL deterministic weights, which makes
outputs routing-invariant so the single-GPU reference check stays valid in
both modes. Timing/tracing run under with_multi_stream(True) around CUDA
graph capture/replay so the production decode overlap (shared expert on the
MoeShared stream, fc1-shard path) is active.

Launch (inside trt-k3-bench):
  mpirun --allow-run-as-root -np 8 python3 \
      kimi_k3_layer/parallel_strategy_search/moe/bench_k3_moe.py \
      --modes tp cp --decode-batch-sizes 8 32 128 --check --graph-timing
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

_PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PACKAGE_DIR))

os.environ.setdefault("TLLM_K3_FUSED_MOE_FINALIZE_ALLREDUCE", "1")
os.environ.setdefault("TLLM_K3_SPECIALIZED_MOE_FINALIZE_ALLREDUCE", "1")
os.environ["ENABLE_PERFECT_ROUTER"] = "1"

from harness import REPO_ROOT, setup_sys_path, verify_provenance  # noqa: E402

setup_sys_path()
sys.path.insert(0, str(REPO_ROOT / "kimi_k3_layer"))

import torch  # noqa: E402

from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402
from tensorrt_llm._utils import mpi_allgather, mpi_barrier, mpi_rank, mpi_world_size  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402

from bench_a2a_megamoe_pipeline import _correctness  # noqa: E402
from layer import make_shared_expert_checkpoint_weights  # noqa: E402

LOCAL_RESULTS = _PACKAGE_DIR / "local_results"
HIDDEN = 7168
LATENT = 3584
NUM_EXPERTS = 896
TOP_K = 16
MOE_INTERMEDIATE = 3072
SHARED_INTERMEDIATE = 6144
WEIGHT_SEED = 20260830


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.pstdev(values),
    }


def _named_randn(name: str, shape, device, std=0.02) -> torch.Tensor:
    import hashlib

    seed = WEIGHT_SEED ^ int.from_bytes(hashlib.sha256(name.encode()).digest()[:6], "little")
    generator = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * std).to(
        torch.bfloat16
    )


def _multi_stream():
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    return with_multi_stream(True)


def build_k3_moe(*, mode: str, world: int, rank: int, device: torch.device,
                 max_num_tokens: int) -> torch.nn.Module:
    from tensorrt_llm._torch.configs import KimiK3Config
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_kimi_k3 import KimiK3MoE
    from tensorrt_llm._torch.utils import AuxStreamType
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    from _torch.modules.moe.quantize_utils import MXFP4MXFP8QuantizeUtil

    tp_mode = mode in ("tp", "tp_te")
    if world == 1:
        mapping = Mapping(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1,
                          enable_attention_dp=False)
    elif mode == "tp_te":
        # TileRT-style TP-in-expert: all experts on every rank, intermediate
        # sharded 1/world; no dispatch/combine; reduce via the fused epilogue.
        mapping = Mapping(
            world_size=world, rank=rank, tp_size=world,
            moe_tp_size=world, moe_ep_size=1,
            enable_attention_dp=False,
        )
    else:
        mapping = Mapping(
            world_size=world, rank=rank, tp_size=world,
            moe_tp_size=1, moe_ep_size=world,
            enable_attention_dp=not tp_mode,
        )
    cfg = KimiK3Config(text_config=dict(
        architectures=["KimiK3ForCausalLM"],
        hidden_size=HIDDEN,
        moe_intermediate_size=MOE_INTERMEDIATE,
        routed_expert_hidden_size=LATENT,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        num_shared_experts=2,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=2.827,
        latent_moe_use_norm=True,
        hidden_act="situ",
        activation_situ_beta=1.0,
        activation_situ_linear_beta=1.0,
        rms_norm_eps=1e-5,
        num_hidden_layers=1,
        vocab_size=163840,
        num_attention_heads=96,
        num_key_value_heads=96,
    ), torch_dtype=torch.bfloat16)
    model_config = ModelConfig(
        pretrained_config=cfg,
        mapping=mapping,
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
        moe_backend="TRTLLM",
        max_num_tokens=max(max_num_tokens, 1),
        skip_create_weights_in_init=True,
        allreduce_strategy="AUTO",
    )
    streams = [torch.cuda.Stream() for _ in range(6)]
    aux_stream_dict = {
        AuxStreamType.Attention: streams[0],
        AuxStreamType.MoeShared: streams[0],
        AuxStreamType.MoeChunkingOverlap: streams[1],
        AuxStreamType.MoeBalancer: streams[2],
        AuxStreamType.MoeOutputMemset: streams[3],
        AuxStreamType.AttentionOutputGate: streams[4],
        AuxStreamType.MoeLatentFc1: streams[5],
    }
    moe = KimiK3MoE(
        model_config,
        layer_idx=0,
        aux_stream_dict=aux_stream_dict,
        reduce_output=(tp_mode and world > 1),
        defer_shared_output_add=False,
    )
    # Production excludes the latent projections from quantization via a
    # model-level regex; replicate by hand before create_weights.
    # Gate: real DeepseekV3Gate (2026-08-30: the historical
    # CUBLAS_STATUS_NOT_SUPPORTED no longer reproduces — verified across
    # eager/autotune/graph phases in debug/dsv3_gate_inmodule.py — so the
    # stub is gone and the gate GEMM is back in timing). Deterministic
    # weights loaded below with the rest.
    moe.fc1_latent_proj.quant_config = QuantConfig()
    moe.fc2_latent_proj.quant_config = QuantConfig()
    for _, module in moe.named_modules():
        if callable(getattr(module, "create_weights", None)):
            module.create_weights()
    moe.cuda(device)

    # ---- deterministic weights ----
    with torch.no_grad():
        moe.gate.weight.copy_(
            _named_randn("k3_gate", (NUM_EXPERTS, HIDDEN), device))
        bias = getattr(moe.gate, "e_score_correction_bias", None)
        if bias is not None:
            bias.zero_()
        moe.fc1_latent_proj.load_weights([{
            "weight": _named_randn("k3_fc1_latent", (LATENT, HIDDEN), device)
        }])
        moe.fc2_latent_proj.load_weights([{
            "weight": _named_randn("k3_fc2_latent", (HIDDEN, LATENT), device)
        }])
        moe.latent_norm.weight.fill_(1.0)

        shared_ckpt = make_shared_expert_checkpoint_weights(HIDDEN, SHARED_INTERMEDIATE, device)
        moe.shared_experts.gate_up_proj.load_weights([shared_ckpt["w1"], shared_ckpt["w3"]])
        moe.shared_experts.down_proj.load_weights([shared_ckpt["w2"]])

        util = MXFP4MXFP8QuantizeUtil(
            num_experts=1,
            dtype=torch.bfloat16,
            intermediate_size=MOE_INTERMEDIATE,
            hidden_size=LATENT,
            quant_config=QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
        )
        one = util.create_weights(weight_alignment=128, input_hidden_alignment=512)
        experts = moe.experts
        local_ids = list(getattr(experts, "initial_local_expert_ids",
                                 range(NUM_EXPERTS // mapping.moe_ep_size)))
        flat = {}
        for eid in local_ids:
            for proj in ("w1", "w2", "w3"):
                flat[f"{eid}.{proj}.weight"] = one[f"0.{proj}.weight"]
                flat[f"{eid}.{proj}.weight_scale"] = one[f"0.{proj}.weight_scale"]
        experts.load_weights(weights=[flat])
        post = getattr(experts, "post_load_weights", None) or getattr(
            getattr(experts, "backend", experts), "post_load_weights", None
        )
        if post is not None:
            post()
    moe.cuda(device)
    return moe


def _autotune_once(run: Callable[[], Any]) -> None:
    import tempfile

    tuner = AutoTuner.get()
    saved = (tuner.warmup, tuner.repeat, tuner.stream_delay_micro_secs)
    tuner.warmup, tuner.repeat, tuner.stream_delay_micro_secs = 0, 1, 10
    try:
        with torch.inference_mode(), autotune(
            cache_path=os.path.join(tempfile.gettempdir(), "k3_moe_prod_autotuner.json")
        ):
            run()
        torch.cuda.synchronize()
    finally:
        tuner.warmup, tuner.repeat, tuner.stream_delay_micro_secs = saved


def _benchmark_graph(run, warmup, iters):
    with torch.inference_mode(), _multi_stream():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        mpi_barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        mpi_barrier()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
    return _stats([starts[i].elapsed_time(ends[i]) for i in range(iters)])


def _benchmark_eager(run, warmup, iters):
    with torch.inference_mode():
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        mpi_barrier()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            run()
            ends[i].record()
        torch.cuda.synchronize()
    return _stats([starts[i].elapsed_time(ends[i]) for i in range(iters)])


def _capture_graph_trace(run, out_path: Path) -> str:
    with torch.inference_mode(), _multi_stream():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        mpi_barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        mpi_barrier()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out_path))
    return str(out_path)


def _hidden(tokens: int, seed: int, device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device=device,
                        generator=generator) * 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=("tp", "tp_te", "cp"), default=["tp", "cp"])
    parser.add_argument("--decode-batch-sizes", type=int, nargs="*",
                        default=[8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--prefill-tokens", type=int, nargs="*", default=[])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check", action="store_true",
                        help="compare vs a single-GPU KimiK3MoE reference (rank0)")
    parser.add_argument("--graph-timing", action="store_true", default=True)
    parser.add_argument("--eager", dest="graph_timing", action="store_false")
    parser.add_argument("--graph-trace", action="store_true")
    parser.add_argument("--tag", default="k3_moe_prod")
    args = parser.parse_args()

    rank, world = mpi_rank(), mpi_world_size()
    local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.device(device).__enter__()  # ambient ctx (MoE builder requirement)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "30301")

    grid = [("decode", b) for b in args.decode_batch_sizes] + [
        ("prefill", t) for t in args.prefill_tokens
    ]
    max_tokens = max(size for _, size in grid)

    mapping_probe = Mapping(world_size=world, rank=rank, tp_size=world,
                            moe_tp_size=1, moe_ep_size=world)
    AutoTuner.get().setup_distributed_state(mapping_probe)

    ref = None
    if args.check and rank == 0:
        ref = build_k3_moe(mode="tp", world=1, rank=0, device=device,
                           max_num_tokens=min(max_tokens, 4096))

    rows, checks = [], {}
    md = SimpleNamespace(all_rank_num_tokens=None)
    for mode in args.modes:
        moe = build_k3_moe(mode=mode, world=world, rank=rank, device=device,
                           max_num_tokens=max_tokens)
        for phase, size in grid:
            if mode in ("tp", "tp_te"):
                x = _hidden(size, args.seed + size, device)  # replicated
                run = lambda x=x, moe=moe: moe(x, md)  # noqa: E731
            else:
                assert size % world == 0, "global tokens must divide world in cp"
                tl = size // world
                gen_seed = args.seed + size
                x_full = _hidden(size, gen_seed, device)
                x = x_full[rank * tl:(rank + 1) * tl].contiguous()
                md_cp = SimpleNamespace(all_rank_num_tokens=[tl] * world)
                run = lambda x=x, moe=moe, md_cp=md_cp: moe(  # noqa: E731
                    x, md_cp, all_rank_num_tokens=md_cp.all_rank_num_tokens)
            _autotune_once(run)
            if args.check:
                with torch.inference_mode():
                    out = run().clone()
                torch.cuda.synchronize()
                verdict = None
                if rank == 0:
                    with torch.inference_mode():
                        want_full = ref(_hidden(size, args.seed + size, device), md).clone()
                    torch.cuda.synchronize()
                    want = want_full if mode in ("tp", "tp_te") else want_full[: size // world]
                    verdict = _correctness(out, want, atol=0.1, rtol=0.05,
                                           min_close_fraction=0.85)
                gathered = mpi_allgather(verdict)
                checks[f"{phase}_{size}_{mode}"] = gathered[0]
            timing = (_benchmark_graph if args.graph_timing else _benchmark_eager)(
                run, args.warmup, args.iters)
            trace_path = None
            if args.graph_trace:
                trace_path = _capture_graph_trace(
                    run,
                    LOCAL_RESULTS / "traces"
                    / f"{args.tag}_{phase}_{mode}_bs{size}_w{world}_graph_rank{rank}.json",
                )
            gathered_t = mpi_allgather(timing)
            row = {
                "phase": phase, "size": size, "mode": mode,
                "score_median_ms": max(t["median_ms"] for t in gathered_t),
                "per_rank": {f"r{i}": t for i, t in enumerate(gathered_t)},
                "traces": mpi_allgather(trace_path) if args.graph_trace else None,
            }
            rows.append(row)
            mpi_barrier()
            if rank == 0:
                print(f"[k3moe] {phase} size={size} {mode}: "
                      f"{row['score_median_ms']:.3f} ms median", flush=True)
        del moe
        torch.cuda.empty_cache()
        mpi_barrier()

    receipt = {
        "kind": "k3_moe_production",
        "module": "modeling_kimi_k3.KimiK3MoE (unchanged; SiTU experts, latent 3584, "
                  "fused moe_finalize_allreduce_rmsnorm_concat epilogue)",
        "labels": {
            "tp_te": "TileRT-style TP-in-expert: moe_tp=world (all 896 experts/rank, "
                     "intermediate /8), no A2A, fused finalize+AR",
            "tp": "enable_attention_dp=False: replicated tokens, comm=None, fused finalize+AR "
                  "(decode<=128 tokens fused kernel; larger/prefill = cat+AR fallback, as prod)",
            "cp": "enable_attention_dp=True: token split, NVLinkOneSided A2A, no allreduce",
        },
        "routing": "ENABLE_PERFECT_ROUTER=1; identical expert weights (routing-invariant outputs)",
        "timing": {"warmup": args.warmup, "iters": args.iters,
                   "launch": "cuda_graph+multi_stream" if args.graph_timing else "eager"},
        "env": {k: os.environ.get(k) for k in (
            "TLLM_K3_FUSED_MOE_FINALIZE_ALLREDUCE",
            "TLLM_K3_SPECIALIZED_MOE_FINALIZE_ALLREDUCE",
            "ENABLE_B10_SHARD_FC1", "ENABLE_B10_COLLECTIVES")},
        "rows": rows,
        "checks": checks,
        "provenance": verify_provenance(),
    }
    if rank == 0:
        LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
        out = LOCAL_RESULTS / f"{args.tag}_w{world}.json"
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[k3moe] receipt={out}", flush=True)


if __name__ == "__main__":
    main()
