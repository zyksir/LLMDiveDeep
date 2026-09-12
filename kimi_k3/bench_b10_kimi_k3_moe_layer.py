#!/usr/bin/env python3
"""TP8 trace bench for the two self-contained Kimi-K3 MoE production blocks.

Profiles ``TrtKimiK3MoEBlock`` (TRT-LLM fork production wiring),
``SglKimiK3MoEBlock`` (sglang's K3 MoE on vendored kernels) and, with
``--emit-blocks ...,trt_rc25``, ``TrtRc25KimiK3MoEBlock`` (upstream
v1.3.0rc25 forward on the rc19 runtime) at the MTP1 decode shapes and
writes ONE gzipped chrome trace whose ``TrtKimiK3MoE[M=…]`` /
``SglKimiK3MoE[M=…]`` / ``TrtRc25KimiK3MoE[M=…]`` record_function spans wrap
CUDA-graph replays — the same capture regime as the serving baselines under
``traces/``. ``--trace-out none`` runs the identical flow without the
profiler (smoke / warm-cache mode).

Launch (bench convention):
    mpirun -n 8 --allow-run-as-root -x PYTHONPATH \
        python3 -u kimi_k3/bench_b10_kimi_k3_moe_layer.py \
        --sizes 1,2,4 --trace-out kimi_k3/traces/mini_kimi_k3_moe.trace.json.gz
PYTHONPATH must include the repo root: the vendored sglang P2P check
re-executes itself as a subprocess and needs ``kimi_k3`` importable.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import torch

# Anchors the emit-trace wall-time report: everything between process
# start and _emit_trace (torch/tensorrt_llm imports, MPI/dist init) is
# reported as "pre" so slow runs can be attributed.
_PROCESS_T0 = time.perf_counter()

_ROOT = Path(__file__).resolve().parents[1]
# The repo root must PRECEDE any PYTHONPATH entry: containers that put e.g.
# /workspace/trtllm (which has its own top-level ``common``) on PYTHONPATH
# would otherwise shadow this repo's ``common`` package.
if str(_ROOT) in sys.path:
    sys.path.remove(str(_ROOT))
sys.path.insert(0, str(_ROOT))

# Pin every JIT/compile cache (applies on import) BEFORE the heavy imports
# read their env: flashinfer resolves FLASHINFER_WORKSPACE_BASE at import
# time (tensorrt_llm can pull it in), and the CuTe DSL folds its env knobs
# into the compile-cache hash.
import kimi_k3.kernels.jit_cache_env as jit_cache_env  # noqa: E402

DECODE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PREFILL_SIZES = (512, 1024, 2048, 4096, 8192, 16384)

# Trace-alignment span labels for the self-contained production blocks.
# Every forward invocation of these classes — and only these — is wrapped
# in torch.profiler.record_function with this label so mini traces can be
# diffed against the serving baselines by span name.
_BLOCK_SPANS = {
    "TrtKimiK3MoEBlock": "TrtKimiK3MoE",
    "TrtRc25KimiK3MoEBlock": "TrtRc25KimiK3MoE",
    "SglKimiK3MoEBlock": "SglKimiK3MoE",
}
# MTP1 speculative decode verifies 2 tokens per request; M=2 is the
# primary alignment shape, 1/4 are the nearby decode sizes.
EMIT_TRACE_SIZES = (1, 2, 4)


def _forward_span(layer, tokens: int | None = None):
    """record_function span for the production blocks; ``tokens`` appends
    an ``[M=...]`` suffix so multi-size traces stay distinguishable while
    prefix-matching on the base name keeps working."""
    label = _BLOCK_SPANS.get(type(layer).__name__)
    if label is None:
        return contextlib.nullcontext()
    if tokens is not None:
        label = f"{label}[M={tokens}]"
    return torch.profiler.record_function(label)


def _phase(rank: int, name: str) -> None:
    """Rank-0 one-liner per pipeline phase so a wedged run is localizable
    from the log alone."""
    if rank == 0:
        print(f"[emit-trace][phase] {name}", flush=True)


def _configure_nccl_graph_policy() -> None:
    """Use the serialized NCCL CUDA-graph policy for these benchmarks."""
    # NCCL 2.30+ supports stream ordering off only when graph mixing is off.
    # These benchmarks serialize graph replays and do not need mixed launches.
    os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
    os.environ["NCCL_GRAPH_STREAM_ORDERING"] = "0"


def _load_cached_expert_weights(
    name: str, parameter: torch.Tensor, rank: int, world: int,
) -> torch.Tensor | None:
    """Optional exact-bytes override for one packed expert parameter.

    When ``out/moe_weight_cache`` (or ``K3_MOE_WEIGHT_CACHE_DIR``) holds
    ``k3_<param name>_r<rank>of<world>.pt`` — e.g. real quantized shards
    exported for numerics work — it wins over the random GPU synthesis.
    Absent files are not an error: the bench's default weights are random.
    """
    cache_dir = Path(os.environ.get(
        "K3_MOE_WEIGHT_CACHE_DIR", str(_ROOT / "out" / "moe_weight_cache")))
    path = cache_dir / f"k3_{name}_r{rank}of{world}.pt"
    if not path.exists():
        return None
    tensor = torch.load(
        path, map_location=parameter.device, weights_only=True)
    if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
        raise ValueError(
            f"weight cache {path} holds {tuple(tensor.shape)}/{tensor.dtype}"
            f", parameter {name} needs {tuple(parameter.shape)}/"
            f"{parameter.dtype}")
    return tensor


def init_weights(moe, world: int, rank: int, seed: int = 0) -> None:
    """Initialize deterministic global weights, then select this rank's shard."""
    from kimi_k3.config import (
        HIDDEN,
        MOE_INTER,
        MOE_LATENT,
        NUM_EXPERTS,
        SHARED_INTER,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, std=0.02):
        return (
            torch.randn(
                *shape,
                generator=gen,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * std
        )

    with torch.no_grad():
        moe.gate.weight.copy_(randn(NUM_EXPERTS, HIDDEN))
        moe.gate.e_score_correction_bias.copy_(
            randn(NUM_EXPERTS, std=0.02).to(
                moe.gate.e_score_correction_bias.dtype
            )
        )
        moe.fc1_latent_proj.weight.copy_(randn(MOE_LATENT, HIDDEN))
        moe.fc2_latent_proj.weight.copy_(randn(HIDDEN, MOE_LATENT))
        moe.latent_norm.weight.copy_(
            1 + randn(MOE_LATENT).to(moe.latent_norm.weight.dtype)
        )

        shared_inter_local = SHARED_INTER // world
        gate = randn(SHARED_INTER, HIDDEN)
        up = randn(SHARED_INTER, HIDDEN)
        down = randn(HIDDEN, SHARED_INTER)
        rows = slice(
            rank * shared_inter_local,
            (rank + 1) * shared_inter_local,
        )
        moe.shared_experts.gate_up_proj.weight.copy_(
            torch.cat([gate[rows], up[rows]])
        )
        moe.shared_experts.down_proj.weight.copy_(down[:, rows])

        backend = getattr(moe.experts, "backend", moe.experts)
        if type(backend).__name__ == "TRTLLMGenFusedMoE":
            for name, parameter in backend.named_parameters():
                experts_local = parameter.shape[0]
                # EP: dim0 is this rank's expert shard. TP-expert
                # (moe_tp_size=world): ALL experts are local and the
                # trailing dims are already inter-sliced by the backend —
                # take the whole table (rank-identical values; fine for
                # timing, outputs stay finite through the AR).
                experts = (slice(rank * experts_local,
                                 (rank + 1) * experts_local)
                           if experts_local < NUM_EXPERTS
                           else slice(0, NUM_EXPERTS))
                if parameter.dtype == torch.uint8 and "scale" in name:
                    # Benign constant block scales (2^(124-127) = 1/8):
                    # finite outputs at any packed-code draw.
                    parameter.fill_(124)
                elif parameter.dtype == torch.uint8:
                    # Packed MXFP4 expert weights: every 4-bit code is a
                    # legal fp4 value and the scales above are benign, so
                    # the rank's shard is synthesized DIRECTLY on GPU in
                    # the final packed layout — no CPU quantize/pack
                    # pipeline. (The former host-side FULL-table randint
                    # staging dominated bench wall time: ~95 s of the
                    # ~157 s session.) Seed folds in the parameter name
                    # and the expert-slice start, so EP ranks draw their
                    # own experts' bytes while TP-expert mode (slice
                    # start 0 everywhere) stays rank-identical, matching
                    # the old full-table-then-slice semantics.
                    cached = _load_cached_expert_weights(
                        name, parameter, rank, world)
                    if cached is not None:
                        parameter.copy_(cached)
                    else:
                        import zlib

                        shard_gen = torch.Generator(
                            device=parameter.device).manual_seed(
                            int(gen.initial_seed())
                            ^ zlib.crc32(name.encode())
                            ^ experts.start)
                        parameter.copy_(torch.randint(
                            0, 256, parameter.shape,
                            generator=shard_gen,
                            device=parameter.device,
                            dtype=torch.uint8,
                        ))
                else:
                    parameter.zero_()
            if hasattr(backend, "post_load_weights"):
                backend.post_load_weights()
        else:
            w31_full = randn(NUM_EXPERTS, 2 * MOE_INTER, MOE_LATENT)
            w2_full = randn(NUM_EXPERTS, MOE_LATENT, MOE_INTER)
            w31_local = backend.w3_w1_weight.data
            w2_local = backend.w2_weight.data
            experts_local = w31_local.shape[0]
            experts = (slice(rank * experts_local,
                             (rank + 1) * experts_local)
                       if experts_local < NUM_EXPERTS
                       else slice(0, NUM_EXPERTS))
            w31_local.copy_(w31_full[experts])
            w2_local.copy_(w2_full[experts])
    torch.cuda.synchronize()


def _sizes(spec: str) -> tuple[int, ...]:
    if spec == "decode":
        return DECODE_SIZES
    if spec == "prefill":
        return PREFILL_SIZES
    if spec == "all":
        return DECODE_SIZES + PREFILL_SIZES
    return tuple(int(item) for item in spec.split(","))


def _emit_trace(args, config, aux, world: int, rank: int) -> None:
    """One torch-profiler session over the self-contained production
    blocks at the MTP1 decode shapes, exported as a gzipped chrome trace.

    Methodology (matches the serving baselines, which are CUDA-graph
    decode captures): eager autotune + warmup, ONE forward per
    (block, size) captured into its own CUDA graph — collectives inside
    the graph — then replay warmups, and only then the profiler window
    over graph REPLAYS, each wrapped in its record_function span. Eager
    capture is wrong here: it inflates host cost and exposes lamport-AR
    spin waits that tight replays do not pay. ``--trace-out none`` runs
    the identical flow without the profiler.
    """
    import gzip
    import tempfile

    import torch.distributed as dist
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.modules.multi_stream_utils import (
        with_multi_stream,
    )

    import kimi_k3.b10_kimi_k3_moe_layer as _layer_mod
    from kimi_k3.b10_kimi_k3_moe_layer import TrtKimiK3MoEBlock
    from kimi_k3.config import HIDDEN

    setup_start = time.perf_counter()
    pre_s = setup_start - _PROCESS_T0
    _phase(rank, "import done")
    profiling = args.trace_out != "none"
    sizes = (EMIT_TRACE_SIZES if args.sizes == "all"
             else tuple(sorted(set(_sizes(args.sizes)))))
    requested = tuple(
        item.strip() for item in args.emit_blocks.split(",") if item.strip())
    unknown = set(requested) - {"trt", "sgl", "trt_rc25"}
    if unknown:
        raise SystemExit(
            f"--emit-blocks accepts trt,sgl,trt_rc25; got {unknown}")

    blocks = []
    collectives = None
    if "trt" in requested:
        # Production wiring: reduces through TRT-LLM's own
        # AllReduce/MoEAllReduce ops, no bench Collectives needed.
        blocks.append(TrtKimiK3MoEBlock(
            config, layer_idx=0, aux_stream_dict=aux).cuda())
    if "trt_rc25" in requested:
        # Upstream v1.3.0rc25 forward replicated on the rc19 runtime;
        # reduces through the experts' own MoE.all_reduce, no bench
        # Collectives needed (see the class docstring for provenance
        # and named deviations).
        from kimi_k3.b10_kimi_k3_moe_layer import TrtRc25KimiK3MoEBlock
        blocks.append(TrtRc25KimiK3MoEBlock(
            config, layer_idx=0, aux_stream_dict=aux).cuda())
    if "sgl" in requested:
        sgl_cls = getattr(_layer_mod, "SglKimiK3MoEBlock", None)
        if sgl_cls is None:
            if rank == 0:
                print("[emit-trace] SglKimiK3MoEBlock not in "
                      "b10_kimi_k3_moe_layer yet — skipping the sgl block",
                      flush=True)
        else:
            # The fallback (unfused) tail reduces through the bench
            # Collectives instance.
            collectives = _build_collectives(world, sizes, max(sizes))
            sgl = sgl_cls(
                config,
                layer_idx=0,
                aux_stream_dict=aux,
                reduce_output=world > 1,
                collectives=collectives,
            ).cuda()
            # The block stands up sglang's CustomAllReduceV2 workspaces
            # itself on first forward (kernels/sgl_adapters/comm.py,
            # get_sgl_ar_state — sglang's k3_ar_fusion._get_state pattern):
            # fused finalize+push-AR+RMSNorm tail when available, the
            # documented plain-TP fallback tail otherwise. The first
            # forward below runs eagerly on every rank in lockstep, before
            # any CUDA graph capture, so the collective build is safe.
            blocks.append(sgl)
    if not blocks:
        raise SystemExit("--emit-blocks selected no available block")
    construct_s = time.perf_counter() - setup_start
    _phase(rank, "blocks constructed")

    weights_start = time.perf_counter()
    for block in blocks:
        init_weights(block, world, rank)
    weights_s = time.perf_counter() - weights_start
    _phase(rank, "weights built")
    # One STATIC buffer per size is what the graphs capture; each profiled
    # replay first copies a distinct batch from the pool into it (outside
    # the record_function span), so every step sees fresh hidden states —
    # and with them a fresh expert-routing draw — instead of N replays of
    # one frozen batch. Rank-identical (seeded device generator): the
    # activations feed TP collectives.
    generator = torch.Generator(device="cuda").manual_seed(2)

    def _draw(tokens: int) -> torch.Tensor:
        return torch.randn(
            tokens, HIDDEN, generator=generator, device="cuda",
            dtype=torch.float32,
        ).to(torch.bfloat16)

    statics = {tokens: _draw(tokens) for tokens in sizes}
    pool = {
        tokens: [_draw(tokens) for _ in range(args.emit_trace_iters)]
        for tokens in sizes
    }

    def _feed(tokens: int, index: int) -> None:
        statics[tokens].copy_(pool[tokens][index % args.emit_trace_iters])

    def run_all_eager(index: int) -> None:
        for block in blocks:
            for tokens in sizes:
                _feed(tokens, index)
                with torch.no_grad():
                    with _forward_span(block, tokens):
                        block(statics[tokens])

    def _sync_barrier():
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()

    graphs = []  # (block, tokens, graph) in a rank-identical order
    with with_multi_stream(True):
        # Eager warmup: autotuner pass first, then plain iterations for
        # JIT/cublas-heuristic/workspace warm state before any capture.
        # cache_path persists the tuned tactic map under out/jit_cache
        # (per-rank suffix appended by the autotuner), so warm sessions
        # skip the re-profiling sweep.
        warmup_start = time.perf_counter()
        with autotune(cache_path=str(jit_cache_env.autotuner_cache_path(
                "bench_b10_kimi_k3_moe_layer"))):
            run_all_eager(0)
        _sync_barrier()
        for index in range(5):
            run_all_eager(index)
        _sync_barrier()
        warmup_s = time.perf_counter() - warmup_start
        _phase(rank, "warmup done")
        # One graph per (block, size), capturing the STATIC buffer; the
        # loop order is identical on every rank, so in-graph collectives
        # capture in lockstep.
        capture_start = time.perf_counter()
        for block in blocks:
            for tokens in sizes:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    with torch.no_grad():
                        block(statics[tokens])
                graphs.append((block, tokens, graph))
                _sync_barrier()
        for index in range(3):  # replay warmup, outside the profiler window
            for _, tokens, graph in graphs:
                _feed(tokens, index)
                graph.replay()
        _sync_barrier()
        capture_s = time.perf_counter() - capture_start
        _phase(rank, "graphs captured")

        _phase(rank, "profiling" if profiling else "replaying (no profiler)")
        profiled_start = time.perf_counter()
        profiler_cm = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            if profiling else contextlib.nullcontext()
        )
        with profiler_cm as profiler:
            for index in range(args.emit_trace_iters):
                for block, tokens, graph in graphs:
                    # distinct batch per step, copied outside the span
                    # (D2D, [M, 7168] bf16 — negligible)
                    _feed(tokens, index)
                    with _forward_span(block, tokens):
                        graph.replay()
            _sync_barrier()
        profiled_s = time.perf_counter() - profiled_start

    export_start = time.perf_counter()
    if profiling:
        _phase(rank, "export")
        if rank == 0:
            out = Path(args.trace_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
                profiler.export_chrome_trace(tmp.name)
                with open(tmp.name, "rb") as src, \
                        gzip.open(out, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
        if world > 1:
            dist.barrier()
    export_s = time.perf_counter() - export_start
    if rank == 0:
        if profiling:
            print(f"[emit-trace] wrote {Path(args.trace_out)} "
                  f"(blocks={[type(b).__name__ for b in blocks]}, "
                  f"sizes={sizes}, replays={args.emit_trace_iters})",
                  flush=True)
        print(f"[emit-trace] wall: pre(imports+dist)={pre_s:.1f}s "
              f"construct={construct_s:.1f}s weights={weights_s:.1f}s "
              f"warmup+autotune={warmup_s:.1f}s capture={capture_s:.1f}s "
              f"profiled={profiled_s:.1f}s export+gzip={export_s:.1f}s",
              flush=True)
    if collectives is not None:
        collectives.destroy()


def _build_collectives(world: int, sizes, max_decode: int):
    import torch.distributed as dist
    from communication.collective import Collectives
    from kimi_k3.config import HIDDEN, MOE_LATENT

    if world == 1:
        return Collectives(None, 0)
    max_tokens = max(sizes)
    packed = HIDDEN + MOE_LATENT
    collectives = Collectives(
        dist.group.WORLD,
        max_numel=max_tokens * packed,
        max_hidden=packed,
        col_ag_max_rows=max_decode,
        col_ag_max_output_columns=MOE_LATENT,
        fused_ar_max_rows=max_decode,
    )
    # Persist the tuned (op, dim, bucket) -> impl map across processes:
    # with K3_AUTO_MAP_JSON set, the first process on this node profiles
    # and exports; later ones import and skip straight to measuring.
    # Keyed by world size — a TP4 map must never serve a TP8 run. Any
    # cell the imported map misses is still lazily tuned (ensure_tuned).
    map_base = os.environ.get("K3_AUTO_MAP_JSON", "")
    map_path = f"{map_base}.w{world}.json" if map_base else ""
    if map_path and os.path.exists(map_path):
        import json
        with open(map_path) as f:
            collectives.import_auto_map(json.load(f))
        if dist.get_rank() == 0:
            print(f"[collectives.autotune] imported map from {map_path}",
                  flush=True)
        return collectives
    collectives.autotune(
        dims=(HIDDEN, MOE_LATENT, packed),
        max_tokens=min(max_tokens, 256),
        ops=("all_reduce", "all_gather", "allreduce_norm"),
        warmup=2,
        iters=10,
        log=(dist.get_rank() == 0),
    )
    if max_tokens > 256:
        collectives.autotune(
            dims=(HIDDEN, MOE_LATENT),
            max_tokens=max_tokens,
            ops=("all_reduce",),
            warmup=1,
            iters=5,
            skip=("flashinfer", "torch_symm:1shot"),
            clear=False,
            log=(dist.get_rank() == 0),
        )
    if map_path and dist.get_rank() == 0:
        import json
        with open(map_path, "w") as f:
            json.dump(collectives.export_auto_map(), f, indent=1)
        print(f"[collectives.autotune] exported map to {map_path}", flush=True)
    return collectives


def main() -> None:
    _configure_nccl_graph_policy()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", default="all",
        help="comma-separated token counts; 'all' means the MTP1 decode "
             f"trace shapes {EMIT_TRACE_SIZES}")
    parser.add_argument(
        "--trace-out",
        default=str(Path(__file__).parent / "traces"
                    / "mini_kimi_k3_moe.trace.json.gz"),
        help="gzipped chrome trace output path, or 'none' to run the "
             "identical flow without the profiler")
    parser.add_argument("--emit-trace-iters", type=int, default=24,
                        help="profiled graph replays per block/size")
    parser.add_argument(
        "--emit-blocks", default="trt,sgl",
        help="which production blocks to profile (trt,sgl,trt_rc25)")
    args = parser.parse_args()

    import torch.distributed as dist
    from tensorrt_llm._torch.utils import AuxStreamType

    from kimi_k3.b10_kimi_k3_moe_layer import k3_model_config

    rank = int(os.environ.get(
        "RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world = int(os.environ.get(
        "WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    # multi-node: the device index is the LOCAL rank
    torch.cuda.set_device(int(os.environ.get(
        "OMPI_COMM_WORLD_LOCAL_RANK",
        os.environ.get("LOCAL_RANK", rank))) % torch.cuda.device_count())
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        # Prefer a fresh port per launch (pass MASTER_PORT via mpirun -x)
        # so stale TCPStore leftovers from a killed run cannot wedge the
        # rendezvous.
        os.environ.setdefault("MASTER_PORT", "29521")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world,
            device_id=torch.device("cuda", torch.cuda.current_device()),
        )

    config = k3_model_config(rank, world)
    # ONE shared stream set for the whole session: every block constructed
    # by _emit_trace receives this same aux_stream_dict, so side-stream
    # work (shared experts, routing) of all blocks runs on the same
    # physical streams — no per-block stream creation.
    aux = {kind: torch.cuda.Stream() for kind in AuxStreamType}
    _emit_trace(args, config, aux, world, rank)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
