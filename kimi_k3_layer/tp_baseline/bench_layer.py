#!/usr/bin/env python3
"""Kimi-K3 MoE layer baseline: TP reference vs CP emulation (mpirun entry).

Subcommands
-----------
correctness  Reduced-expert-count run (default 64 experts) whose layer output
             is checked against a sequential single-GPU reference of the same
             math: the unchanged ``MXFP4MXFP8RefGatedMLPFusedMoE`` (per-expert
             quantized GatedMLP loop, from TRT-LLM's unittest fixtures) for
             the routed path, plus a TP1 full-weight ``GatedMLP`` with the
             identical checkpoint weights for the shared path. Same
             close-fraction/cosine metrics as bench_a2a_megamoe_pipeline.py.
measure      Full 896-expert layer (cached deterministic weights, same cache
             as the MegaMoE receipts) timed over a prefill/decode grid in both
             modes with per-phase CUDA-event breakdown.

Launch (inside trt-k3-prod, GPUs picked idle, serialized):
  docker exec trt-k3-prod bash -c 'cd /node-storage/LLMDiveDeep && \
    CUDA_VISIBLE_DEVICES=4,5,6,7 mpirun --allow-run-as-root -np 4 \
    python3 kimi_k3_layer/tp_baseline/bench_layer.py correctness'

All measurement outputs go under kimi_k3_layer/tp_baseline/local_results/.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

_PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PACKAGE_DIR))

from harness import REPO_ROOT, setup_sys_path, verify_provenance  # noqa: E402

setup_sys_path()
sys.path.insert(0, str(REPO_ROOT / "kimi_k3_layer"))  # for bench_a2a_megamoe_* imports

import torch  # noqa: E402

from bench_moe.build import (  # noqa: E402
    _backend_name_from_module,
    _build_moe_module,
    _comm_method_name,
)
from bench_moe.mapping import _build_mapping_from_config, _build_model_config, _create_routing_method  # noqa: E402
from bench_moe.backend import MoeBackendType  # noqa: E402
from bench_moe.quantize import get_test_quant_params  # noqa: E402
from bench_moe.routing import _build_routing_plan, _project_router_logits_for_plan  # noqa: E402
from bench_moe.specs import ConfigSpec, ModelSpec, RoutingControlSpec  # noqa: E402
from bench_moe.utils import _set_device_from_local_rank  # noqa: E402
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402
from tensorrt_llm._utils import mpi_allgather, mpi_barrier, mpi_rank, mpi_world_size  # noqa: E402

from layer import (  # noqa: E402
    HIDDEN_SIZE,
    KimiK3MoELayerBaseline,
    KimiK3SharedExperts,
    MOE_INTERMEDIATE,
    SHARED_INTERMEDIATE,
)

# Reuse the proven LLMDiveDeep MegaMoE orchestration unchanged.
from bench_a2a_megamoe_pipeline import _build_module, _correctness  # noqa: E402
from bench_a2a_megamoe_sweep import _per_rank_tokens, _prepare_local_backend_weights  # noqa: E402

LOCAL_RESULTS = _PACKAGE_DIR / "local_results"
NUM_EXPERTS = 896
TOP_K = 16


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def _make_case_inputs(
    *,
    global_tokens: int,
    rank: int,
    world_size: int,
    num_experts: int,
    routing_method: Any,
    device: torch.device,
    seed: int,
    need_full: bool,
    need_all_logits: bool = False,
) -> dict[str, Any]:
    """Deterministic per-rank inputs; any rank can regenerate every slice."""
    per_rank = _per_rank_tokens(global_tokens, world_size)

    def _slice(src_rank: int) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(
            seed + global_tokens * 131 + src_rank
        )
        return torch.randn(
            (per_rank[src_rank], HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )

    hidden_local = _slice(rank)
    hidden_full = (
        torch.cat([_slice(r) for r in range(world_size)], dim=0) if need_full else None
    )

    routing_spec = RoutingControlSpec(
        routing_mode="forced",
        comm_pattern="balanced_alltoall",
        expert_pattern="balanced",
        seed=seed + global_tokens,
    )
    plan = _build_routing_plan(
        routing_spec,
        num_tokens=global_tokens,
        world_size=world_size,
        top_k=TOP_K,
        num_experts=num_experts,
        moe_ep_size=world_size,
    )

    def _logits(src_rank: int) -> torch.Tensor:
        logits, status, reason = _project_router_logits_for_plan(
            plan,
            src_rank=src_rank,
            routing_method=routing_method,
            num_experts=num_experts,
            top_k=TOP_K,
            experts_per_rank=num_experts // world_size,
            moe_ep_size=world_size,
            device=device,
            dtype=torch.bfloat16,
        )
        if status != "exact":
            raise RuntimeError(f"routing projection not exact: {status}: {reason}")
        return logits

    if os.environ.get("TPB_DEBUG_RANDOM_LOGITS") == "1":
        # Debug-only: bypass the balanced projection; plain random logits as
        # in the unchanged unittest (test_moe_module) correctness flow.
        def _logits(src_rank: int) -> torch.Tensor:  # noqa: F811
            generator = torch.Generator(device=device).manual_seed(
                seed + global_tokens * 977 + src_rank
            )
            return torch.randn(
                (per_rank[src_rank], num_experts),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )

    logits_local = _logits(rank)
    logits_full = (
        torch.cat([_logits(r) for r in range(world_size)], dim=0)
        if need_all_logits
        else None
    )
    return {
        "hidden_local": hidden_local,
        "hidden_full": hidden_full,
        "logits_local": logits_local,
        "logits_full": logits_full,
        "all_rank_num_tokens": per_rank,
    }


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.pstdev(values),
    }


@contextmanager
def _record_phases(
    layer: KimiK3MoELayerBaseline,
    current_iteration: list[int],
    records: dict[str, list[tuple[int, torch.cuda.Event, torch.cuda.Event]]],
) -> Iterator[None]:
    """CUDA-event wrap of each phase (pattern of bench_a2a_megamoe_pipeline)."""
    targets: dict[str, tuple[Any, str]] = {
        "shared": (layer, "shared_forward"),
        "dispatch": (layer.routed_moe.comm, "dispatch"),
        "expert_forward": (layer.routed_moe.backend, "run_moe"),
        "combine": (layer.routed_moe.comm, "combine"),
    }
    if layer.mode == "tp":
        targets["tp_allreduce"] = (layer, "allreduce_forward")
    originals: dict[str, tuple[Any, str, Callable[..., Any]]] = {}
    for phase, (owner, method_name) in targets.items():
        original = getattr(owner, method_name)
        originals[phase] = (owner, method_name, original)

        def wrapped(*args, _phase=phase, _original=original, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = _original(*args, **kwargs)
            end.record()
            records[_phase].append((current_iteration[0], start, end))
            return output

        setattr(owner, method_name, wrapped)
    try:
        yield
    finally:
        for owner, method_name, original in originals.values():
            if method_name in ("shared_forward", "allreduce_forward"):
                # Instance attribute shadowed the class method; remove shadow.
                if method_name in owner.__dict__:
                    del owner.__dict__[method_name]
            else:
                setattr(owner, method_name, original)


def _benchmark_layer(
    layer: KimiK3MoELayerBaseline,
    run: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            run()
    torch.cuda.synchronize()
    mpi_barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    phase_names = ["shared", "dispatch", "expert_forward", "combine"]
    if layer.mode == "tp":
        phase_names.append("tp_allreduce")
    phase_records: dict[str, list] = {name: [] for name in phase_names}
    current_iteration = [-1]

    with torch.inference_mode(), _record_phases(layer, current_iteration, phase_records):
        for idx in range(iters):
            current_iteration[0] = idx
            starts[idx].record()
            run()
            ends[idx].record()
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "total": _stats([starts[i].elapsed_time(ends[i]) for i in range(iters)])
    }
    phases: dict[str, dict[str, float]] = {}
    for phase, entries in phase_records.items():
        per_iteration = [0.0] * iters
        for iteration, start, end in entries:
            per_iteration[iteration] += start.elapsed_time(end)
        phases[phase] = _stats(per_iteration)
    result["phases"] = phases
    return result


def _capture_chrome_trace(
    run: Callable[[], torch.Tensor],
    out_path: Path,
    steady_iters: int = 3,
) -> str:
    """One steady-state iteration under torch.profiler, exported as a Chrome trace.

    Runs OUTSIDE the timed benchmark loop (a separate pass after timing) so
    profiler overhead can never contaminate the receipts' latency numbers.
    """
    with torch.inference_mode():
        for _ in range(steady_iters):
            run()
        torch.cuda.synchronize()
        mpi_barrier()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            run()
            torch.cuda.synchronize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out_path))
    return str(out_path)


def _run_layer_autotune(layer: KimiK3MoELayerBaseline, run: Callable[[], torch.Tensor]) -> str:
    """One untimed full-layer forward under autotune (fast settings).

    Mirrors the unchanged bench_moe ``_run_autotune`` but wraps the whole
    layer so the shared-expert Linear kernels are tuned too.
    """
    tuner = AutoTuner.get()
    saved = (tuner.warmup, tuner.repeat, tuner.stream_delay_micro_secs)
    tuner.warmup, tuner.repeat, tuner.stream_delay_micro_secs = 0, 1, 10
    cache_path = os.path.join(tempfile.gettempdir(), "tp_baseline_autotuner_cache.json")
    try:
        with torch.inference_mode(), autotune(cache_path=cache_path):
            run()
        torch.cuda.synchronize()
        return "success:fast"
    finally:
        tuner.warmup, tuner.repeat, tuner.stream_delay_micro_secs = saved


# --------------------------------------------------------------------------
# module builders
# --------------------------------------------------------------------------
def _build_routed_cached(
    *,
    mapping: Any,
    max_local_tokens: int,
    device: torch.device,
    weight_cache_dir: Path,
) -> torch.nn.Module:
    """Full-scale routed module with the cached deterministic local weights."""
    from _torch.modules.moe.quantize_utils import MXFP4MXFP8QuantizeUtil

    os.environ["MOE_BENCH_WEIGHT_CACHE_DIR"] = str(weight_cache_dir.resolve())
    model = ModelSpec(
        name="kimi_k3_shape",
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=MOE_INTERMEDIATE,
        quant_algo="W4A8_MXFP4_MXFP8",
        routing_method="RENORMALIZE",
    )
    original = MXFP4MXFP8QuantizeUtil.prepare_weights_from_backend
    MXFP4MXFP8QuantizeUtil.prepare_weights_from_backend = _prepare_local_backend_weights
    try:
        moe = _build_module(
            model,
            mapping,
            max_num_tokens=max(max_local_tokens, 1),
            device=device,
            backend="TRTLLM",
            comm_method="NVLINK_ONE_SIDED",
        )
    finally:
        MXFP4MXFP8QuantizeUtil.prepare_weights_from_backend = original
    if _backend_name_from_module(moe) != "TRTLLM":
        raise RuntimeError("routed path did not construct TRTLLMGenFusedMoE")
    if _comm_method_name(moe) != "NVLinkOneSided":
        raise RuntimeError(f"routed path built {_comm_method_name(moe)}, not NVLinkOneSided")
    return moe


def _build_routed_with_reference(
    *,
    num_experts: int,
    mapping: Any,
    max_local_tokens: int,
    device: torch.device,
):
    """Reduced-scale routed module + unchanged sequential reference module.

    Adapted from the unchanged ``bench_moe.build._build_moe_module`` (same
    calls, same order) but keeps the reference weights that the stock builder
    discards, so the smoke gate can compare against
    ``MXFP4MXFP8RefGatedMLPFusedMoE`` — a single-GPU per-expert loop of the
    same quantized math.
    """
    from bench_moe.build import _create_moe_for_benchmark
    from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    model = ModelSpec(
        name="kimi_k3_reduced",
        num_experts=num_experts,
        top_k=TOP_K,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=MOE_INTERMEDIATE,
        quant_algo="W4A8_MXFP4_MXFP8",
        routing_method="RENORMALIZE",
    )
    mc = model.to_moe_model_config()
    routing_method = _create_routing_method(
        model.routing_method_cls,
        top_k=mc.top_k,
        num_experts=mc.num_experts,
        bias_dtype=torch.bfloat16,
        profile_model_config=mc,
    )
    model_config = _build_model_config(
        model=model,
        mapping=mapping,
        moe_backend="TRTLLM",
        use_cuda_graph=False,
        max_num_tokens=max(max_local_tokens, 1),
        use_low_precision_moe_combine=False,
        dtype=torch.bfloat16,
    )
    probe_x = torch.randn((HIDDEN_SIZE // 32, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
        QuantAlgo.W4A8_MXFP4_MXFP8, probe_x, MoeBackendType("TRTLLM")
    )
    quant_kwargs.pop("ref_cls", None)
    quantize_util = quantize_util_cls(
        num_experts=mc.num_experts,
        dtype=torch.bfloat16,
        intermediate_size=mc.intermediate_size,
        hidden_size=mc.hidden_size,
        quant_config=quant_config,
        num_local_experts=mc.num_experts // mapping.moe_ep_size,
    )
    os.environ["TRTLLM_FORCE_COMM_METHOD"] = "NVLINK_ONE_SIDED"
    try:
        moe = _create_moe_for_benchmark(
            routing_method=routing_method,
            num_experts=mc.num_experts,
            hidden_size=mc.hidden_size,
            intermediate_size=mc.intermediate_size,
            dtype=torch.bfloat16,
            reduce_results=True,
            model_config=model_config,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
            bias=False,
        )
    finally:
        os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
    backend_weights, ref_weights, ref_module_kwargs = quantize_util.prepare_weights_from_backend(
        moe, **quant_kwargs
    )
    moe.load_weights([backend_weights])
    moe.post_load_weights()
    moe.cuda(f"cuda:{torch.cuda.current_device()}")
    if _comm_method_name(moe) != "NVLinkOneSided":
        raise RuntimeError(f"routed path built {_comm_method_name(moe)}, not NVLinkOneSided")

    ref_module = quantize_util.create_ref_module(routing_method, **ref_module_kwargs)
    ref_module.load_weights([ref_weights])
    ref_module.cuda(device)
    return moe, ref_module, routing_method


def _module_weight_bytes(module: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.data_ptr() in seen or not tensor.is_cuda:
            continue
        seen.add(tensor.data_ptr())
        total += tensor.numel() * tensor.element_size()
    return total


def _build_layers(
    *,
    modes: list[str],
    routed: torch.nn.Module,
    world_size: int,
    rank: int,
    device: torch.device,
) -> dict[str, KimiK3MoELayerBaseline]:
    layers = {}
    for mode in modes:
        shared = KimiK3SharedExperts(
            mode=mode, world_size=world_size, rank=rank, device=device
        )
        layers[mode] = KimiK3MoELayerBaseline(
            mode=mode, routed_moe=routed, shared_experts=shared, rank=rank
        )
    return layers


def _environment_snapshot(device: torch.device) -> dict[str, Any]:
    import tensorrt_llm

    props = torch.cuda.get_device_properties(device)
    return {
        "gpu": props.name,
        "device_index": device.index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "tensorrt_llm": tensorrt_llm.__version__,
        "clocks_locked": False,
        "provenance": verify_provenance(),
    }


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def run_correctness(args: argparse.Namespace) -> None:
    rank, world_size = mpi_rank(), mpi_world_size()
    device = torch.device("cuda", _set_device_from_local_rank())
    if args.num_experts % world_size:
        raise ValueError("--num-experts must divide world size")

    mapping = _build_mapping_from_config(
        ConfigSpec(backend="TRTLLM", parallel_mode="DEP"), world_size
    )
    AutoTuner.get().setup_distributed_state(mapping)
    max_local = max(
        max(_per_rank_tokens(tokens, world_size)) for tokens in args.tokens
    )
    routed, ref_routed, routing_method = _build_routed_with_reference(
        num_experts=args.num_experts,
        mapping=mapping,
        max_local_tokens=max_local,
        device=device,
    )
    layers = _build_layers(
        modes=["tp", "cp"], routed=routed, world_size=world_size, rank=rank, device=device
    )
    # Sequential single-GPU shared reference: TP1 GatedMLP, identical weights.
    ref_shared = KimiK3SharedExperts(mode="cp", world_size=1, rank=0, device=device)

    cases: dict[str, Any] = {}
    all_passed = True
    for global_tokens in args.tokens:
        inputs = _make_case_inputs(
            global_tokens=global_tokens,
            rank=rank,
            world_size=world_size,
            num_experts=args.num_experts,
            routing_method=routing_method,
            device=device,
            seed=args.seed,
            need_full=True,
            need_all_logits=True,
        )
        offset = sum(inputs["all_rank_num_tokens"][:rank])
        local_tokens = inputs["all_rank_num_tokens"][rank]
        if os.environ.get("TPB_DEBUG_COMPONENTS") == "1":
            with torch.inference_mode():
                dbg_ref_shared = ref_shared(inputs["hidden_full"]).float()
                dbg_ref_routed = ref_routed(
                    inputs["hidden_full"], inputs["logits_full"]
                ).float()
                dbg_layer_shared = layers["cp"].shared_forward(inputs["hidden_local"]).float()
                dbg_layer_routed = layers["cp"].routed_forward(
                    inputs["hidden_local"],
                    inputs["logits_local"],
                    inputs["all_rank_num_tokens"],
                ).float()
            torch.cuda.synchronize()
            sl = slice(offset, offset + local_tokens)
            for name, ref_t, got_t in (
                ("shared", dbg_ref_shared[sl], dbg_layer_shared),
                ("routed", dbg_ref_routed[sl], dbg_layer_routed),
            ):
                diff = (ref_t - got_t).abs()
                print(
                    f"[debug rank{rank} tokens={global_tokens}] {name}: "
                    f"ref_absmax={ref_t.abs().max().item():.2f} "
                    f"layer_absmax={got_t.abs().max().item():.2f} "
                    f"rmse={diff.square().mean().sqrt().item():.3f} "
                    f"cos={torch.nn.functional.cosine_similarity(ref_t.flatten(), got_t.flatten(), dim=0).item():.4f}",
                    flush=True,
                )
            if os.environ.get("TPB_DEBUG_ONLY") == "1":
                mpi_barrier()
                routed.destroy()
                return
        with torch.inference_mode():
            ref_total = (
                ref_shared(inputs["hidden_full"]).float()
                + ref_routed(inputs["hidden_full"], inputs["logits_full"]).float()
            )
            ref_local = ref_total[offset : offset + local_tokens]
            outputs = {}
            for mode, layer in layers.items():
                AutoTuner.get().clear_cache()
                run = lambda: layer.forward(  # noqa: E731
                    inputs["hidden_local"],
                    inputs["logits_local"],
                    inputs["all_rank_num_tokens"],
                    hidden_full=inputs["hidden_full"],
                )
                _run_layer_autotune(layer, run)
                outputs[mode] = run().clone()
        torch.cuda.synchronize()
        case: dict[str, Any] = {}
        for mode in ("tp", "cp"):
            metrics = _correctness(
                outputs[mode],
                ref_local,
                atol=args.atol,
                rtol=args.rtol,
                min_close_fraction=args.min_close_fraction,
            )
            gathered = mpi_allgather(metrics)
            case[f"{mode}_vs_single_gpu_ref"] = {
                f"rank{i}": m for i, m in enumerate(gathered)
            }
            all_passed &= all(m["passed"] for m in gathered)
        cross = _correctness(
            outputs["tp"],
            outputs["cp"],
            atol=args.atol,
            rtol=args.rtol,
            min_close_fraction=args.min_close_fraction,
        )
        gathered_cross = mpi_allgather(cross)
        case["tp_vs_cp"] = {f"rank{i}": m for i, m in enumerate(gathered_cross)}
        all_passed &= all(m["passed"] for m in gathered_cross)
        cases[str(global_tokens)] = case
        if rank == 0:
            print(f"[correctness] tokens={global_tokens} done", flush=True)

    receipt = {
        "status": "PASS" if all_passed else "FAIL",
        "kind": "correctness",
        "reference": (
            "sequential single-GPU MXFP4MXFP8RefGatedMLPFusedMoE (routed, unchanged "
            "TRT-LLM unittest fixture) + TP1 full-weight GatedMLP (shared), "
            "identical weights/inputs, reduced expert count"
        ),
        "shape": {
            "world_size": world_size,
            "num_experts": args.num_experts,
            "top_k": TOP_K,
            "hidden_size": HIDDEN_SIZE,
            "routed_intermediate": MOE_INTERMEDIATE,
            "shared_intermediate": SHARED_INTERMEDIATE,
            "tokens": args.tokens,
            "quant": "W4A8_MXFP4_MXFP8",
        },
        "tolerances": {
            "atol": args.atol,
            "rtol": args.rtol,
            "min_close_fraction": args.min_close_fraction,
        },
        "cases": cases,
        "environment": _environment_snapshot(device),
    }
    if rank == 0:
        out = LOCAL_RESULTS / f"correctness_w{world_size}_e{args.num_experts}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[correctness] status={receipt['status']} receipt={out}", flush=True)
    routed.destroy()
    if not all_passed:
        raise AssertionError("correctness gate failed; see receipt")


def run_measure(args: argparse.Namespace) -> None:
    rank, world_size = mpi_rank(), mpi_world_size()
    device = torch.device("cuda", _set_device_from_local_rank())
    if NUM_EXPERTS % world_size:
        raise ValueError(f"{NUM_EXPERTS} experts must divide world size {world_size}")

    mapping = _build_mapping_from_config(
        ConfigSpec(backend="TRTLLM", parallel_mode="DEP"), world_size
    )
    AutoTuner.get().setup_distributed_state(mapping)

    grid = [("prefill", tokens) for tokens in args.prefill_tokens] + [
        ("decode", batch) for batch in args.decode_batch_sizes
    ]
    max_local = max(
        max(_per_rank_tokens(size, world_size)) for _, size in grid
    )
    routed = _build_routed_cached(
        mapping=mapping,
        max_local_tokens=max_local,
        device=device,
        weight_cache_dir=args.weight_cache_dir,
    )
    layers = _build_layers(
        modes=list(args.modes),
        routed=routed,
        world_size=world_size,
        rank=rank,
        device=device,
    )
    routed_bytes = _module_weight_bytes(routed)
    shared_bytes = {
        mode: layer.shared_experts.weight_bytes() for mode, layer in layers.items()
    }

    rows = []
    for phase, size in grid:
        inputs = _make_case_inputs(
            global_tokens=size,
            rank=rank,
            world_size=world_size,
            num_experts=NUM_EXPERTS,
            routing_method=routed.routing_method,
            device=device,
            seed=args.seed,
            need_full="tp" in args.modes,
        )
        # One shared autotune pass per shape: the routed work is identical in
        # both modes, so both must run with identical expert-GEMM tactics.
        # Clearing between modes would let the fast autotuner pick different
        # tactics for the same shapes and skew the comparison.
        AutoTuner.get().clear_cache()
        mode_runs = {}
        for mode in args.modes:
            layer = layers[mode]
            mode_runs[mode] = lambda layer=layer: layer.forward(
                inputs["hidden_local"],
                inputs["logits_local"],
                inputs["all_rank_num_tokens"],
                hidden_full=inputs["hidden_full"],
            )
            _run_layer_autotune(layer, mode_runs[mode])
        for mode in args.modes:
            layer = layers[mode]
            run = mode_runs[mode]
            with torch.inference_mode():
                first = run()
                finite = bool(torch.isfinite(first).all())
            torch.cuda.synchronize()
            mpi_barrier()
            timing = _benchmark_layer(layer, run, args.warmup, args.iters)
            trace_path = None
            if args.trace:
                trace_path = _capture_chrome_trace(
                    run,
                    LOCAL_RESULTS
                    / "traces"
                    / (
                        f"tp_baseline_{phase}_{mode}_bs{size}"
                        f"_w{world_size}_rank{rank}.json"
                    ),
                )
            gathered = mpi_allgather(timing)
            all_finite = all(mpi_allgather(finite))
            row = {
                "phase": phase,
                "global_size": size,
                "mode": mode,
                "finite": all_finite,
                "score_median_ms": max(t["total"]["median_ms"] for t in gathered),
                "score_mean_ms": max(t["total"]["mean_ms"] for t in gathered),
                "phase_median_ms": {
                    name: max(t["phases"][name]["median_ms"] for t in gathered)
                    for name in gathered[0]["phases"]
                },
                "per_rank": {f"rank{i}": t for i, t in enumerate(gathered)},
                "chrome_traces": {
                    "note": "captured in a separate pass AFTER the timed "
                    "iterations; profiler overhead never touches the numbers "
                    "above",
                    "per_rank": mpi_allgather(trace_path),
                }
                if args.trace
                else None,
            }
            rows.append(row)
            if rank == 0:
                print(
                    f"[measure] {phase} size={size} mode={mode}: "
                    f"{row['score_median_ms']:.3f} ms median "
                    f"(phases {row['phase_median_ms']})",
                    flush=True,
                )

    receipt = {
        "kind": "measure",
        "label": {
            "tp": "finalized TRT-LLM reference (shared TP-sharded + allreduce)",
            "cp": "CP emulation (tokens split, shared replicated, no TP collective)",
        },
        "shape": {
            "world_size": world_size,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "hidden_size": HIDDEN_SIZE,
            "routed_intermediate": MOE_INTERMEDIATE,
            "shared_intermediate": SHARED_INTERMEDIATE,
            "quant": "W4A8_MXFP4_MXFP8",
            "activation": "SwiGLU",
            "routing": "forced balanced_alltoall projected through native routing",
        },
        "timing": {"warmup": args.warmup, "iters": args.iters, "launch": "eager"},
        "weights": {
            "routed_bytes_per_gpu": routed_bytes,
            "shared_bytes_per_gpu": shared_bytes,
            "note": (
                "Routed experts: cached deterministic seed-42 local weights "
                "(same cache as the MegaMoE receipts); logical experts repeat "
                "every num_experts/world across ranks. Shared experts: "
                "unchanged MXFP4MXFP8QuantizeUtil.create_weights (internal "
                "seed 42, num_experts=1, intermediate 6144)."
            ),
        },
        "rows": rows,
        "environment": _environment_snapshot(device),
    }
    if rank == 0:
        out = LOCAL_RESULTS / f"{args.tag}_w{world_size}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[measure] receipt={out}", flush=True)
    routed.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    corr = sub.add_parser("correctness")
    corr.add_argument("--num-experts", type=int, default=64)
    corr.add_argument("--tokens", type=int, nargs="+", default=[256, 4096])
    corr.add_argument("--seed", type=int, default=1234)
    corr.add_argument("--atol", type=float, default=0.3)
    corr.add_argument("--rtol", type=float, default=0.15)
    corr.add_argument("--min-close-fraction", type=float, default=0.85)

    meas = sub.add_parser("measure")
    meas.add_argument("--prefill-tokens", type=int, nargs="*", default=[8192, 32768])
    meas.add_argument("--decode-batch-sizes", type=int, nargs="*", default=[64, 512])
    meas.add_argument("--modes", nargs="+", choices=("tp", "cp"), default=["tp", "cp"])
    meas.add_argument("--warmup", type=int, default=20)
    meas.add_argument("--iters", type=int, default=100)
    meas.add_argument("--seed", type=int, default=1234)
    meas.add_argument("--tag", default="cp_vs_tp")
    meas.add_argument(
        "--trace",
        action="store_true",
        help="after timing each case, capture one steady-state iteration per "
        "rank with torch.profiler and export Chrome traces under "
        "local_results/traces/",
    )
    meas.add_argument(
        "--weight-cache-dir", type=Path, default=REPO_ROOT / "out" / "moe_weight_cache"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if mpi_world_size() < 2:
        raise RuntimeError("launch under mpirun with at least 2 ranks")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29873")
    torch.manual_seed(args.seed + mpi_rank())
    torch.cuda.manual_seed_all(args.seed + mpi_rank())
    if args.command == "correctness":
        run_correctness(args)
    else:
        run_measure(args)


if __name__ == "__main__":
    main()
