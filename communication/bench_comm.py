#!/usr/bin/env python3
"""Correctness checks and report-ready benchmarks for every collective.

  # correctness + all backend benchmarks:
  docker exec trt-dev bash -c "cd /workspace/LLMDiveDeep && \
      mpirun --allow-run-as-root -np 8 python3 \
      communication/bench_comm.py"

  # + generate local_result/lookup.{md,csv} and the preloadable JSON map:
  ... python3 communication/bench_comm.py \
      --ops allreduce,allgather,reducescatter,all-to-all,\
allreduce-norm,gemm-allreduce,allreduce-norm-gemm \
      --bs 1..8k --dims 7168 --enable-b10

Correctness: every available backend is checked against a plain-NCCL
(+ fp32-norm) reference and called twice to exercise resource reuse.
Kernel-specific eager and CUDA-graph replay checks live in
``communication/kernel_benchmarks``.

Machinery: single-node guard (positive + faked cross-node negative),
lazy b10 mover build via explicit impl=, autotune map fill + all-rank
agreement + pow2 buckets + small-message dispatch, and the oversize->seq
capacity fallback.

Benchmarking times every backend per (op, dim, bs) cell via
the same ``_profile_cell`` machinery production autotune uses (tie
re-measure, contamination guard, own-kernel margin gate — the winners
ARE production dispatch) and writes one markdown table per (op, dim),
a long CSV (with bus bandwidth for the pure comm ops), and a JSON map
for :meth:`Collectives.import_auto_map`. Op names take aliases; batch
sizes take k-suffixes (``8k``) and pow2 ranges (``1..8k``).
Pass ``--check-only`` to run correctness and dispatch checks without timing.

Standalone custom-kernel checks live in
``communication/kernel_benchmarks``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from communication.collective import Collectives  # noqa: E402
from communication.planner import _next_pow2, _OP_IMPLS  # noqa: E402

_OP_ALIASES = {
    "allreduce": "all_reduce", "ar": "all_reduce",
    "allgather": "all_gather", "ag": "all_gather",
    "allgatherquant": "all_gather_col_quant",
    "allgathercolquant": "all_gather_col_quant",
    "agquant": "all_gather_col_quant",
    "quantizedallreduce": "quantized_all_reduce",
    "quantallreduce": "quantized_all_reduce",
    "qar": "quantized_all_reduce",
    "reducescatter": "reduce_scatter", "rs": "reduce_scatter",
    "alltoall": "all_to_all", "a2a": "all_to_all",
    "normreduce": "allreduce_norm", "allreducenorm": "allreduce_norm",
    "arnorm": "allreduce_norm",
    "gemm": "gemm_allreduce", "gemmallreduce": "gemm_allreduce",
    "gemmar": "gemm_allreduce",
    "allreducenormgemm": "allreduce_norm_gemm",
    "arnormgemm": "allreduce_norm_gemm", "arng": "allreduce_norm_gemm",
}
# ops whose operand is [world*B, dim]
_WORLD_SIZED = ("all_gather", "reduce_scatter", "all_to_all")
NVLINK_GBPS = 900.0


def parse_ops(spec: str) -> list[str]:
    ops = []
    for token in spec.split(","):
        key = token.strip().lower().replace("-", "").replace("_", "")
        if not key:
            continue
        op = _OP_ALIASES.get(key, token.strip().lower().replace("-", "_"))
        if op not in _OP_IMPLS:
            raise SystemExit(
                f"unknown op {token!r}; known: {sorted(_OP_IMPLS)}")
        ops.append(op)
    return ops


def _parse_size(token: str) -> int:
    token = token.strip().lower()
    if token.endswith("k"):
        return int(float(token[:-1]) * 1024)
    if token.endswith("m"):
        return int(float(token[:-1]) * 1024 * 1024)
    return int(token)


def parse_bs(spec: str) -> list[int]:
    """``1,2,4,8k`` and pow2 ranges ``1..8k`` (=1,2,4,...,8192)."""
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ".." in token:
            lo, hi = (_parse_size(t) for t in token.split("..", 1))
            b = max(1, lo)
            while b <= hi:
                out.add(b)
                b <<= 1
        else:
            out.add(_parse_size(token))
    return sorted(out)


def busbw_gbps(op: str, world: int, bs: int, dim: int, us: float) -> float:
    """Communication-equivalent bus bandwidth for ops with a clear payload."""
    if op == "allreduce_norm_gemm":
        return float("nan")
    if op == "quantized_all_reduce":
        numel = bs * dim
        packed = numel + 4 * math.ceil(numel / 256)
        wire_bytes = packed * 2 * (world - 1) / world
        return wire_bytes / (us * 1e3)
    bytes_full = bs * dim * 2  # bf16
    if op in _WORLD_SIZED or op == "all_gather_col_quant":
        bytes_full *= world
        if op == "all_gather_col_quant":
            bytes_full //= world  # dim already names the gathered width
        algo = bytes_full * (world - 1) / world
    else:  # all_reduce, allreduce_norm, gemm_allreduce
        algo = bytes_full * 2 * (world - 1) / world
    return algo / (us * 1e3)


def payload_bytes(op: str, world: int, bs: int, dim: int) -> int:
    rows = bs * (world if op in _WORLD_SIZED else 1)
    return rows * dim * 2  # bfloat16


def compute_tflops(comm: Collectives, op: str, bs: int, dim: int,
                   us: float) -> float:
    if op == "gemm_allreduce":
        flops = 2 * bs * comm._boundary_gemm_k * dim
    elif op == "allreduce_norm_gemm":
        flops = 2 * bs * dim * comm._boundary_gemm_n
    else:
        return float("nan")
    return flops / (us * 1e6)


def quantization_error(
    comm: Collectives,
    bs: int,
    dim: int,
    impl: str,
) -> dict[str, float | bool]:
    gen = torch.Generator(device="cuda").manual_seed(9107 + comm.rank)
    x = (0.02 * torch.randn(
        bs, dim, generator=gen, device="cuda")).to(torch.bfloat16)
    ref = x.clone()
    dist.all_reduce(ref, group=comm.group)
    out = comm.quantized_all_reduce(x, impl=impl)
    diff = out.float() - ref.float()
    ref32 = ref.float()
    ref_norm = torch.linalg.vector_norm(ref32)
    out_norm = torch.linalg.vector_norm(out.float())
    denom = torch.clamp(ref_norm * out_norm, min=1e-20)
    return {
        "max_abs_error": diff.abs().max().item(),
        "mean_abs_error": diff.abs().mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "relative_l2": (
            torch.linalg.vector_norm(diff) / ref_norm.clamp_min(1e-20)
        ).item(),
        "cosine_similarity": ((ref32 * out.float()).sum() / denom).item(),
        "lossy_transport_active": not torch.equal(out, ref),
    }


def winner_reason(op: str, impl: str, bs: int) -> str:
    """Compact causal hypothesis tied to the measured winner."""
    if op == "allreduce_norm":
        if impl == "seq":
            return "separate tuned AR and RMSNorm avoid fused-kernel overhead"
        return "fused AR+norm removes one launch and one global-memory pass"
    if op == "quantized_all_reduce":
        return "FP8 wire halves value bytes; quantize/dequantize cost is amortized"
    if op == "gemm_allreduce":
        if impl == "seq":
            return "vendor GEMM and tuned AR beat fused setup/padding overhead"
        return "GEMM epilogue overlaps tile reduction after fixed costs amortize"
    if op == "allreduce_norm_gemm":
        if impl == "seq":
            return "vendor kernels win before fusion setup is amortized"
        if impl == "norm_gemm_seq":
            return "fused AR+norm saves traffic while vendor GEMM stays efficient"
        return "M-split fuses reduce+norm and executes only rank-owned GEMM tiles"
    if op == "all_gather_col_quant":
        if impl == "col_quant":
            return "quantized write-out removes the separate quant launch/read pass"
        if impl == "col_seq":
            return "tuned column AG wins despite a separate quant kernel"
        return "NCCL's low fixed overhead beats the bounded custom pipeline"
    if impl in ("trt", "flashinfer:1shot"):
        return "specialized low-latency protocol minimizes small-message setup"
    if impl.startswith(("b10_", "torch_symm")):
        return "direct NVLink/NVLS path amortizes setup and raises payload bandwidth"
    if impl == "nccl_symm":
        return "registered symmetric buffers improve NCCL transport setup"
    if impl == "nccl":
        return "vendor collective has the best transport schedule at this shape"
    if impl == "cutedsl":
        return "single fused persistent kernel amortizes launch and boundary traffic"
    return "lowest measured max-rank latency"


# --------------------------------------------------------- correctness

def check_correctness(comm: Collectives, op: str, bs: int, dim: int,
                      impls: list[str]) -> list[str]:
    """Every backend vs NCCL at one size, including repeated calls."""
    world = comm.world
    rows = bs * world if op in _WORLD_SIZED else bs
    if op == "all_gather_col_quant":
        if dim % world:
            raise ValueError(
                "all_gather_col_quant output dim must divide world size")
        probe = torch.empty(
            (bs, dim // world), device="meta", dtype=torch.bfloat16)
    else:
        probe = torch.empty((rows, dim), device="meta", dtype=torch.bfloat16)
    impls = [i for i in impls if comm._fits(op, i, probe)]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(17 + comm.rank)

    def rand(*shape):
        return (0.02 * torch.randn(*shape, generator=gen,
                                   device="cuda")).to(torch.bfloat16)

    def twice(fn):
        fn()
        return fn()

    def announce(impl: str) -> None:
        if comm.rank == 0:
            print(
                f"[check] op={op} dim={dim} bs={bs} backend={impl}",
                flush=True,
            )

    ok: list[str] = []
    if op == "all_reduce":
        x = rand(bs, dim)
        ref = x.clone()
        dist.all_reduce(ref, group=comm.group)
        for impl in impls:
            announce(impl)
            y = twice(lambda: comm.all_reduce(x.clone(), impl=impl)).clone()
            assert (y.float() - ref.float()).abs().max() < 0.25, (op, impl)
            ok.append(impl)
    elif op == "quantized_all_reduce":
        x = rand(bs, dim)
        ref = x.clone()
        dist.all_reduce(ref, group=comm.group)
        for impl in impls:
            announce(impl)
            y = twice(
                lambda: comm.quantized_all_reduce(x, impl=impl)).clone()
            assert y.shape == x.shape and torch.isfinite(y).all(), (op, impl)
            diff = y.float() - ref.float()
            relative_l2 = (
                torch.linalg.vector_norm(diff)
                / torch.linalg.vector_norm(ref.float()).clamp_min(1e-20)
            )
            assert relative_l2 < 0.1, (op, impl, relative_l2.item())
            ok.append(impl)
    elif op == "all_gather":
        x = rand(bs, dim)
        ref = torch.empty(world * bs, dim, device="cuda",
                          dtype=torch.bfloat16)
        dist.all_gather_into_tensor(ref, x, group=comm.group)
        for impl in impls:
            announce(impl)
            y = twice(lambda: comm.all_gather(x, impl=impl)).clone()
            assert torch.equal(y, ref), (op, impl)
            ok.append(impl)
    elif op == "all_gather_col_quant":
        x = rand(bs, dim // world)
        shards = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(shards, x, group=comm.group)
        gathered = torch.cat(shards, dim=1)
        ref, ref_scale = torch.ops.trtllm.mxfp8_quantize(
            gathered.contiguous(), False)
        ref_scale = ref_scale.view(bs, -1)
        for impl in impls:
            announce(impl)
            y, scale = twice(
                lambda: comm.all_gather_col_quant(x, impl=impl))
            assert torch.equal(y, ref), (op, impl)
            assert torch.equal(scale, ref_scale), (op, impl, "scale")
            ok.append(impl)
    elif op == "reduce_scatter":
        x = rand(world * bs, dim)
        ref = torch.empty(bs, dim, device="cuda", dtype=torch.bfloat16)
        dist.reduce_scatter_tensor(ref, x, group=comm.group)
        for impl in impls:
            announce(impl)
            y = twice(lambda: comm.reduce_scatter(x, impl=impl)).clone()
            assert (y.float() - ref.float()).abs().max() < 0.25, (op, impl)
            ok.append(impl)
    elif op == "all_to_all":
        x = rand(world * bs, dim)
        ref = torch.empty_like(x)
        dist.all_to_all_single(ref, x, group=comm.group)
        for impl in impls:
            announce(impl)
            y = twice(lambda: comm.all_to_all(x, impl=impl)).clone()
            assert torch.equal(y, ref), (op, impl)
            ok.append(impl)
    elif op == "allreduce_norm":
        x = rand(bs, dim)
        residual = rand(bs, dim)
        gamma = torch.ones(dim, device="cuda", dtype=torch.bfloat16)
        dist.broadcast(residual, src=0, group=comm.group)
        ar = x.clone()
        dist.all_reduce(ar, group=comm.group)
        ref = F.rms_norm((ar + residual).float(), (dim,), gamma.float(),
                         1e-6).to(torch.bfloat16)
        ref_res = ar + residual
        for impl in impls:
            announce(impl)
            y, res = twice(lambda: comm.allreduce_norm(
                x, gamma, 1e-6, residual=residual, impl=impl))
            y = y.clone()
            assert (y.float() - ref.float()).abs().max() < 0.1, (op, impl)
            if res is not None:  # backends may skip it for zero-res path
                rerr = (res.float() - ref_res.float()).abs().max()
                assert rerr < 0.25, (op, impl, rerr)
            ok.append(impl)
    elif op == "gemm_allreduce":
        x = rand(bs, comm._boundary_gemm_k)
        w = rand(dim, comm._boundary_gemm_k)
        dist.broadcast(w, src=0, group=comm.group)
        ref = torch.mm(x, w.T)
        dist.all_reduce(ref, group=comm.group)
        for impl in impls:
            announce(impl)
            y = twice(lambda: comm.gemm_allreduce(x, w, impl=impl)).clone()
            assert (y.float() - ref.float()).abs().max() < 0.25, (op, impl)
            ok.append(impl)
    elif op == "allreduce_norm_gemm":
        x = rand(bs, dim)
        residual = rand(bs, dim)
        gamma = torch.ones(dim, device="cuda", dtype=torch.bfloat16)
        w = rand(comm._boundary_gemm_n, dim)
        for t in (residual, w):
            dist.broadcast(t, src=0, group=comm.group)
        ar = x.clone()
        dist.all_reduce(ar, group=comm.group)
        resf = (ar + residual).float()
        rms = resf.pow(2).mean(-1, keepdim=True).add_(1e-5).rsqrt_()
        ref = ((resf * rms).to(torch.bfloat16) * gamma) @ w.T
        ref_res = ar + residual
        for impl in impls:
            announce(impl)
            y, res = twice(lambda: comm.allreduce_norm_gemm(
                x, residual, gamma, w, impl=impl))
            y = y.clone()
            assert (y.float() - ref.float()).abs().max() < 0.35, (op, impl)
            rerr = (res.float() - ref_res.float()).abs().max()
            assert rerr < 0.25, (op, impl, rerr)
            ok.append(impl)
    return ok


def run_correctness(comm, ops, bs_list, dims, col_quant_dim, log) -> None:
    world = comm.world
    for op in ops:
        candidates = comm._op_candidates(op)
        op_dims = [col_quant_dim] if op == "all_gather_col_quant" else dims
        for dim in op_dims:
            # smallest bs, plus the smallest world-aligned bs ABOVE
            # world so world-gated impls (2shot / b10:sm) get covered
            small = min([b for b in bs_list if b >= 2] or bs_list)
            aligned = [b for b in bs_list if b % world == 0 and b > world]
            checked: set[str] = set()
            if op == "quantized_all_reduce":
                supported = [
                    b for b in bs_list
                    if b * dim >= 8192 and (b * dim) % 8 == 0
                ]
                check_sizes = supported[:1]
            else:
                check_sizes = sorted({small, *aligned[:1]})
            for check_bs in check_sizes:
                checked |= set(check_correctness(
                    comm, op, check_bs, dim, candidates))
            verdict = (
                "approximate outputs finite; error measured in report"
                if op == "quantized_all_reduce"
                else "backends match NCCL reference"
            )
            log(f"[check] {op} dim={dim}: {len(checked)}/"
                f"{len(candidates)} {verdict}")


# ----------------------------------------------------------- machinery

def run_machinery(comm, rank, world, dim, bs_list, log) -> None:
    """Dispatch-machinery checks (ported from the old test_autotune +
    test_boundary_ops). Runs LAST: the mover build changes candidate
    sets and autotune() clears the map."""
    # --- single-node guard: opt-in positive + faked cross-node negative
    comm.assert_single_node()
    import socket
    real = socket.gethostname
    socket.gethostname = lambda: f"fake-node{rank % 2}"
    try:
        comm.assert_single_node()
        raise AssertionError("cross-node host set did not raise")
    except RuntimeError as e:
        assert "NVLink domain" in str(e), e
    finally:
        socket.gethostname = real
    log("ok: assert_single_node (positive + faked cross-node)")

    # --- oversize operands resolve to the plain/seq fallback, no raise
    big = 2 * (comm.max_numel // dim)
    probe = torch.empty((big, dim), device="meta", dtype=torch.bfloat16)
    assert comm._resolve("gemm_allreduce", probe, "auto") == "seq"
    assert comm._resolve("allreduce_norm_gemm", probe, "auto") == "seq"
    xb = (0.02 * torch.randn(big, comm._boundary_gemm_k,
                             device="cuda")).to(torch.bfloat16)
    wb = (0.02 * torch.randn(dim, comm._boundary_gemm_k,
                             device="cuda")).to(torch.bfloat16)
    assert comm.gemm_allreduce(xb, wb).shape == (big, dim)
    log("ok: oversize operands fall back to seq")

    # --- explicit impl= lazily builds the b10 movers (collective)
    mover_rows = max(1, min(8, comm.max_numel // (dim * world)))
    x = torch.randn(mover_rows, dim, device="cuda", dtype=torch.bfloat16)
    y_dma = comm.all_gather(x, impl="b10_copy_engine")
    y_sm = comm.all_gather(x, impl="b10_copy_engine:sm")
    assert y_dma.shape == y_sm.shape == (mover_rows * world, dim)
    log("ok: explicit impl= builds the b10 movers lazily")

    # --- autotune: map fills, pow2 buckets, all ranks agree, and a
    # A specialized implementation (never NCCL) wins small buckets.
    amap = comm.autotune(dims=(dim,), max_tokens=64,
                         ops=("all_reduce",), warmup=2, iters=15,
                         log=False)
    assert amap, "autotune produced an empty map"
    local = [(o, d, b, impl) for (o, d, b), (impl, _) in sorted(amap.items())]
    obj = [local]
    dist.broadcast_object_list(obj, src=0)
    assert local == obj[0], f"rank {rank} map disagrees with rank 0"
    for (_, _, bucket) in amap:
        assert bucket == _next_pow2(bucket), bucket
    if comm._flashinfer is not None:
        for b in (1, 2, 4, 8, 16, 32):
            hit = amap.get(("all_reduce", dim, b))
            if hit is not None:
                assert hit[0] != "nccl", (b, hit)
    log("ok: autotune map fills, ranks agree, small buckets avoid nccl")


# --------------------------------------------------------------- bench

def run_bench(comm, args, ops, bs_list, dims, world, log) -> None:
    rows: list[dict] = []
    gpu = torch.cuda.get_device_name()
    md: list[str] = [
        "# Communication benchmark",
        "",
        f"- GPU: `{gpu}`",
        f"- world size: `{world}`",
        "- dtype: `bfloat16`",
        "- latency: median of interleaved CUDA-event windows, "
        "max-reduced across ranks",
        f"- communication speed-of-light assumption: `{NVLINK_GBPS:.0f} GB/s` "
        "per GPU",
        "- measured values and theoretical estimates are reported separately",
        "- quantized AllReduce algorithm BW is BF16-equivalent; bus BW, SOL, "
        "and efficiency use actual 8-bit values plus FP32 group scales",
        f"- CuTeDSL fused candidates: "
        f"`{'enabled' if args.enable_cutedsl else 'opt-in; skipped'}` "
        "(AOT AR+norm and GEMM+AR; AR+norm+GEMM is excluded)",
        f"- b10 copy-engine movers: "
        f"`{'enabled' if args.enable_b10 else 'disabled'}`",
        "",
    ]
    for op in ops:
        candidates = comm._op_candidates(op)
        op_dims = ([args.col_quant_dim]
                   if op == "all_gather_col_quant" else dims)
        for dim in op_dims:
            md += [f"## `{op}` dim={dim}", "",
                   "| bs | " + " | ".join(candidates)
                   + " | winner | why |",
                   "|--:|" + "---:|" * len(candidates) + "---|---|"]
            details = [
                "",
                "### Detailed measurements",
                "",
                "| bs | backend | latency (us) | payload (bytes) | "
                "algorithm BW (GB/s) | bus BW (GB/s) | SOL (us) | "
                "efficiency | compute (TFLOP/s) | speedup vs best seq | "
                "correct | notes |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
                ":---:|---|",
            ]
            for bs in bs_list:
                need = bs * dim * (world if op in _WORLD_SIZED else 1)
                if need > comm.max_numel:
                    continue
                if args.report:
                    warmup = 1
                    iters = 6 if bs <= 256 else (4 if bs <= 2048 else 2)
                else:
                    warmup = 2
                    iters = 12 if bs <= 256 else (6 if bs <= 2048 else 3)
                winner, us, cell = comm._profile_cell(
                    op, dim, bs, candidates, warmup=warmup, iters=iters)
                if winner is None:
                    if op == "quantized_all_reduce":
                        md.append(
                            f"| {bs} | — | N/A | vLLM quantized AR "
                            "requires at least 8192 elements |")
                    continue
                comm._auto_map[(op, dim, bs)] = (winner, us)
                timed = dict(cell)
                quant_errors = (
                    {
                        impl: quantization_error(comm, bs, dim, impl)
                        for impl, _ in cell
                    }
                    if op == "quantized_all_reduce"
                    else {}
                )
                seq_times = [
                    t for name, t in cell
                    if name == "seq" or name.endswith("_seq")
                ]
                seq_us = min(seq_times) if seq_times else float("nan")
                for impl, t in cell:
                    error = quant_errors.get(impl, {})
                    payload = payload_bytes(op, world, bs, dim)
                    algorithm_bw = payload / (t * 1e3)
                    bus_bw = busbw_gbps(op, world, bs, dim, t)
                    has_bus_model = bus_bw == bus_bw  # false only for NaN
                    sol_us = (
                        payload * (bus_bw / algorithm_bw)
                        / (NVLINK_GBPS * 1e3)
                        if has_bus_model
                        else float("nan")
                    )
                    efficiency = (
                        100 * bus_bw / NVLINK_GBPS
                        if has_bus_model
                        else float("nan")
                    )
                    tflops = compute_tflops(comm, op, bs, dim, t)
                    has_compute = tflops == tflops
                    speedup = seq_us / t if seq_us == seq_us else float("nan")
                    has_speedup = speedup == speedup
                    rows.append(dict(
                        op=op,
                        dim=dim,
                        bs=bs,
                        shape=f"{bs}x{dim}",
                        backend=impl,
                        latency_us=round(t, 3),
                        payload_bytes=payload,
                        algorithm_bandwidth_gbps=round(algorithm_bw, 3),
                        bus_bandwidth_gbps=(
                            round(bus_bw, 3) if has_bus_model else "n/a"
                        ),
                        speed_of_light_us=(
                            round(sol_us, 3) if has_bus_model else "n/a"
                        ),
                        speed_of_light_bandwidth_gbps=(
                            NVLINK_GBPS if has_bus_model else "n/a"
                        ),
                        efficiency_pct=(
                            round(efficiency, 2) if has_bus_model else "n/a"
                        ),
                        compute_tflops=(
                            round(tflops, 3) if has_compute else "n/a"
                        ),
                        speedup_vs_best_sequential=(
                            round(speedup, 3) if has_speedup else "n/a"
                        ),
                        correctness=(
                            "approximate"
                            if op == "quantized_all_reduce" else True
                        ),
                        max_abs_error=error.get("max_abs_error", "n/a"),
                        mean_abs_error=error.get("mean_abs_error", "n/a"),
                        rmse=error.get("rmse", "n/a"),
                        relative_l2=error.get("relative_l2", "n/a"),
                        cosine_similarity=error.get(
                            "cosine_similarity", "n/a"),
                        lossy_transport_active=error.get(
                            "lossy_transport_active", "n/a"),
                        winner=impl == winner,
                        world_size=world,
                        gpu=gpu,
                        dtype="bfloat16",
                        iterations=iters,
                        notes=("winner: " + winner_reason(op, impl, bs)
                               if impl == winner else ""),
                    ))
                    tflops_cell = f"{tflops:.2f}" if has_compute else "n/a"
                    speedup_cell = f"{speedup:.2f}x" if has_speedup else "n/a"
                    details.append(
                        f"| {bs} | {impl} | {t:.3f} | {payload} | "
                        f"{algorithm_bw:.2f} | "
                        f"{bus_bw:.2f} | {sol_us:.3f} | {efficiency:.1f}% | "
                        f"{tflops_cell} | {speedup_cell} | "
                        f"{'approx' if error else 'yes'} | "
                        f"{'winner' if impl == winner else ''} |"
                        if has_bus_model
                        else
                        f"| {bs} | {impl} | {t:.3f} | {payload} | "
                        f"{algorithm_bw:.2f} | n/a | n/a | n/a | "
                        f"{tflops_cell} | {speedup_cell} | "
                        f"{'approx' if error else 'yes'} | "
                        f"{'winner; ' if impl == winner else ''}"
                        "fused/compute op |"
                    )
                md.append(
                    f"| {bs} | "
                    + " | ".join(
                        f"**{timed[c]:.1f}**" if c == winner
                        else (f"{timed[c]:.1f}" if c in timed else "—")
                        for c in candidates)
                    + f" | {winner} | {winner_reason(op, winner, bs)} |")
                log(f"{op:<19} dim={dim} bs={bs:<6} -> {winner:<14} "
                    f"{us:7.1f}us")
            md.extend(["", *details, ""])

    quant_comparisons: list[dict] = []
    quant_rows = [r for r in rows if r["op"] == "quantized_all_reduce"]
    exact_rows = [r for r in rows if r["op"] == "all_reduce"]
    if "quantized_all_reduce" in ops and exact_rows:
        md += [
            "## Quantized versus exact BF16 AllReduce",
            "",
            "The vLLM/Kraken kernels use per-group INT8 or E4M3 values plus "
            "scales on the wire and return BF16. They are approximate and "
            "are never eligible for exact `all_reduce(auto)` dispatch.",
            "",
            "| bs | quantized | exact backend | quantized (us) | exact (us) | "
            "speedup vs exact | speedup vs best exact | max abs error | "
            "RMSE | relative L2 | cosine | active |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        q_by_shape: dict[tuple[int, int], list[dict]] = {}
        for row in quant_rows:
            q_by_shape.setdefault((row["bs"], row["dim"]), []).append(row)
        exact_shapes = sorted({(r["bs"], r["dim"]) for r in exact_rows})
        for bs, dim in exact_shapes:
            qs = q_by_shape.get((bs, dim), [])
            matching = [
                r for r in exact_rows
                if r["bs"] == bs and r["dim"] == dim
            ]
            best_exact = min(matching, key=lambda r: r["latency_us"])
            if not qs:
                for exact in matching:
                    comparison = {
                        "bs": bs,
                        "dim": dim,
                        "quantized_backend": "vllm_int8/vllm_fp8",
                        "exact_backend": exact["backend"],
                        "quantized_latency_us": "n/a",
                        "exact_latency_us": exact["latency_us"],
                        "speedup_vs_exact": "n/a",
                        "best_exact_backend": best_exact["backend"],
                        "speedup_vs_best_exact": "n/a",
                        "max_abs_error": "n/a",
                        "mean_abs_error": "n/a",
                        "rmse": "n/a",
                        "relative_l2": "n/a",
                        "cosine_similarity": "n/a",
                        "lossy_transport_active": False,
                    }
                    quant_comparisons.append(comparison)
                    md.append(
                        f"| {bs} | vllm_int8/vllm_fp8 | {exact['backend']} | "
                        f"N/A | {exact['latency_us']:.3f} | N/A | N/A | "
                        "N/A | N/A | N/A | N/A | unsupported (<8192 elements) |"
                    )
                continue
            for q in qs:
                for exact in matching:
                    comparison = {
                        "bs": q["bs"],
                        "dim": q["dim"],
                        "quantized_backend": q["backend"],
                        "exact_backend": exact["backend"],
                        "quantized_latency_us": q["latency_us"],
                        "exact_latency_us": exact["latency_us"],
                        "speedup_vs_exact": round(
                            exact["latency_us"] / q["latency_us"], 3),
                        "best_exact_backend": best_exact["backend"],
                        "speedup_vs_best_exact": round(
                            best_exact["latency_us"] / q["latency_us"], 3),
                        "max_abs_error": q["max_abs_error"],
                        "mean_abs_error": q["mean_abs_error"],
                        "rmse": q["rmse"],
                        "relative_l2": q["relative_l2"],
                        "cosine_similarity": q["cosine_similarity"],
                        "lossy_transport_active": q[
                            "lossy_transport_active"],
                    }
                    quant_comparisons.append(comparison)
                    md.append(
                        f"| {q['bs']} | {q['backend']} | {exact['backend']} | "
                        f"{q['latency_us']:.3f} | {exact['latency_us']:.3f} | "
                        f"{comparison['speedup_vs_exact']:.3f}x | "
                        f"{comparison['speedup_vs_best_exact']:.3f}x | "
                        f"{q['max_abs_error']:.6g} | {q['rmse']:.6g} | "
                        f"{q['relative_l2']:.6g} | "
                        f"{q['cosine_similarity']:.8f} | "
                        f"{'yes' if q['lossy_transport_active'] else 'fallback'} |"
                    )
        md.append("")

    if comm.rank == 0:
        md += [
            "## Dev-loop cost",
            "",
            f"- distributed initialization, correctness, and benchmark: "
            f"`{time.perf_counter() - args.started_at:.1f} s`",
            "",
        ]
        out = _ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{out}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if quant_comparisons:
            with open(f"{out}_quantized_comparison.csv", "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(quant_comparisons[0]))
                writer.writeheader()
                writer.writerows(quant_comparisons)
        Path(f"{out}.md").write_text("\n".join(md))
        Path(f"{out}_map.json").write_text(
            json.dumps(comm.export_auto_map(), indent=1))
        print(f"wrote {out}.md {out}.csv {out}_map.json\n"
              f"preload with: comm.import_auto_map("
              f"json.load(open('{args.out}_map.json')))", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report",
        action="store_true",
        help="one-schedule Kimi-K3 communication report preset",
    )
    ap.add_argument("--bench", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--check-only", action="store_true",
                    help="run correctness checks without benchmark tables")
    ap.add_argument("--skip-correctness", action="store_true",
                    help="skip backend correctness checks")
    ap.add_argument("--skip-machinery", action="store_true",
                    help="skip dispatcher/autotune integration checks")
    ap.add_argument("--ops", default="allreduce,allgather,"
                    "reducescatter,all-to-all,allreduce-norm,"
                    "gemm-allreduce,allreduce-norm-gemm,"
                    "allgather-col-quant,quantized-allreduce")
    ap.add_argument("--bs", default="1..4k",
                    help="comma list, k-suffix and pow2 ranges (1..8k)")
    ap.add_argument("--dims", default="7168", help="comma list")
    ap.add_argument("--gemm-k", type=int, default=896,
                    help="GEMM contract dim for gemm_allreduce")
    ap.add_argument("--gemm-n", type=int, default=6288,
                    help="GEMM out width for allreduce_norm_gemm")
    ap.add_argument("--col-quant-dim", type=int, default=3584,
                    help="gathered width for column all-gather + MXFP8")
    ap.add_argument("--enable-b10", action="store_true",
                    help="include the b10_copy_engine movers")
    ap.add_argument("--enable-cutedsl", action="store_true",
                    help="include cached experimental CuTeDSL fused kernels")
    ap.add_argument("--disable-flashinfer", action="store_true",
                    help="exclude FlashInfer backends")
    ap.add_argument("--disable-trt", action="store_true",
                    help="exclude TensorRT-LLM backends")
    ap.add_argument("--out", default="communication/local_result/lookup")
    args = ap.parse_args()
    args.started_at = time.perf_counter()

    if args.report:
        args.ops = (
            "allreduce,allgather,reducescatter,all-to-all,"
            "allreduce-norm,gemm-allreduce,allreduce-norm-gemm,"
            "allgather-col-quant,quantized-allreduce"
        )
        args.bs = (
            "1,2,4,8,16,32,64,128,256,512,1k,2k,4k,8k,16k"
        )
        args.dims = "7168"
        args.enable_b10 = True
        args.skip_machinery = True
        if args.out == "communication/local_result/lookup":
            args.out = "communication/local_result/communication_report"

    ops = parse_ops(args.ops)
    bs_list = parse_bs(args.bs)
    dims = [_parse_size(t) for t in args.dims.split(",") if t.strip()]

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                              os.environ.get("RANK", "0")))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                               os.environ.get("WORLD_SIZE", "1")))
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29545")
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", rank),
    )
    assert world >= 2, "need multi-rank"

    def log(*a) -> None:
        if rank == 0:
            print(*a, flush=True)

    max_bs, max_dim = max(bs_list), max(dims)
    # world-sized staging only when an AG/RS/A2A op is requested
    mult = world if any(op in _WORLD_SIZED for op in ops) else 1
    log("[init] building collective backends")
    comm = Collectives(
        dist.group.WORLD,
        max_numel=max_bs * max_dim * mult,
        dtype=torch.bfloat16,
        max_hidden=max_dim,
        flashinfer_max_tokens=max_bs,
        enable_flashinfer=not args.disable_flashinfer,
        enable_trt=not args.disable_trt,
        enable_b10=args.enable_b10,
        enable_cutedsl=args.enable_cutedsl,
        boundary_gemm_k=args.gemm_k,
        boundary_gemm_n=args.gemm_n,
        boundary_max_tokens=max_bs,
        col_ag_max_rows=(
            max_bs if "all_gather_col_quant" in ops else 0),
        col_ag_max_output_columns=(
            args.col_quant_dim if "all_gather_col_quant" in ops else None),
    )
    log("[init] collective backends ready")

    if not args.skip_correctness:
        run_correctness(
            comm, ops, bs_list, dims, args.col_quant_dim, log)
    if not args.check_only:
        run_bench(comm, args, ops, bs_list, dims, world, log)
    # last: the mover build changes candidate sets and autotune()
    # clears the map (bench results are already exported)
    if not args.skip_machinery:
        run_machinery(comm, rank, world, dims[0], bs_list, log)

    log("PASS communication/bench_comm.py")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
