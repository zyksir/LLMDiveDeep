#!/usr/bin/env python3
"""Standalone Kimi-K3 routing+permutation benchmark (open-source backends).

Creates synthetic inputs, calls each implementation's API, checks outputs
against the PyTorch oracle and the ABI invariants, and reports CUDA-graph
latency per stage and for the whole route+permute region across token counts,
with a speed-of-light column.

Run inside the TRT-LLM container with exactly one visible GPU:
  CUDA_VISIBLE_DEVICES=0 python3 kimi_k3_layer/kernels/bench_routing_permutation.py --quick
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kimi_k3_layer.kernels.routing_permutation import (
    LOCAL_EXPERT_START,
    NUM_EXPERTS,
    NUM_LOCAL_EXPERTS,
    TOP_K,
    check_fused_region,
    check_permutation,
    check_route,
    max_permuted_tokens,
    route_reference,
)
from kimi_k3_layer.kernels.routing_permutation_impls import (
    CANDIDATE,
    DEFAULT,
    FORK,
    SUPPLEMENTAL,
    load_impl,
)

ALL_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1024, 2048, 4096, 8192, 16384)
QUICK_BATCHES = (1, 8, 128, 4096)
CHECK_BATCHES = (1, 8, 128, 2048)
ATTAINABLE_BW = 6.8e12  # bytes/s; ~85% of B200 HBM peak (stated assumption)

# region compositions: these route impls x every loaded permute impl
# (fused impls are their own region row)
REGION_ROUTES = ("sgl_radix", "vllm_grouped_topk")


def graph_latency_us(fn, *, calls_per_graph: int, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / calls_per_graph)
    del graph
    return statistics.median(samples)


def make_logits(batch: int, case: str, generator: torch.Generator) -> torch.Tensor:
    logits = torch.randn(
        (batch, NUM_EXPERTS), dtype=torch.float32, device="cuda", generator=generator
    )
    if case == "tied":
        logits = (logits * 2).round() * 0.5
    elif case == "concentrated":
        boosted = list(range(LOCAL_EXPERT_START, LOCAL_EXPERT_START + 8))
        boosted += [0, 1, 2, 3, 892, 893, 894, 895]
        logits[:, boosted] += 8.0
    elif case == "no_local":
        logits[:, LOCAL_EXPERT_START : LOCAL_EXPERT_START + NUM_LOCAL_EXPERTS] -= 40.0
    return logits


def route_atol(name: str) -> float:
    return 2e-3 if name == "trt_routing_custom" else 1e-5


def check_impl_outputs(name, impl, prepared, logits, bias, ref_ids, ref_weights):
    """Fill inputs, run once eagerly, and verify every produced output."""
    if "logits" in prepared.inputs:
        prepared.inputs["logits"].copy_(logits)
        prepared.inputs["bias"].copy_(bias)
    if "ids" in prepared.inputs:
        prepared.inputs["ids"].copy_(ref_ids)
    if "weights" in prepared.inputs:
        prepared.inputs["weights"].copy_(ref_weights)
    prepared.run()
    torch.cuda.synchronize()

    if impl.KIND == "fused":
        return check_fused_region(
            name, logits, bias, prepared.outputs, atol=route_atol(name)
        )
    err = 0.0
    if impl.KIND == "route":
        err = check_route(
            name, prepared.outputs["ids"], prepared.outputs["weights"],
            ref_ids, ref_weights, atol=route_atol(name),
        )
    if impl.KIND == "permute":
        check_permutation(
            name, ref_ids, prepared.outputs, padding_filled=prepared.padding_filled
        )
    return err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default=None, help="comma-separated token counts")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--impls", default=None, help="comma-separated impl filter")
    parser.add_argument("--include-fork", action="store_true")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--calls-per-graph", type=int, default=0, help="0 = auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--json",
        type=Path,
        default=_REPO / "kimi_k3_layer/local_result/routing_permutation_bench.json",
    )
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("expose exactly one GPU (CUDA_VISIBLE_DEVICES=<n>)")

    if args.batches:
        batches = tuple(int(b) for b in args.batches.split(","))
    else:
        batches = QUICK_BATCHES if args.quick else ALL_BATCHES
    check_batches = tuple(b for b in CHECK_BATCHES if b in batches or b <= max(batches))

    names = list(DEFAULT)
    if args.include_fork:
        names += list(FORK)
    if args.include_candidates:
        names += list(CANDIDATE)
    if args.impls:
        selected = set(args.impls.split(","))
        known = {**DEFAULT, **SUPPLEMENTAL, **FORK, **CANDIDATE}
        unknown = selected - set(known)
        if unknown:
            raise SystemExit(f"unknown impls: {sorted(unknown)}")
        names = [n for n in known if n in selected]

    t0 = time.perf_counter()
    impls = {}
    for name in names:
        t = time.perf_counter()
        impl = load_impl(name)
        impl.load()
        impls[name] = impl
        print(f"loaded {name:28s} {time.perf_counter() - t:7.2f}s")
    print(f"setup total {time.perf_counter() - t0:.2f}s")

    device = torch.device("cuda")
    generator = torch.Generator(device="cuda").manual_seed(20260818)
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device=device, generator=generator) * 0.5

    results = {"env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
               "provenance": {n: impls[n].PROVENANCE for n in impls},
               "correctness": [], "latency_us": []}

    # ------------------------------ correctness ------------------------------
    if not args.no_check:
        t = time.perf_counter()
        cases = ("random", "tied", "concentrated", "no_local")
        for batch in check_batches:
            for case in cases:
                if case != "random" and batch > 128:
                    continue
                logits = make_logits(batch, case, generator)
                ref_ids, ref_weights = route_reference(logits, bias)
                for name, impl in impls.items():
                    try:
                        prepared = impl.prepare(batch, device)
                    except NotImplementedError as exc:
                        results["correctness"].append(
                            {"batch": batch, "case": case, "impl": name, "status": f"N/A: {exc}"}
                        )
                        continue
                    err = check_impl_outputs(
                        name, impl, prepared, logits, bias, ref_ids, ref_weights
                    )
                    results["correctness"].append(
                        {"batch": batch, "case": case, "impl": name,
                         "status": "pass", "weight_max_abs_err": err}
                    )
        n_pass = sum(1 for r in results["correctness"] if r["status"] == "pass")
        print(f"correctness: {n_pass}/{len(results['correctness'])} pass "
              f"({time.perf_counter() - t:.2f}s)")
        for r in results["correctness"]:
            if r["status"] != "pass":
                print(f"  {r['impl']} B={r['batch']} {r['case']}: {r['status']}")

    # ------------------------------- benchmark -------------------------------
    if not args.no_bench:
        t = time.perf_counter()
        floor_us = graph_latency_us(
            lambda: torch.cuda._sleep(1), calls_per_graph=200, warmup=10, repeats=5
        )
        print(f"no-op graph kernel floor: {floor_us:.3f} us")
        results["floor_us"] = floor_us

        # Practical-bound probes: measured mandatory work per stage. Loaded
        # from the prebuilt candidate extension; never compiles here.
        probe = None
        try:
            from kimi_k3_layer.kernels.routing_permutation_impls import (
                fused_llmdd_v1 as _probe_mod,
            )

            _probe_mod.load()
            probe = _probe_mod.module()
        except Exception as exc:  # noqa: BLE001 - probes are optional
            print(f"SOL probes unavailable ({exc}); strict bounds only")

        for batch in batches:
            calls = args.calls_per_graph or (200 if batch <= 1024 else 50)
            logits = make_logits(batch, "random", generator)
            ref_ids, ref_weights = route_reference(logits, bias)
            row = {"batch": batch, "stage": {}, "region": {}}

            for name, impl in impls.items():
                try:
                    prepared = impl.prepare(batch, device)
                except NotImplementedError:
                    row["stage" if impl.KIND != "fused" else "region"][name] = None
                    continue
                if "logits" in prepared.inputs:
                    prepared.inputs["logits"].copy_(logits)
                    prepared.inputs["bias"].copy_(bias)
                if "ids" in prepared.inputs:
                    prepared.inputs["ids"].copy_(ref_ids)
                if "weights" in prepared.inputs:
                    prepared.inputs["weights"].copy_(ref_weights)
                latency = graph_latency_us(
                    prepared.run, calls_per_graph=calls,
                    warmup=args.warmup, repeats=args.repeats,
                )
                key = "region" if impl.KIND == "fused" else "stage"
                row[key][name] = latency

            permute_names = [n for n in impls if impls[n].KIND == "permute"]
            for route_name in REGION_ROUTES:
                if route_name not in impls:
                    continue
                for perm_name in permute_names:
                    route_prep = impls[route_name].prepare(batch, device)
                    try:
                        perm_prep = impls[perm_name].prepare(
                            batch, device,
                            ids=route_prep.outputs["ids"],
                            weights=route_prep.outputs["weights"],
                        )
                    except NotImplementedError:
                        row["region"][f"{route_name}+{perm_name}"] = None
                        continue
                    route_prep.inputs["logits"].copy_(logits)
                    route_prep.inputs["bias"].copy_(bias)

                    def region_run(r=route_prep, p=perm_prep):
                        r.run()
                        p.run()

                    row["region"][f"{route_name}+{perm_name}"] = graph_latency_us(
                        region_run, calls_per_graph=calls,
                        warmup=args.warmup, repeats=args.repeats,
                    )

            # strict lower bound: launch floor vs bytes/attainable BW
            route_bytes = batch * NUM_EXPERTS * 4 + batch * TOP_K * 8
            perm_bytes = batch * TOP_K * (8 + 4 + 8) + 64
            row["sol_us"] = {
                "route": max(floor_us, route_bytes / ATTAINABLE_BW * 1e6),
                "permute": max(floor_us, perm_bytes / ATTAINABLE_BW * 1e6),
                "region_unfused": max(2 * floor_us, (route_bytes + perm_bytes) / ATTAINABLE_BW * 1e6),
                "region_fused": max(floor_us, (route_bytes + perm_bytes) / ATTAINABLE_BW * 1e6),
            }
            # practical bound: measured mandatory-work probes for this shape
            if probe is not None:
                probe_out_f = torch.empty(batch, dtype=torch.float32, device=device)
                probe_out_i = torch.empty(1, dtype=torch.int32, device=device)
                route_pb = graph_latency_us(
                    lambda: probe.probe_route_work(logits, bias, probe_out_f),
                    calls_per_graph=calls, warmup=args.warmup, repeats=args.repeats,
                )
                if batch <= 512:
                    perm_pb = graph_latency_us(
                        lambda: probe.probe_permute_work(ref_ids, probe_out_i),
                        calls_per_graph=calls, warmup=args.warmup, repeats=args.repeats,
                    )
                else:
                    # single-CTA probe layout is invalid here (multi-CTA
                    # implementations beat it); measured bound = moving the
                    # permutation's mandatory bytes (ids read + map writes)
                    # with a real copy kernel (clone reads+writes equally).
                    mand_bytes = batch * TOP_K * 8 + 2 * max_permuted_tokens(batch) * 4
                    clone_src = torch.empty(
                        max(1, mand_bytes // 8), dtype=torch.int32, device=device
                    )
                    perm_pb = graph_latency_us(
                        lambda: clone_src.clone(),
                        calls_per_graph=calls, warmup=args.warmup, repeats=args.repeats,
                    )
                row["sol_us"]["route_practical"] = route_pb
                row["sol_us"]["permute_practical"] = perm_pb
                row["sol_us"]["region_practical"] = route_pb + perm_pb
                row["sol_us"]["region_practical_pdl"] = route_pb + perm_pb - floor_us
            results["latency_us"].append(row)
            print(f"B={batch} done")
        print(f"benchmark wall time: {time.perf_counter() - t:.2f}s")

        # ------------------------------- tables -------------------------------
        # One table per operation so each row is compared only against the SOL
        # bound of that same operation.
        region_names = sorted(
            {k for row in results["latency_us"] for k in row["region"]}
        )

        def fmt(v, width):
            return f"{v:{width}.3f}" if isinstance(v, float) else f"{'N/A':>{width}}"

        for kind, sol_key in (("route", "route"), ("permute", "permute")):
            names = [n for n in impls if impls[n].KIND == kind]
            if not names:
                continue
            print(f"\n== {kind} stage latency (us) ==")
            print(f"{'B':>6} " + " ".join(f"{n[:18]:>18}" for n in names)
                  + f" {'practical bound':>16} {'strict floor':>13}")
            for row in results["latency_us"]:
                cells = " ".join(fmt(row["stage"].get(n), 18) for n in names)
                pb = row["sol_us"].get(f"{sol_key}_practical")
                print(f"{row['batch']:>6} {cells} {fmt(pb, 16)} "
                      f"{row['sol_us'][sol_key]:13.3f}")

        print("\n== whole-region latency (us): route + permute ==")
        print(f"{'B':>6} " + " ".join(f"{n[:30]:>30}" for n in region_names)
              + f" {'practical(2k PDL)':>18} {'strict floor':>13}")
        for row in results["latency_us"]:
            cells = " ".join(fmt(row["region"].get(n), 30) for n in region_names)
            pb = row["sol_us"].get("region_practical_pdl")
            print(f"{row['batch']:>6} {cells} {fmt(pb, 18)} "
                  f"{row['sol_us']['region_unfused']:13.3f}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
