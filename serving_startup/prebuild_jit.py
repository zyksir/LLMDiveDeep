#!/usr/bin/env python3
"""Prebuild every JIT-compiled kernel this stack needs, at image-build time.

Removes the compile tax from serving startup (~44 min serialized on a cold
node, measured on 8xB200 / 1.3.0rc23). Three things make that possible:

* **No GPU required.** Pinning `FLASHINFER_CUDA_ARCH_LIST` /
  `TORCH_CUDA_ARCH_LIST` stops every build path from querying a device, so
  this runs in an ordinary build container. The artifacts are shape-generic.
* **Parallel across modules.** Each module is an independent ninja
  invocation; upstream triggers them one after another on first use.
* **Parallel inside a module.** `FLASHINFER_NVCC_THREADS` defaults to 1 and
  nothing passes `--split-compile`, so a single `ptxas` grinds for ~10 min on
  one core while the rest of the machine idles. Measured on `moe_utils`:
  586 s -> 129 s.

Usage (defaults target sm100a / B200):

    python3 serving_startup/prebuild_jit.py                  # into ~/.cache
    python3 serving_startup/prebuild_jit.py --base /tmp/jit  # isolated tree
    python3 serving_startup/prebuild_jit.py --serial         # A/B reference

Bake into an image by running it in a build stage, then keeping the resulting
cache tree (`$BASE/.cache/flashinfer`, `$TORCH_EXTENSIONS_DIR`) in the layer.
A second run is a no-op, which is the check that caching actually works.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Each entry runs in its own process: flashinfer's generators are not
# thread-safe (shared cubin dir, per-module file locks) and a crash in one
# module must not take the others down.
# {arch} is filled per-target-arch by targets(): flashinfer ships per-sm
# generators for the CUTLASS MoE (sm90/sm100/sm103/sm120), and naming one
# literally bakes the wrong cubins into every other part's image (a GB300
# image built sm_100 CUTLASS MoE until this was arch-selected). The
# trtllm-gen fused-moe generator is only *named* sm100; it serves sm103 too.
FLASHINFER_TARGETS = {
    "moe_utils": "from flashinfer.jit.moe_utils import gen_moe_utils_module as g",
    "trtllm_gen_fused_moe":
        "from flashinfer.fused_moe import gen_trtllm_gen_fused_moe_sm100_module as g",
    "cutlass_fused_moe":
        "from flashinfer.fused_moe import gen_cutlass_fused_moe_sm{arch}_module as g",
    "trtllm_comm":
        "from flashinfer.comm import gen_trtllm_comm_module as g",
    "mxfp8_quantization":
        "from flashinfer.jit.fp8_quantization import gen_mxfp8_quantization_sm100_module as g",
}

# Per-module split-compile opt-out. Empty by default: measured cold and solo,
# `cutlass_fused_moe_sm100` builds in 508 s with --split-compile=16 and 544 s
# without, so the flag is not harmful anywhere we have data for. Kept as a hook
# because the win is very uneven across modules (4.3x on moe_utils, ~1.07x
# here), so a future module may want out.
NO_SPLIT_COMPILE: set[str] = set()

# Longest first. A plain thread pool interleaves long and short modules and can
# leave the longest running alone at the end; starting the long ones first lets
# the short ones fill the gaps. `cutlass_fused_moe_sm100` is the floor for the
# whole prebuild at ~510 s cold and solo -- a deployment that knows it will
# never use the CUTLASS MoE backend can drop it with --only and save ~40% of
# the wall clock.
BUILD_ORDER = (
    "cutlass_fused_moe",
    "kimi_routing_permutation",
    "trtllm_gen_fused_moe",
    "trtllm_comm",
    "moe_utils",
    "mxfp8_quantization",
    "kimi_radix",
    "k3_comm_cuda",
    "low_contention_fused_copy",
    "b10_multimem_ar",
)

FLASHINFER_CHILD = """
import time
{import_line}
t = time.time()
g().build_and_load()
print(f"SECONDS {{time.time() - t:.1f}}")
"""

# kimi_k3_layer's own FlashInfer-JIT modules (TRT-LLM Gen routing/permutation).
ROUTING_CHILD = """
import sys, time
sys.path.insert(0, {repo!r})
from kimi_k3_layer.kernels.routing_permutation_impls import load_impl
t = time.time()
for name in ("trt_moe_sort", "trt_routing_custom"):
    load_impl(name).load()
print(f"SECONDS {{time.time() - t:.1f}}")
"""

# torch.utils.cpp_extension modules. `col_quant_cuda.get_module()` reads
# torch.cuda.get_device_capability() eagerly to populate TORCH_CUDA_ARCH_LIST,
# which raises without a device -- stub it to the pinned arch for the build
# only. (A two-line upstream fix would be to honour an already-set
# TORCH_CUDA_ARCH_LIST instead; keeping the workaround here avoids touching
# shipped code for a build-time concern.)
INLINE_CHILD = """
import sys, time
sys.path.insert(0, {repo!r})
import torch
major, minor = {cap!r}
torch.cuda.get_device_capability = lambda *a, **k: (major, minor)
t = time.time()
mod = __import__({module!r}, fromlist=[{attr!r}])
getattr(mod, {attr!r})()
print(f"SECONDS {{time.time() - t:.1f}}")
"""

# The vendored SGLang radix router (tvm_ffi build; ~4 min cold). Arch comes
# from common.arch, which honours B10_FORCE_SM, so this builds GPU-less.
RADIX_CHILD = """
import os, sys, time
sys.path.insert(0, {repo!r})
os.environ["B10_FORCE_SM"] = {force_sm!r}
t = time.time()
from kimi_k3_layer.kernels.routing_radix import warmup
warmup()
print(f"SECONDS {{time.time() - t:.1f}}")
"""

INLINE_TARGETS = {
    "k3_comm_cuda": ("communication.kernels.col_quant_cuda", "get_module"),
    "b10_multimem_ar": ("communication.kernels.b10_multimem", "_jit"),
    "low_contention_fused_copy": ("communication.kernels.b10_copy_engine",
                                  "_fused_ext"),
}


def build_env(args, name: str) -> dict:
    """Env for one build process.

    Concurrency budget: `jobs` ninja workers each allowed `split` ptxas
    threads, times the number of modules built at once. Defaults keep the
    nominal product near the core count rather than oversubscribing it.
    """
    base = Path(args.base).resolve()
    env = dict(
        os.environ,
        FLASHINFER_CUDA_ARCH_LIST=args.arch,
        TORCH_CUDA_ARCH_LIST=f"{args.arch}a",
        FLASHINFER_WORKSPACE_BASE=str(base),
        TORCH_EXTENSIONS_DIR=str(base / ".cache" / "torch_extensions"),
        MAX_JOBS=str(args.jobs),
    )
    if args.cubin_dir:
        env["FLASHINFER_CUBIN_DIR"] = args.cubin_dir
    split_ok = not args.no_split_compile and name not in NO_SPLIT_COMPILE
    if split_ok:
        env["FLASHINFER_NVCC_THREADS"] = str(args.nvcc_threads)
        env["FLASHINFER_EXTRA_CUDAFLAGS"] = f"--split-compile={args.split}"
    return env


def run_one(args, name: str, source: str) -> dict:
    started = time.time()
    proc = subprocess.run([sys.executable, "-c", source],
                          env=build_env(args, name),
                          capture_output=True, text=True)
    wall = time.time() - started
    reported = next((float(line.split()[1])
                     for line in proc.stdout.splitlines()
                     if line.startswith("SECONDS")), None)
    ok = proc.returncode == 0 and reported is not None
    print(f"[{'ok ' if ok else 'FAIL'}] {name:<32} {wall:7.1f}s", flush=True)
    if not ok:
        sys.stderr.write(f"--- {name} stdout ---\n{proc.stdout[-2000:]}\n"
                         f"--- {name} stderr ---\n{proc.stderr[-2000:]}\n")
    return {"name": name, "wall_s": round(wall, 1), "build_s": reported,
            "ok": ok}


def replace_arch(args, target_arch: str):
    """Copy of `args` pinned to one arch, with an arch-specific cache base."""
    import argparse as _argparse

    clone = _argparse.Namespace(**vars(args))
    clone.arch = target_arch
    if len([v for v in args.arch.split(",") if v.strip()]) > 1:
        clone.base = str(Path(args.base) / f"sm{target_arch.replace('.', '')}a")
    return clone


def targets(args) -> list[tuple[str, str]]:
    cap = tuple(int(part) for part in args.arch.split("."))
    arch_num = f"{cap[0]}{cap[1]}"  # '100' / '103' for the per-sm generators
    items = [(name, FLASHINFER_CHILD.format(
                  import_line=line.format(arch=arch_num)))
             for name, line in FLASHINFER_TARGETS.items()]
    items.append(("kimi_routing_permutation",
                  ROUTING_CHILD.format(repo=str(REPO))))
    items.append(("kimi_radix",
                  RADIX_CHILD.format(repo=str(REPO),
                                     force_sm=f"{cap[0]}.{cap[1]}")))
    items += [(name, INLINE_CHILD.format(repo=str(REPO), cap=cap,
                                         module=module, attr=attr))
              for name, (module, attr) in INLINE_TARGETS.items()]
    if args.only:
        wanted = set(args.only.split(","))
        items = [item for item in items if item[0] in wanted]
        missing = wanted - {name for name, _ in items}
        if missing:
            raise SystemExit(f"unknown target(s): {sorted(missing)}")
    rank = {name: index for index, name in enumerate(BUILD_ORDER)}
    items.sort(key=lambda item: rank.get(item[0], len(rank)))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(Path.home()),
                        help="cache root; flashinfer uses $BASE/.cache/flashinfer")
    parser.add_argument(
        "--arch", default="10.0",
        help="target arch(es), comma-separated: 10.0 = B200/GB200, "
             "10.3 = B300/GB300. Listing both bakes one image that serves "
             "both fleets (separate cache trees, so no cubin is reused across "
             "arches).")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="modules built at once (1 = serial)")
    parser.add_argument("--jobs", type=int, default=16, help="MAX_JOBS per module")
    parser.add_argument("--split", type=int, default=16,
                        help="nvcc --split-compile threads (0 = all cores)")
    parser.add_argument("--nvcc-threads", type=int, default=8,
                        help="FLASHINFER_NVCC_THREADS (upstream default is 1)")
    parser.add_argument("--no-split-compile", action="store_true",
                        help="upstream flags, for A/B reference")
    parser.add_argument("--serial", action="store_true",
                        help="shorthand for --concurrency 1")
    parser.add_argument("--cubin-dir", default="",
                        help="share an already-downloaded cubin dir")
    parser.add_argument("--only", default="", help="comma-separated target subset")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.serial:
        args.concurrency = 1

    arches = [value.strip() for value in args.arch.split(",") if value.strip()]
    started = time.time()
    results: list[dict] = []
    for target_arch in arches:
        # Each arch gets its own cache subtree. The flashinfer workspace is
        # already arch-keyed ($BASE/.cache/flashinfer/<ver>/<arch>/), but the
        # torch extension dir is not, so keep them apart explicitly.
        per_arch = replace_arch(args, target_arch)
        items = targets(per_arch)
        print(f"prebuilding {len(items)} modules  "
              f"arch=sm{target_arch.replace('.', '')}a  "
              f"concurrency={args.concurrency}  jobs={args.jobs}  "
              f"split_compile={'off' if args.no_split_compile else args.split}  "
              f"base={per_arch.base}", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for result in pool.map(lambda item: run_one(per_arch, *item), items):
                results.append({**result, "arch": target_arch})
    wall = time.time() - started

    serialized = sum(r["wall_s"] for r in results)
    failed = [f"{r['arch']}/{r['name']}" for r in results if not r["ok"]]
    print(f"\nwall {wall:.1f}s   sum-of-parts {serialized:.1f}s   "
          f"overlap saved {serialized - wall:.1f}s   arches={','.join(arches)}")
    if failed:
        print(f"FAILED: {failed}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"wall_s": round(wall, 1), "serialized_s": round(serialized, 1),
             "arches": arches, "concurrency": args.concurrency,
             "split": args.split,
             "no_split_compile": args.no_split_compile,
             "results": results}, indent=2) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
