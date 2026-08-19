"""Cached CuTe JIT bootstrap.

``cute.compile`` disables CuTe's caches.  A normal ``@cute.jit`` call uses
the disk cache but repeats Python-side IR hashing on every launch.  Bootstrap
once through the normal call, then retain its cached executable.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


_AOT_ABI = 1


class _LoadedCallable:
    def __init__(self, module: Any, name: str) -> None:
        self.module = module
        self.fn = getattr(module, name)

    def __call__(self, *args: Any) -> Any:
        return self.fn(*args)


def launch_and_bind(jitted: Callable, *args: Any):
    """Execute once through cached JIT and return its compiled executor."""
    jitted(*args)
    method = jitted.__call__.__func__
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    dsl = method._dsl_object
    keys = list(dsl.jit_cache._dict)
    if not keys:
        raise RuntimeError("CuTe JIT did not retain a cached executable")
    return dsl.jit_cache.get(keys[-1])


def load_or_compile(jitted: Callable, compile_args: tuple[Any, ...],
                    cache_key: str):
    """Load a rank/shape-specific AOT object, or compile and export it once."""
    from cutlass.cute import runtime

    arch = "".join(map(str, torch.cuda.get_device_capability()))
    name = f"cutedsl_v{_AOT_ABI}_sm{arch}_{cache_key}"
    root = Path(os.environ.get(
        "CUTE_DSL_AOT_DIR",
        Path(__file__).resolve().parents[2] / "local_debug" / "cutedsl_aot",
    ))
    obj = root / f"{name}.o"
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_hit = torch.tensor(
        [int(obj.is_file())], dtype=torch.int32, device="cuda")
    dist.all_reduce(local_hit, op=dist.ReduceOp.MIN)
    loaded = None
    if local_hit.item():
        try:
            loaded = _LoadedCallable(runtime.load_module(str(obj)), name)
        except Exception:
            obj.unlink(missing_ok=True)
    load_ok = torch.tensor(
        [int(loaded is not None)], dtype=torch.int32, device="cuda")
    dist.all_reduce(load_ok, op=dist.ReduceOp.MIN)
    if load_ok.item():
        return loaded
    obj.unlink(missing_ok=True)

    root.mkdir(parents=True, exist_ok=True)
    compiled = launch_and_bind(jitted, *compile_args)
    # Object export is CPU-heavy; eight concurrent LLVM object writers
    # contend badly enough to make no progress. Serialize rank artifacts.
    for exporter in range(world):
        if rank == exporter:
            compiled.export_to_c(
                file_path=str(root),
                file_name=name,
                function_prefix=name,
            )
        dist.barrier()
    return compiled
