"""Cross-rank communication verification for the 4-block demo.

Two independent mechanisms, reported side by side in the receipt:

1. Python call-site counters (attribution): ``AllReduce.forward`` is wrapped
   at class level — this covers the attention o_proj reductions (row-parallel
   ``Linear`` holds an ``AllReduce`` instance) and the shared-expert
   allreduce — with per-instance labels resolved by scanning the stack's
   modules. The MoE ``comm.dispatch`` / ``comm.combine`` (NVLinkOneSided A2A)
   and the bench-side routed-output ``allgather`` are wrapped per instance.

2. torch.profiler kernel scan (ground truth): one forward is profiled and
   every CUDA kernel whose name matches a communication pattern (nccl*,
   *allreduce*, *allgather*, *alltoall*/*a2a*, *one_sided*, *mnnvl*,
   *lamport*, *oneshot*, *twoshot*, *reducescatter*) is counted by name. This
   catches any collective the Python wrappers might miss.

The CP-mode verdict requires BOTH: python counters show only MoE A2A
dispatch/combine (zero allreduce, zero allgather), and the profiler scan
shows no reduction/gather collectives beyond the A2A kernels.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch

COMM_KERNEL_PATTERNS = (
    "nccl",
    "allreduce",
    "all_reduce",
    "allgather",
    "all_gather",
    "alltoall",
    "all_to_all",
    "a2a",
    "one_sided",
    "onesided",
    "mnnvl",
    "lamport",
    "oneshot",
    "twoshot",
    "reducescatter",
    "reduce_scatter",
    "multimem",
)

A2A_ONLY_PATTERNS = ("a2a", "alltoall", "all_to_all", "one_sided", "onesided")


def _allreduce_labels(stack: torch.nn.Module) -> dict[int, str]:
    """Map id(AllReduce instance) -> human label, by walking the stack."""
    from tensorrt_llm._torch.distributed import AllReduce

    labels: dict[int, str] = {}
    for name, module in stack.named_modules():
        if isinstance(module, AllReduce):
            labels[id(module)] = f"allreduce[{name}]"
    for idx, block in enumerate(stack.blocks):
        shared_ar = getattr(block.moe.shared_experts, "allreduce", None)
        if shared_ar is not None:
            labels[id(shared_ar)] = f"allreduce[block{idx}.moe.shared_experts]"
    return labels


@contextmanager
def count_comm_calls(stack: torch.nn.Module) -> Iterator[dict[str, int]]:
    """Count every wrapped comm call during the ``with`` body."""
    from tensorrt_llm._torch.distributed import AllReduce

    counts: dict[str, int] = {}
    labels = _allreduce_labels(stack)

    def bump(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    original_ar_forward = AllReduce.forward

    def wrapped_ar_forward(self, *args, **kwargs):
        bump(labels.get(id(self), "allreduce[unattributed]"))
        return original_ar_forward(self, *args, **kwargs)

    AllReduce.forward = wrapped_ar_forward

    # All four blocks share one routed MoE module; wrap its comm instance once.
    moe = stack.blocks[0].moe.routed_moe
    original_dispatch = moe.comm.dispatch
    original_combine = moe.comm.combine
    moe.comm.dispatch = lambda *a, **k: (bump("moe_a2a_dispatch"), original_dispatch(*a, **k))[1]
    moe.comm.combine = lambda *a, **k: (bump("moe_a2a_combine"), original_combine(*a, **k))[1]

    original_gather = stack.gather_routed

    def wrapped_gather(*args, **kwargs):
        bump("allgather[routed_output]")
        return original_gather(*args, **kwargs)

    stack.gather_routed = wrapped_gather
    try:
        yield counts
    finally:
        AllReduce.forward = original_ar_forward
        moe.comm.dispatch = original_dispatch
        moe.comm.combine = original_combine
        if "gather_routed" in stack.__dict__:
            del stack.__dict__["gather_routed"]


def profile_comm_kernels(run: Callable[[], Any]) -> dict[str, int]:
    """One profiled forward; CUDA kernel name -> launch count (comm only)."""
    with torch.inference_mode():
        run()  # warm any lazy paths so the profiled iteration is steady
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            run()
            torch.cuda.synchronize()
    kernel_counts: dict[str, int] = {}
    for event in prof.key_averages():
        if event.device_type != torch.profiler.DeviceType.CUDA:
            continue
        name_lower = event.key.lower()
        if any(pattern in name_lower for pattern in COMM_KERNEL_PATTERNS):
            kernel_counts[event.key] = kernel_counts.get(event.key, 0) + event.count
    return kernel_counts


def classify_kernels(kernel_counts: dict[str, int]) -> dict[str, list[str]]:
    """Split matched comm kernels into MoE-A2A vs everything else."""
    a2a, other = [], []
    for name in sorted(kernel_counts):
        (a2a if any(p in name.lower() for p in A2A_ONLY_PATTERNS) else other).append(name)
    return {"a2a_kernels": a2a, "non_a2a_comm_kernels": other}


def cp_verdict(counts: dict[str, int], kernel_counts: dict[str, int]) -> dict[str, Any]:
    """CP mode passes iff the only cross-rank comm is the MoE A2A."""
    non_a2a_calls = {
        k: v for k, v in counts.items() if not k.startswith("moe_a2a_")
    }
    non_a2a_kernels = classify_kernels(kernel_counts)["non_a2a_comm_kernels"]
    passed = not non_a2a_calls and not non_a2a_kernels
    return {
        "passed": passed,
        "unexpected_python_call_sites": non_a2a_calls,
        "unexpected_comm_kernels": {k: kernel_counts[k] for k in non_a2a_kernels},
    }
