"""Stand-in for ``sglang.srt.model_executor.runner_backend_utils.
tc_piecewise_cuda_graph.is_in_tc_piecewise_cuda_graph``.

Upstream this reads a module-level flag that is only ever set by sglang's
tc_piecewise CUDA-graph runner (``enable_tc_piecewise_cuda_graph``). That
runner never executes in this bench harness, so the flag is constantly
False — which is exactly what this stub returns.
"""

from __future__ import annotations


def is_in_tc_piecewise_cuda_graph() -> bool:
    return False


__all__ = ["is_in_tc_piecewise_cuda_graph"]
