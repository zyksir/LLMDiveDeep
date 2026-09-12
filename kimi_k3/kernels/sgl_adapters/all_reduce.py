"""Vendored sglang K3 fused AR ops (see package docstring).

The communicator these ops run on comes from ``comm.get_sgl_ar_state`` /
``comm.build_sgl_ar_state`` (which register it) — ``register_comm`` is
re-exported for callers that hold an externally built plane.

Both fused families are re-exported: 1shot multicast-push (``push_res``,
``push_norm``, ``finalize_all_reduce_push_norm``) works on any contiguous
bf16 tensor fitting a push slot; low-SM NVLS 2shot pull (``pull_res``,
``pull_norm``) requires the operand in multicast-bound symmetric memory
and its multicast VA (``input_mc_ptr``). The ``*_norm`` variants fuse an
RMSNorm epilogue over ``[.., NORM_DIM]`` rows (width hardcoded C++-side).
"""

from ..sgl_copied_kernels.ops.kimi_k3.all_reduce import (
    all_reduce_pull_norm,
    all_reduce_pull_res,
    all_reduce_push_norm,
    all_reduce_push_res,
    finalize_all_reduce_push_norm,
    register_comm,
)

# Row width of the *_norm variants' RMSNorm epilogue (the K3 latent
# width). MUST match kNormDim in jit/csrc/kimi_k3/comm/ar_fusion.cuh —
# operands of any other width cannot take the fused-norm path.
NORM_DIM = 3584

__all__ = [
    "NORM_DIM",
    "all_reduce_pull_norm",
    "all_reduce_pull_res",
    "all_reduce_push_norm",
    "all_reduce_push_res",
    "finalize_all_reduce_push_norm",
    "register_comm",
]
