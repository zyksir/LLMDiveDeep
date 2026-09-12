"""``sgl`` backend: sglang's Kimi-K3 fused MNNVL collectives (vendored).

Wraps the sglang comm stack vendored under
``kimi_k3/kernels/sgl_copied_kernels/`` THROUGH its sanctioned adapter
surface ``kimi_k3.kernels.sgl_adapters`` (nothing here touches the
vendored tree directly). Kernels: ``jit/csrc/kimi_k3/comm/ar_fusion.cuh``
(AR family), ``gemm_ar.cuh`` and ``gemm_ag.cuh`` (GEMM fusions); python
surface ``ops/kimi_k3/{all_reduce,gemm_ar,gemm_ag}.py``. Dispatch logic
mirrors sglang's ``k3_ar_fusion.py``.

Impls (all bf16-only; numel must be a multiple of 8):

    sgl:push_res    all_reduce — 1shot multicast push through the
                    CustomAllReduceV2 push plane. The vendored kernel
                    reduces IN PLACE, so the operand is staged into this
                    backend's plain-CUDA scratch first (unless already
                    there); capacity = the push plane's slot bytes
                    (tuned default 768 KB on SM100 TP8).
    sgl:pull_res    all_reduce — low-SM NVLS 2shot (multimem ld_reduce +
                    st) ON the operand, which must live in
                    multicast-bound symmetric memory: the operand is
                    staged into the shared torch_symm staging buffer,
                    whose rendezvous handle carries the multicast VA the
                    kernel needs (sglang's ``symm_buffer``/``get_mc_ptr``
                    contract). Producers can skip the stage-in copy via
                    ``Collectives.symm_input(..., impl="sgl:pull_res")``.
                    Capacity = ``max_numel``; needs the torch_symm
                    backend AND a nonzero multicast binding.
    sgl:push_norm   allreduce_norm. Fully fused (AR + RMSNorm in one
    sgl:pull_norm   kernel) only for residual-free operands at the K3
                    latent width H=3584 (``kNormDim`` is hardcoded in
                    ar_fusion.cuh). Any other call composes the fused
                    AR+residual-fold (``push_res``/``pull_res`` — the
                    residual add rides the reduce) with a separate
                    rmsnorm kernel. NOTE: autotune times the composed
                    residual form; the fully fused kernel only serves
                    residual-free H=3584 calls.
    sgl:gemm_ar     gemm_allreduce — one kernel computes the local
                    ``x @ w.T`` partial AND the cross-rank sum through a
                    peer-mapped P2P region. N (= w.shape[0]) is fixed to
                    7168 (K3 hidden), M in [1, 512]; any K (per-K JIT
                    module, one shared K-independent comm region). All
                    ranks must call with the same M in lockstep.

NOT expressible as Collectives ops — separate documented entry points on
this backend (reachable via ``Collectives._sgl_state()`` or directly via
``kimi_k3.kernels.sgl_adapters``):

    finalize_push_norm  deferred MoE finalize + push AR + RMSNorm
                        (consumes the trtllm-gen ``do_finalize=False``
                        triple; top_k fixed to 16). See
                        ``communication/MOE_FINALIZE_AR.md`` for why
                        this fusion shape wins.
    gemm_ag_up_proj     column-parallel up_proj GEMV + multicast AG +
                        add3. Takes the FULL replicated [7168, 3584]
                        weight plus bias operands, M <= 12, TP8-shaped —
                        no Collectives op has that signature.

Capability gate: SM100/103, NVLink multicast (NVLS), world size in the
vendored tuned table, whole-world group, bf16 — probed cheaply by
:func:`available` (never raises, no collectives; False on a CPU box or
non-multicast fabric). Construction is COLLECTIVE (CustomAllReduceV2
symm-slab rendezvous + JIT build via ``get_sgl_ar_state()``, shared
process-wide with the K3 blocks) and must run OUTSIDE CUDA-graph
capture; the kernels themselves are capture-safe (push phase counters,
pull semaphores and the gemm_ar epoch tickets are device-resident and
advance on replay — sglang serves them under CUDA graphs). Push and
pull share the CustomAllReduceV2 planes with any other user in the
process (the K3 blocks): calls must stay single-stream rank-lockstep.

sglang's own dispatch uses push below ~512 KB and pull above; that
heuristic is deliberately NOT baked in here — both impls are autotune
candidates and ``Collectives``' pick() decides per (op, shape).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ..context import Ctx
from . import flashinfer_backend


def available() -> bool:
    """Cheap, non-collective probe (no JIT, no allocation): the vendored
    sglang fused-AR preconditions hold here (CUDA + SM100/103 + dist
    initialized + supported world size + NVLink multicast). False —
    never a raise — on CPU-only boxes or when the kimi_k3 adapters are
    not importable."""
    try:
        from kimi_k3.kernels.sgl_adapters import comm as sgl_comm
    except ImportError:
        return False
    return sgl_comm.sgl_ar_available()


class SglBackend:
    """Owns the push scratch and the pull multicast view of the shared
    torch_symm staging; the CustomAllReduceV2 planes are process-shared
    state from ``sgl_adapters.comm.get_sgl_ar_state()`` (NOT torn down
    here — the K3 blocks may hold the same state)."""

    def __init__(self, ctx: Ctx, symm=None) -> None:
        from kimi_k3.kernels.sgl_adapters import comm as sgl_comm
        from kimi_k3.kernels.sgl_adapters import comm_gemm

        if ctx.dtype != torch.bfloat16:
            raise RuntimeError("sgl backend is bf16-only")
        state = sgl_comm.get_sgl_ar_state()  # COLLECTIVE on first call
        if state is None:
            raise RuntimeError(
                "sgl fused AR unavailable (arch/world-size/multicast)")
        if state.world_size != ctx.world:
            raise RuntimeError(
                f"sgl comm plane spans world_size={state.world_size}, "
                f"Collectives group is {ctx.world} — the vendored ops "
                "key on world size and need the whole-world group")
        self.ctx = ctx
        self.state = state
        self.max_push_bytes = int(state.max_push_size)
        # The push kernel reduces its operand IN PLACE ("any contiguous
        # bf16 tensor"); staging into this plain-CUDA scratch keeps the
        # caller's operand intact — results are VIEWS of it, valid until
        # the next push call (torch_symm aliasing contract).
        self._push_buf = torch.empty(
            self.max_push_bytes // 2, dtype=ctx.dtype, device=ctx.device)
        # Pull staging: the kernel needs its operand's multicast VA, so
        # the operand must live in a multicast-bound symm buffer. The
        # torch_symm staging buffer IS one — its (cached) rendezvous
        # handle carries multicast_ptr; offsets map local VA -> mc VA
        # exactly like sglang's k3_ar_fusion.find_mc_ptr.
        self._symm = symm
        self._mc_base = 0
        if symm is not None and state.max_pull_size > 0:
            import torch.distributed._symmetric_memory as symm_mem_mod
            handle = symm_mem_mod.rendezvous(symm._symm_in, symm.gname)
            self._mc_base = int(getattr(handle, "multicast_ptr", 0) or 0)
        # gemm_ar geometry for shape-only supports(); the P2P region is
        # built lazily on the first gemm_allreduce call (collective).
        self._gemm_ar_n = comm_gemm.GEMM_AR_N
        self._gemm_ar_max_tokens = comm_gemm.GEMM_AR_MAX_TOKENS

    # ---------------------------------------------------------- staging

    @property
    def has_pull(self) -> bool:
        return self._mc_base != 0

    def _mc_ptr_of(self, v: torch.Tensor) -> int:
        """Multicast VA of a torch_symm-staging-resident view."""
        return self._mc_base + (
            v.data_ptr() - self._symm._symm_in.data_ptr())

    def _push_staged(self, x: torch.Tensor) -> torch.Tensor:
        base = self._push_buf.data_ptr()
        span = self._push_buf.numel() * self._push_buf.element_size()
        if x.is_contiguous() and base <= x.data_ptr() < base + span:
            return x  # already scratch-resident
        v = self._push_buf[: x.numel()].view_as(x)
        v.copy_(x)
        return v

    # --------------------------------------------------------- capacity

    def supports(self, op: str, x: torch.Tensor, variant: str) -> bool:
        """Shape-only (meta-safe, no collective)."""
        n = x.numel()
        if op in ("all_reduce", "allreduce_norm"):
            if n <= 0 or n % 8:  # kernels move 16B bf16x8 vectors
                return False
            if variant.startswith("push"):
                return n * x.element_size() <= self.max_push_bytes
            if variant.startswith("pull"):
                return self.has_pull and n <= self.ctx.max_numel
            return False
        if op == "gemm_allreduce":  # x is the [B, N] output probe
            return (x.shape[1] == self._gemm_ar_n
                    and 0 < x.shape[0] <= self._gemm_ar_max_tokens)
        return False

    # -------------------------------------------------------------- ops

    def all_reduce(self, x: torch.Tensor, variant: str,
                   residual: torch.Tensor | None = None) -> torch.Tensor:
        """``allreduce(x) [+ residual]``; the residual must be identical
        on every rank (a fully reduced stream). Returns a VIEW of the
        push scratch / symm staging, valid until the next call."""
        from kimi_k3.kernels.sgl_adapters import all_reduce as sgl_ar
        if variant == "push_res":
            return sgl_ar.all_reduce_push_res(
                self.state.world_size, self._push_staged(x), residual)
        if variant == "pull_res":
            v = self._symm.staged(x)
            return sgl_ar.all_reduce_pull_res(
                self.state.world_size, v, residual,
                input_mc_ptr=self._mc_ptr_of(v))
        raise ValueError(f"unknown sgl all_reduce variant {variant!r}")

    def allreduce_norm(self, x, gamma, eps, residual, variant):
        """Returns (norm, reduced). Fully fused kernel only for
        residual-free operands at H=3584 (kNormDim, hardcoded in
        ar_fusion.cuh) — the fused epilogue overwrites the reduce with
        the norm, so ``reduced`` is None there. Every other call runs
        the fused AR+residual-fold plus a separate rmsnorm."""
        push = variant == "push_norm"
        if variant not in ("push_norm", "pull_norm"):
            raise ValueError(
                f"unknown sgl allreduce_norm variant {variant!r}")
        rows, hidden = x.shape
        from kimi_k3.kernels.sgl_adapters import all_reduce as sgl_ar
        if residual is None and hidden == sgl_ar.NORM_DIM:
            if push:
                norm = sgl_ar.all_reduce_push_norm(
                    self.state.world_size, self._push_staged(x), gamma,
                    eps, num_norm_rows=rows)
            else:
                v = self._symm.staged(x)
                norm = sgl_ar.all_reduce_pull_norm(
                    self.state.world_size, v, gamma, eps,
                    num_norm_rows=rows, input_mc_ptr=self._mc_ptr_of(v))
            return norm, None
        red = self.all_reduce(
            x, "push_res" if push else "pull_res", residual)
        return flashinfer_backend.rmsnorm(red, gamma, eps), red

    def gemm_allreduce(self, x: torch.Tensor,
                       w: torch.Tensor) -> torch.Tensor:
        """``allreduce(x @ w.T)`` in one kernel; w must be [7168, K],
        M <= 512. First call per process is COLLECTIVE (P2P region
        rendezvous + per-K JIT build) — run it eagerly before any
        CUDA-graph capture."""
        from kimi_k3.kernels.sgl_adapters import comm_gemm
        comm_gemm.ensure_gemm_ar(x.shape[1])
        return comm_gemm.o_proj_gemm_ar(x, w)

    # ---------------------- fused entry points outside the op grammar

    def finalize_push_norm(self, out, gemm2_out,
                           expanded_idx_to_permuted_idx, expert_weights,
                           gamma, eps: float = 1e-6) -> torch.Tensor:
        """Deferred MoE finalize + 1shot push AR + RMSNorm on every row;
        ``out`` ([T, 3584] bf16) is output-only and must fit a push
        slot. See the module docstring — no Collectives op expresses
        the trtllm-gen deferred-finalize operand triple."""
        from kimi_k3.kernels.sgl_adapters import all_reduce as sgl_ar
        return sgl_ar.finalize_all_reduce_push_norm(
            self.state.world_size, out, gemm2_out,
            expanded_idx_to_permuted_idx, expert_weights, gamma, eps)

    def gemm_ag_up_proj(self, x, weight, b, c=None) -> torch.Tensor:
        """``up_proj(x) + b (+ c)`` via column GEMV + multicast AG +
        add3; ``weight`` is the FULL replicated [7168, 3584] up_proj,
        M <= 12, TP8-shaped. Separate entry point — no Collectives op
        takes a replicated weight plus bias operands."""
        from kimi_k3.kernels.sgl_adapters import comm_gemm
        return comm_gemm.gemm_ag_up_proj(
            self.state.world_size, x, weight, b, c, torch.empty_like(b))
