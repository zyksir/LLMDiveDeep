"""Unified collective API: one dispatching class, many backends.

Ops (formula + shapes in each method's docstring):

    all_gather           [B, H] shard -> [world*B, H]
    reduce_scatter       [world*B, H] partials -> [B, H]
    all_reduce           [B, H] partials -> [B, H]
    quantized_all_reduce lossy FP8-wire sum -> BF16 [B, H]
    all_to_all           [world*B, H] -> [world*B, H]
    all_gather_col_quant column AG + quantized write-out (mxfp8)
    allreduce_norm       rmsnorm(allreduce(x) [+ residual], gamma, eps)
    gemm_allreduce       allreduce(x @ w.T)
    allreduce_norm_gemm  rmsnorm(allreduce(x) + residual, gamma, eps) @ w.T

Backends (``impl=`` names; wrappers live in ``backends/`` and custom
implementations in ``kernels/``; this file only dispatches):

    torch_symm:multimem | torch_symm:1shot | torch_symm:2shot
                         torch symmetric-memory kernels — if
                         torch.ops.symm_mem loads, ALL variants are
                         available
    torch_low_contention torch's low-contention symm-mem AG / RS
    b10_copy_engine      our LowContentionComm DMA mover
                         (b10_copy_engine:sm = SM copy-kernel mover)
    flashinfer:1shot | flashinfer:2shot
                         FlashInfer Lamport AR; the fused AR+residual+
                         RMSNorm kernel serves the norm ops
    trt                  TRT-LLM custom allreduce (MIN_LATENCY; fused
                         RESIDUAL_RMS_NORM kernel for allreduce_norm)
    vllm_int8 | vllm_fp8 vLLM/Kraken two-shot per-group quantized AR;
                         explicit lossy API only
    cutedsl              experimental fused boundary kernels; explicit
                         opt-in only, excluded from runtime/report candidates
    nccl                 plain torch.distributed (unbounded fallback)
    seq                  fully unfused baseline of each fused op:
                         best all_reduce + norm; mm + best all_reduce;
                         or best all_reduce + norm + mm
    norm_gemm_seq        best fused allreduce_norm + separate mm
    col_quant | col_seq | nccl_seq
                         fused column AG+MXFP8; tuned column AG then
                         quant; or NCCL column AG then quant

If flashinfer / tensorrt_llm import, their impls are available for
EVERY op they can serve; if torch.ops.symm_mem is available, every
torch_symm variant is. The b10 movers pre-allocate two ``max_numel``
symmetric buffers and are off by default (``enable_b10``; an explicit
``impl=`` request builds them on first call).

``impl="auto"`` looks up a profiled ``(op, dim, token_bucket) -> impl``
map. By default the map fills LAZILY: an ``auto`` call whose exact cell
is missing profiles just that cell on first use (``lazy_autotune``;
inside CUDA-graph capture, static heuristics answer instead). Upfront
:meth:`autotune` and :meth:`import_auto_map` also fill it;
``communication/bench_comm.py`` renders it as latency tables.
Autotune objective: every rank times each candidate, ranks
``all_reduce(MAX)``, pick ``argmin_impl(max_rank_us)`` — the winner
minimizes the WORST rank. Candidates are timed interleaved with median-
over-windows and near-tie re-measure, so the pick is stable across
sessions.

Every AUTO resolution is capacity-gated: an operand too big for a
backend's staging/workspace resolves to the unbounded plain impl
(nccl, or seq for the fused ops) — layers never need size checks.
``max_numel`` sizes the symm staging as a PERFORMANCE knob, never a
functional message limit.

The symm / FlashInfer / TRT transports need one NVLink domain. That is
NOT asserted at construction (GB200 NVL72 domains span hosts);
classic-cluster deployments can opt in via :meth:`assert_single_node`.

World size 1: every op returns its input (construction with
``group=None`` builds no transport), so layer code never branches on
world size.

Optional bounded column-AG and dedicated fused-AR workspaces are
configured at construction and exposed through the same facade.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .backends import (
    b10_copy_engine_backend,
    b10_multimem_backend,
    col_quant_backend,
    flashinfer_backend,
    nccl_backend,
    nccl_symm_backend,
    quantized_backend,
    torch_lc_backend,
    torch_symm_backend,
    trt_backend,
)
from .context import Ctx

from .planner import Planner

class Collectives:
    """Every open-source collective behind one dispatching API."""

    def __init__(
        self,
        group,
        max_numel: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
        *,
        max_hidden: int = 10752,
        enable_flashinfer: bool = True,
        flashinfer_max_tokens: int | None = None,
        enable_trt: bool = True,
        enable_b10: bool = False,
        enable_cutedsl: bool = False,
        col_ag_max_rows: int = 0,
        col_ag_max_output_columns: int | None = None,
        fused_ar_max_rows: int = 0,
        decode_ag_tokens: int = 0,
        fused_ar_tokens: int = 0,
        boundary_gemm_k: int = 896,
        boundary_gemm_n: int = 6288,
        boundary_max_tokens: int | None = None,
        lazy_autotune: bool = True,
        skip_impls: tuple[str, ...] = (),
    ) -> None:
        """``max_numel`` sizes the symm staging (performance knob, not
        a message limit). ``boundary_gemm_k`` / ``boundary_gemm_n`` are
        the GEMM contract dimensions used when profiling fused ops;
        ``boundary_max_tokens`` independently bounds pipeline workspaces
        so world-sized collective staging does not inflate them.

        ``enable_flashinfer`` builds the shared FlashInfer workspace at
        init (``flashinfer_max_tokens`` caps it independently of
        ``max_numel``); ``enable_trt`` allows the lazy TRT-LLM
        workspace (built on first use, collective); ``enable_b10``
        builds the b10 movers eagerly — off by default, an explicit
        ``impl=`` request still builds them on first call.
        ``enable_cutedsl`` adds experimental fused CuTeDSL kernels to
        autotune candidates; explicit ``impl="cutedsl"`` remains
        available when it is false.

        ``lazy_autotune``: an ``impl="auto"`` call whose exact
        ``(op, dim, bucket)`` cell is missing profiles just that cell
        on first use (collective — sound when all ranks issue the same
        shapes, as TP layers do; static heuristics answer inside
        CUDA-graph capture instead).

        ``skip_impls`` removes backends from every autotune candidate
        set — exact names (``"flashinfer:2shot"``) or whole families
        (``"trt"``, ``"b10_copy_engine"``). Useful when a backend's
        registration cost is unwanted (autotune()/ensure_tuned() also
        take a per-call ``skip``). Explicit ``impl=`` requests are
        never blocked.

        ``col_ag_max_rows`` and ``col_ag_max_output_columns`` size the
        bounded column all-gather engine. ``decode_ag_tokens`` is a
        backward-compatible alias for ``col_ag_max_rows``; without an
        explicit output width it derives capacity from ``max_numel``.
        ``fused_ar_tokens`` similarly aliases ``fused_ar_max_rows``."""
        if col_ag_max_rows and decode_ag_tokens:
            raise ValueError(
                "set only col_ag_max_rows or legacy decode_ag_tokens")
        if fused_ar_max_rows and fused_ar_tokens:
            raise ValueError(
                "set only fused_ar_max_rows or legacy fused_ar_tokens")
        col_ag_max_rows = col_ag_max_rows or decode_ag_tokens
        fused_ar_max_rows = fused_ar_max_rows or fused_ar_tokens
        if col_ag_max_output_columns is not None and col_ag_max_rows <= 0:
            raise ValueError(
                "col_ag_max_output_columns requires col_ag_max_rows")
        if col_ag_max_rows > 0 and col_ag_max_output_columns is None:
            col_ag_max_output_columns = max(1, max_numel // col_ag_max_rows)
        self.group = group
        self.world = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group=group) if group is not None else 0
        self.dtype = dtype
        self.max_numel = max_numel
        self.max_hidden = max_hidden
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        # (op, dim, token_bucket) -> (impl, max_rank_us)
        self._auto_map: dict[tuple[str, int, int], tuple[str, float]] = {}
        self._lazy_autotune = lazy_autotune
        self._skip_impls = tuple(skip_impls)
        self._boundary_gemm_k = boundary_gemm_k
        self._boundary_gemm_n = boundary_gemm_n
        self._boundary_max_tokens = (
            boundary_max_tokens
            if boundary_max_tokens is not None
            else max(1, max_numel // max_hidden)
        )
        self._fused_ar_max_rows = fused_ar_max_rows
        self._symm = None
        self._flashinfer = None
        self._b10 = None
        self._b10_enabled = enable_b10
        self._cutedsl_enabled = enable_cutedsl
        # None = untried (lazy, collective build), False = unavailable
        self._trt: trt_backend.TrtBackend | bool | None = \
            None if enable_trt else False
        self._col_ag = None
        self._fi_norm = self._fi_dedicated = None
        # Experimental backend; explicit-only unless enable_cutedsl=True.
        self._cutedsl = None
        self._mm_ar = None
        self._quantized = None
        self._planner = Planner(self)
        if self.world == 1:
            # every op short-circuits to the identity: no transport
            return
        self.ctx = Ctx(group, self.world, self.rank, self.device,
                       dtype, max_numel)
        self._quantized = quantized_backend.QuantizedBackend(
            self.ctx, self._boundary_max_tokens * max_hidden)
        if torch_symm_backend.available():
            self._symm = torch_symm_backend.TorchSymmBackend(self.ctx)
        self._b10 = b10_copy_engine_backend.B10CopyEngineBackend(
            self.ctx,
            shared_buffer=(self._symm._symm_out
                           if self._symm is not None else None))
        if enable_b10:
            self._b10.build()
        if enable_flashinfer and flashinfer_backend.available():
            self._flashinfer = flashinfer_backend.FlashInferBackend(
                self.ctx,
                max_tokens=(flashinfer_max_tokens
                            or max(1, max_numel // max(1, max_hidden))),
                max_hidden=max_hidden,
            )
        # The fused norm and shared reductions may run concurrently, so
        # each gets a dedicated workspace.
        if col_ag_max_rows > 0:
            self._col_ag = col_quant_backend.build_col_quant_all_gather(
                self.rank,
                self.world,
                col_ag_max_rows,
                col_ag_max_output_columns,
            )
        if fused_ar_max_rows > 0:
            self._fi_norm = flashinfer_backend.FlashInferAllReduce(
                self.rank, self.world, max_tokens=fused_ar_max_rows)
            self._fi_dedicated = flashinfer_backend.FlashInferAllReduce(
                self.rank, self.world, max_tokens=fused_ar_max_rows)

    # ------------------------------------------------- backend state

    def _b10_ready(self) -> bool:
        return self._b10 is not None and (self._b10_enabled
                                          or self._b10.ready())

    def _multimem_state(self):
        """Our wide-MLP NVLS multimem AR kernel (b10_multimem), built
        on first use. COLLECTIVE on first call (symm rendezvous + JIT
        build). False when sm100/NVLS/nvcc is unavailable."""
        if self._mm_ar is None:
            self._mm_ar = False
            try:
                self._mm_ar = b10_multimem_backend.B10MultimemBackend(self.ctx)
            except Exception:  # noqa: BLE001 (optional backend)
                pass
        return self._mm_ar

    def _cutedsl_state(self):
        """Build experimental CuTeDSL kernels only for explicit requests."""
        if self._cutedsl is None:
            self._cutedsl = False
            try:
                from .backends import cutedsl_backend
                self._cutedsl = cutedsl_backend.CuteDslBackend(
                    self.ctx, self._boundary_max_tokens)
            except Exception:  # noqa: BLE001 (optional backend)
                pass
        return self._cutedsl

    def _require_cutedsl(self):
        backend = self._cutedsl_state()
        if not backend:
            raise RuntimeError("cutedsl backend unavailable")
        return backend

    def _trt_state(self):
        """The TRT backend, built on first use. COLLECTIVE on first
        call (IPC workspace rendezvous) — reached only from explicit
        ``impl=`` requests and autotune/lazy-tune paths, which are
        rank-lockstep. False when tensorrt_llm is not installed or the
        group is a proper subgroup (TRT-LLM assumes the whole world is
        the TP group)."""
        if self._trt is None:
            self._trt = False
            if (trt_backend.available()
                    and dist.get_world_size(self.group)
                    == dist.get_world_size()):
                try:
                    self._trt = trt_backend.TrtBackend(self.ctx)
                except Exception:  # noqa: BLE001 (optional backend)
                    pass
        return self._trt

    def assert_single_node(self) -> None:
        """OPTIONAL guard for classic (non-NVLink-fabric) clusters.

        The symm-mem / FlashInfer-IPC / TRT transports require every
        rank in one NVLink domain — on classic clusters, one node.
        NOT called automatically: on NVLink-fabric systems (GB200
        NVL72) a domain legitimately spans hosts and this hostname
        check would misfire. COLLECTIVE: every rank must call
        together."""
        if self.world == 1:
            return
        import socket

        hosts: list[str | None] = [None] * self.world
        dist.all_gather_object(hosts, socket.gethostname(), group=self.group)
        if len(set(hosts)) != 1:
            raise RuntimeError(
                "Collectives group spans hosts "
                f"{sorted(set(hosts))} — the symm-mem/multimem/"
                "FlashInfer-IPC transports need one NVLink domain. On "
                "a classic cluster use a per-node subgroup for these "
                "ops and NCCL across nodes (skip this check on "
                "NVLink-fabric systems like GB200 NVL72).")
        if self.world > torch.cuda.device_count():
            raise RuntimeError(
                f"group world size {self.world} exceeds this node's "
                f"{torch.cuda.device_count()} visible GPUs — "
                "cross-node group?")

    # ------------------------------------------------------------ staging

    def symm_input(self, shape, dtype=None, *, impl: str = "torch_symm",
                   offset: int = 0) -> torch.Tensor:
        """A symm-resident view to produce into: hands the symm-staged
        impls their operand without the stage-in copy. ``impl`` (a
        :meth:`pick` result) selects WHICH staging buffer — the
        torch_symm family and b10_multimem own separate ones.
        ``offset`` (elements) partitions a buffer between two live
        operands (e.g. a second AR's producer writing above the first
        AR's [0, n) staging region — see :meth:`staging_overlaps`)."""
        if impl.startswith("torch_symm"):
            assert self._symm is not None, \
                "no symm staging at this world size"
            return self._symm.symm_input(shape, dtype, offset)
        if impl == "b10_multimem":
            mm = self._multimem_state()
            assert mm, "b10_multimem backend unavailable"
            assert dtype is None or dtype == self.dtype
            return mm.symm_input(shape, offset)
        raise ValueError(f"impl {impl!r} has no symm staging buffer")

    @staticmethod
    def symm_staged(impl: str) -> bool:
        """True when this all_reduce impl pays a stage-in copy that a
        producer can skip by writing into :meth:`symm_input` (the
        torch_symm family and plain b10_multimem; nccl/trt/flashinfer
        read their operand directly — nothing to save)."""
        return impl.startswith("torch_symm") or impl == "b10_multimem"

    @staticmethod
    def staging_overlaps(impl_a: str, impl_b: str) -> bool:
        """True when two symm-staged impls share ONE staging buffer —
        consecutive collectives then need disjoint ``offset`` regions
        for zero-copy producers."""
        def fam(impl: str) -> str:
            return "torch_symm" if impl.startswith("torch_symm") else impl
        return (Collectives.symm_staged(impl_a)
                and Collectives.symm_staged(impl_b)
                and fam(impl_a) == fam(impl_b))

    # ------------------------------------------------------ planner facade

    def pick(self, op: str, shape) -> str:
        return self._planner.pick(op, shape)

    def autotune(self, **kwargs):
        return self._planner.autotune(**kwargs)

    def export_auto_map(self) -> list[dict]:
        return self._planner.export_auto_map()

    def import_auto_map(self, rows: list[dict]) -> None:
        self._planner.import_auto_map(rows)

    def print_auto_map(self) -> None:
        self._planner.print_auto_map()

    def ensure_tuned(self, op: str, x: torch.Tensor, **kwargs) -> str:
        return self._planner.ensure_tuned(op, x, **kwargs)

    def _op_candidates(self, op: str, skip=()):
        return self._planner._op_candidates(op, skip)

    def _profile_cell(self, *args, **kwargs):
        return self._planner._profile_cell(*args, **kwargs)

    def _fits(self, op: str, impl: str, x: torch.Tensor) -> bool:
        return self._planner._fits(op, impl, x)

    def _guard(self, op: str, impl: str, x: torch.Tensor) -> str:
        return self._planner._guard(op, impl, x)

    def _resolve(self, op: str, x: torch.Tensor, impl: str, *, lazy=None) -> str:
        return self._planner._resolve(op, x, impl, lazy=lazy)

    def _fallback(self, op: str, x: torch.Tensor) -> str:
        return self._planner._fallback(op, x)

    # ------------------------------------------------------------ ops

    def all_gather(self, x: torch.Tensor,
                   impl: str = "auto") -> torch.Tensor:
        """``concat_world(x)`` along dim 0.

        x:       [B, H]  my shard
        returns  [world*B, H]  rank-major

        World size 1 returns ``x``."""
        if self.world == 1:
            return x
        impl = self._resolve("all_gather", x, impl)
        if impl == "torch_symm:multimem":
            return self._symm.all_gather(x)
        if impl == "nccl_symm":
            return nccl_symm_backend.all_gather(self.group, self._symm, x)
        if impl == "torch_low_contention":
            return torch_lc_backend.all_gather(
                self.group, self._symm.staged(x))
        if impl == "b10_copy_engine":
            return self._b10.all_gather(x, "dma")
        if impl == "b10_copy_engine:sm":
            return self._b10.all_gather(x, "sm")
        if impl == "nccl":
            return nccl_backend.all_gather(self.group, x)
        raise ValueError(f"unknown all_gather impl {impl!r}")

    def reduce_scatter(self, x: torch.Tensor,
                       impl: str = "auto") -> torch.Tensor:
        """``sum_world(x)``, scattered: my row block of the total.

        x:       [world*B, H]  partials
        returns  [B, H]  my reduced block

        World size 1 returns ``x``."""
        if self.world == 1:
            return x
        impl = self._resolve("reduce_scatter", x, impl)
        if impl == "nccl_symm":
            return nccl_symm_backend.reduce_scatter(
                self.group, self._symm, x)
        if impl == "torch_low_contention":
            return torch_lc_backend.reduce_scatter(
                self.group, self._symm.staged(x))
        if impl == "b10_copy_engine":
            return self._b10.reduce_scatter(x, "dma")
        if impl == "b10_copy_engine:sm":
            return self._b10.reduce_scatter(x, "sm")
        if impl == "nccl":
            return nccl_backend.reduce_scatter(self.group, x)
        raise ValueError(f"unknown reduce_scatter impl {impl!r}")

    def all_reduce(self, x: torch.Tensor,
                   impl: str = "auto") -> torch.Tensor:
        """``sum_world(x)`` on every rank.

        x:       [B, H]  rank-local partial
        returns  [B, H]  reduced

        ALIASING: the torch_symm impls return a VIEW of this
        instance's staging buffer, valid until the next symm
        collective — consume or ``.clone()`` first. Other impls return
        owned tensors. World size 1 returns ``x``."""
        if self.world == 1:
            return x
        impl = self._resolve("all_reduce", x, impl)
        family, _, variant = impl.partition(":")
        if family == "flashinfer":
            return self._flashinfer.all_reduce(x, variant == "1shot")
        if impl == "trt":
            return self._require_trt().all_reduce(x)
        if family == "torch_symm":
            return self._symm.all_reduce(x, variant)
        if impl == "nccl_symm":
            return nccl_symm_backend.all_reduce(self.group, self._symm, x)
        if impl == "b10_multimem:lamport":
            return self._multimem_state().all_reduce_lamport(x)
        if impl == "b10_multimem":
            return self._multimem_state().all_reduce(x)
        if impl == "b10_copy_engine:sm":
            return self._b10.all_reduce(x, "sm")
        if impl == "b10_copy_engine":
            return self._b10.all_reduce(x, "dma")
        if impl == "nccl":
            return nccl_backend.all_reduce(self.group, x)
        raise ValueError(f"unknown all_reduce impl {impl!r}")

    def quantized_all_reduce(
        self,
        x: torch.Tensor,
        *,
        impl: str = "vllm_int8",
    ) -> torch.Tensor:
        """Lossy ``sum_world(x)`` with a lower-precision wire format.

        The vLLM/Kraken two-shot kernels quantize BF16 per group to
        INT8 or E4M3, reduce through symmetric NVLink buffers, and return
        BF16. This API is deliberately separate from exact
        :meth:`all_reduce`; lossy transport is never an ``auto`` candidate."""
        if self.world == 1:
            return x
        if impl == "vllm_int8":
            return self._quantized.all_reduce(x, use_fp8=False)
        if impl == "vllm_fp8":
            return self._quantized.all_reduce(x, use_fp8=True)
        raise ValueError(f"unknown quantized_all_reduce impl {impl!r}")

    def all_to_all(self, x: torch.Tensor,
                   impl: str = "auto") -> torch.Tensor:
        """Chunk d of my rows goes to rank d.

        x:       [world*B, H]  chunk d -> rank d
        returns  [world*B, H]  received, source-major

        World size 1 returns ``x``."""
        if self.world == 1:
            return x
        impl = self._resolve("all_to_all", x, impl)
        if impl == "nccl_symm":
            return nccl_symm_backend.all_to_all(
                self.group, self._symm, x)
        if impl == "b10_copy_engine":
            return self._b10.all_to_all(x, "dma")
        if impl == "b10_copy_engine:sm":
            return self._b10.all_to_all(x, "sm")
        if impl == "nccl":
            return nccl_backend.all_to_all(self.group, x)
        raise ValueError(f"unknown all_to_all impl {impl!r}")

    def _require_trt(self):
        trt = self._trt_state()
        if not trt:
            raise RuntimeError(
                "trt backend unavailable (tensorrt_llm not installed, "
                "subgroup, or enable_trt=False)")
        return trt

    def allreduce_norm(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float = 1e-5,
        *,
        residual: torch.Tensor | None = None,
        impl: str = "auto",
    ):
        """``rmsnorm(allreduce(x) [+ residual], gamma, eps)``

        x:        [B, H]  rank-local partial
        gamma:    [H]
        residual: [B, H]  optional
        returns   norm [B, H], or (norm [B, H], new_residual [B, H])
                  when ``residual`` is given — new_residual is
                  ``allreduce(x) + residual``, the next residual stream

        No output buffers anywhere: every backend returns the tensors
        its kernels produced (own allocations, graph-pool tensors, or
        views of internal staging) — valid until the next call on this
        instance; clone to persist. Seq version (``impl="seq"``):
        ``all_reduce`` then the norm as separate dispatched calls.
        World size 1 skips the AR."""
        tokens, hidden = x.shape
        if self.world == 1:
            src = x if residual is None else x + residual
            norm = torch.nn.functional.rms_norm(
                src, (hidden,), gamma, eps)
            return norm if residual is None else (norm, src)
        impl = self._resolve("allreduce_norm", x, impl)
        family, _, variant = impl.partition(":")
        if family == "flashinfer":
            # Prefer the dedicated bounded workspace when wired and fitting.
            ws = None
            if (variant == "1shot" and self._fi_norm is not None
                    and tokens <= self._fused_ar_max_rows):
                ws = self._fi_norm._workspace
            if self._flashinfer is None:
                # The dedicated wrapper returns norm only, so residual
                # calls use the sequential fallback.
                if residual is not None:
                    return self.allreduce_norm(
                        x, gamma, eps, residual=residual, impl="seq")
                return self._fi_norm.norm_reduce(
                    x, gamma, self.ctx.zero_residual(tokens, hidden),
                    eps)
            norm, res = self._flashinfer.allreduce_norm(
                x, gamma, eps, residual,
                oneshot=variant == "1shot", workspace=ws)
            return norm if residual is None else (norm, res)
        if impl == "trt":
            norm, res = self._require_trt().allreduce_norm(
                x, gamma, eps, residual)
            return norm if residual is None else (norm, res)
        if impl == "b10_multimem":
            res_in = residual if residual is not None else \
                self.ctx.zero_residual(tokens, hidden)
            norm, res = self._multimem_state().allreduce_norm(
                x, gamma, eps, res_in)
            return norm if residual is None else (norm, res)
        if impl == "cutedsl":
            res_in = residual if residual is not None else \
                self.ctx.zero_residual(tokens, hidden)
            norm, res = self._require_cutedsl().allreduce_norm(
                x, gamma, eps, res_in)
            return norm if residual is None else (norm, res)
        if impl == "seq":
            red = self.all_reduce(x)
            res = None
            if residual is not None:
                res = torch.add(red, residual)
                red = res
            norm = flashinfer_backend.rmsnorm(red, gamma, eps)
            return norm if residual is None else (norm, res)
        raise ValueError(f"unknown allreduce_norm impl {impl!r}")

    def gemm_allreduce(self, x: torch.Tensor, w: torch.Tensor,
                       impl: str = "auto") -> torch.Tensor:
        """``allreduce(x @ w.T)`` — the row-parallel projection tail.

        x:       [B, K_local]  my K-shard activation
        w:       [N, K_local]  my K-shard weight
        returns  [B, N]  reduced on every rank

        No output buffers: each backend returns its kernels' native
        output with NO copy — a fresh/graph-pool tensor or a view of
        internal/symm staging, valid until the next call on this
        instance (clone to persist). Seq version (``impl="seq"``): the
        GEMM, then ``all_reduce`` on its output as separate dispatched
        calls. World size 1 is just the GEMM."""
        n = w.shape[0]
        if self.world == 1:
            return torch.mm(x, w.T)
        probe = torch.empty((x.shape[0], n), device="meta", dtype=self.dtype)
        impl = self._resolve("gemm_allreduce", probe, impl)
        if impl == "cutedsl":
            return self._require_cutedsl().gemm_allreduce(x, w)
        if impl == "seq":
            return self.all_reduce(torch.mm(x, w.T))
        raise ValueError(f"unknown gemm_allreduce impl {impl!r}")

    def allreduce_norm_gemm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        w: torch.Tensor,
        eps: float = 1e-5,
        impl: str = "auto",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``rmsnorm(allreduce(x) + residual, gamma, eps) @ w.T`` —
        the MoE-out -> next-layer boundary as ONE schedule.

        x:        [B, H]  rank-local partial
        residual: [B, H]
        gamma:    [H]
        w:        [N, H]
        returns   (y [B, N], new_residual [B, H]) — new_residual is
                  ``allreduce(x) + residual``, the next residual stream

        No output buffers: backends return their kernels' native
        outputs with NO copy (fresh/graph-pool tensors or views of
        internal staging), valid until the next call on this instance.
        ``impl="seq"`` is best AR + separate norm + separate GEMM.
        ``impl="norm_gemm_seq"`` keeps the best fused AR+norm and uses
        a separate GEMM. World size 1 skips the AR."""
        tokens, hidden = x.shape
        if self.world == 1:
            res = x + residual
            return torch.mm(torch.nn.functional.rms_norm(
                res, (hidden,), gamma, eps), w.T), res
        impl = self._resolve("allreduce_norm_gemm", x, impl)
        if impl == "cutedsl":
            return self._require_cutedsl().allreduce_norm_gemm(
                x, residual, gamma, w, eps)
        if impl == "seq":
            norm, res = self.allreduce_norm(
                x, gamma, eps, residual=residual, impl="seq")
            return torch.mm(norm, w.T), res
        if impl == "norm_gemm_seq":
            norm, res = self.allreduce_norm(
                x, gamma, eps, residual=residual, impl="auto")
            return torch.mm(norm, w.T), res
        raise ValueError(f"unknown allreduce_norm_gemm impl {impl!r}")

    # ------------------------------------- bounded engines (facade)

    @property
    def has_col_ag(self) -> bool:
        return self._col_ag is not None

    @property
    def has_fused_norm_reduce(self) -> bool:
        return self._fi_norm is not None

    def _col_ag_fits(self, x: torch.Tensor) -> bool:
        """Does the gathered [B, world*C] message fit the wired column
        AG engine's Lamport slots?"""
        if self._col_ag is None or not self._col_ag.comms:
            return False
        out_bytes = x.numel() * self.world * x.element_size()
        return out_bytes <= self._col_ag.comms[0].slot_bytes

    def all_gather_col(self, x: torch.Tensor) -> torch.Tensor:
        """Tuned COLUMN all-gather.

        x:       [B, C]  my column shard
        returns  [B, world*C]  (may be a view of the engine's slot,
                 valid until its next call)

        World size 1 returns ``x``. Oversize for the wired engine (or
        no engine) falls back to the NCCL column AG — never a capacity
        assert."""
        if self.world == 1:
            return x
        if not self._col_ag_fits(x):
            return col_quant_backend.nccl_all_gather_cols(x)
        return self._col_ag(x)

    def all_gather_col_quant(
        self,
        x: torch.Tensor,
        quant: str = "mxfp8",
        *,
        impl: str = "auto",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``quant(concat_cols_world(x))`` — column AG with the
        quantize fused on the write-out.

        x:       [B, C]  bf16 column shard
        returns  ([B, world*C] float8_e4m3, ue8m0 scales)

        Bit-exact with ``torch.ops.trtllm.mxfp8_quantize(ag(x))``.
        ``impl="col_quant"`` fuses quantization into the tuned column
        AG write-out. ``"col_seq"`` uses that same tuned AG followed by
        a separate quant kernel; ``"nccl_seq"`` is the unbounded NCCL
        AG + quant baseline. Auto profiles all fitting variants and
        falls back to ``nccl_seq`` when the bounded engine is absent or
        too small."""
        if quant != "mxfp8":
            raise ValueError(f"unsupported quant {quant!r} "
                             "(only 'mxfp8' is wired)")
        if self.world == 1:
            data, scale = torch.ops.trtllm.mxfp8_quantize(
                x.contiguous(), False)
            return data, scale.view(x.shape[0], -1)
        impl = self._resolve("all_gather_col_quant", x, impl)
        if impl == "col_quant":
            data, scale = self._col_ag.all_gather_mxfp8(x)
            return data, scale.view(x.shape[0], -1)
        if impl == "col_seq":
            gathered = self._col_ag(x)
            data, scale = torch.ops.trtllm.mxfp8_quantize(
                gathered.contiguous(), False)
            return data, scale.view(x.shape[0], -1)
        if impl == "nccl_seq":
            data, scale = torch.ops.trtllm.mxfp8_quantize(
                col_quant_backend.nccl_all_gather_cols(x).contiguous(), False)
            return data, scale.view(x.shape[0], -1)
        raise ValueError(f"unknown all_gather_col_quant impl {impl!r}")

    def tune_col_ag(self, x: torch.Tensor) -> None:
        """Tune the column-AG plan for this shape OUTSIDE graph capture
        (first captured call would otherwise tune inside the graph)."""
        if self._col_ag is not None and self.world > 1:
            self._col_ag.tune(x)

    def all_reduce_dedicated(self, x: torch.Tensor) -> torch.Tensor:
        """Plain oneshot AR on the second dedicated fused-AR
        workspace — safe to run CONCURRENTLY with
        :meth:`allreduce_norm`'s dedicated workspace. World size 1
        returns ``x``. Oversize (or none wired) falls back to plain
        NCCL AR."""
        if self.world == 1:
            return x
        if self._fi_dedicated is None or x.shape[0] > self._fused_ar_max_rows:
            return nccl_backend.all_reduce(self.group, x)
        return self._fi_dedicated(x)

    def all_reduce_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for :meth:`all_reduce_dedicated`."""
        return self.all_reduce_dedicated(x)

    def destroy(self) -> None:
        """Tear down the FlashInfer IPC workspaces (best effort)."""
        for fi in (self._fi_norm, self._fi_dedicated, self._flashinfer):
            if fi is not None:
                fi.destroy()
        self._fi_norm = self._fi_dedicated = self._flashinfer = None
