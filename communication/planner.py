"""Generic collective selection, profiling, and auto-map policy."""

from __future__ import annotations

import statistics
from typing import Callable

import os

import torch
import torch.distributed as dist

# Bare family name -> its default variant
_IMPL_ALIAS = {
    "torch_symm": "torch_symm:multimem",
    "flashinfer": "flashinfer:1shot",
    "b10_copy_engine:dma": "b10_copy_engine",
}

# The capacity-unbounded impl per op — any auto resolution whose impl
# cannot carry the operand lands here instead of raising.
_PLAIN_IMPL = {
    "all_gather": "nccl",
    "all_gather_col_quant": "nccl_seq",
    "reduce_scatter": "nccl",
    "all_reduce": "nccl",
    "quantized_all_reduce": "vllm_int8",
    "all_to_all": "nccl",
    "allreduce_norm": "seq",
    "gemm_allreduce": "seq",
    "allreduce_norm_gemm": "seq",
}

# CUDA-graph-capture swap: impls whose correctness depends on HOST-side
# state cannot be captured — AUTO resolutions swap to the capture-safe
# twin while capture is active; explicit ``impl=`` requests still fail
# loudly (the lamport backend additionally hard-asserts).
#   * ``b10_copy_engine:sm``: opens with a wait on the mover's PREVIOUS
#     (pre-capture) work — illegal inside capture.
#   * ``b10_multimem:lamport``: slot rotation + call counter are host
#     state baked at capture time -> WRONG outputs on replay.
#   * ``b10_multimem`` itself IS capture-safe since the per-block
#     DEVICE rounds fix (its barrier targets advance on replay);
#     original bug: local_result/prefill_graph_smallbatch_analysis.md §3.
_CAPTURE_UNSAFE = {
    "all_reduce": ("b10_copy_engine:sm", "b10_multimem:lamport"),
}
_CAPTURE_SWAP = {
    ("all_reduce", "b10_copy_engine:sm"): "torch_symm:2shot",
    ("all_reduce", "b10_multimem:lamport"): "torch_symm:multimem",
}

# Candidates profiled by autotune, in tie-break preference order.
#
# For fused ops, ``seq`` is the fully unfused baseline composed from
# autotuned communication plus separate compute kernels. ``norm_gemm_seq``
# keeps the best fused AR+norm but launches GEMM separately, isolating the
# benefit of the three-way AR+norm+GEMM fusion.
# K3_DISABLE_IMPLS: comma-separated impl (or family) names to drop from every
# candidate ladder, e.g. "torch_symm" removes torch_symm:1shot/2shot/multimem.
# For bring-up on fabrics where an impl is unusable, without code edits.
_DISABLED_IMPLS = tuple(
    x.strip() for x in os.environ.get("K3_DISABLE_IMPLS", "").split(",") if x.strip()
)


def _enabled(impl: str) -> bool:
    return not any(impl == d or impl.startswith(d + ":") or impl.split(":")[0] == d
                   for d in _DISABLED_IMPLS)


_OP_IMPLS: dict[str, tuple[str, ...]] = {
    "all_gather": ("torch_symm:multimem", "nccl_symm",
                   "torch_low_contention",
                   "b10_copy_engine", "b10_copy_engine:sm", "nccl"),
    "all_gather_col_quant": ("col_quant", "col_seq", "nccl_seq"),
    "reduce_scatter": ("nccl_symm", "torch_low_contention",
                       "b10_copy_engine",
                       "b10_copy_engine:sm", "nccl"),
    # b10_multimem:lamport is NOT a default candidate: its only win
    # was bs=64 by ~2.5%, below the own-kernel margin gate (explicit
    # impl="b10_multimem:lamport" still works)
    # sgl:push_res wins small messages, sgl:pull_res large (sglang's own
    # crossover ~512KB) — both are candidates; the profile decides.
    "all_reduce": ("flashinfer:1shot", "trt",
                   "sgl:push_res", "sgl:pull_res", "b10_multimem",
                   "torch_symm:multimem", "nccl_symm",
                   "torch_symm:1shot",
                   "torch_symm:2shot", "flashinfer:2shot",
                   "b10_copy_engine:sm", "nccl"),
    "quantized_all_reduce": ("vllm_int8", "vllm_fp8"),
    "all_to_all": ("nccl_symm", "b10_copy_engine",
                   "b10_copy_engine:sm", "nccl"),
    "allreduce_norm": ("flashinfer:1shot", "trt",
                       "sgl:push_norm", "sgl:pull_norm", "b10_multimem",
                       "flashinfer:2shot", "seq"),
    "gemm_allreduce": ("sgl:gemm_ar", "seq"),
    "allreduce_norm_gemm": ("norm_gemm_seq", "seq"),
}

if _DISABLED_IMPLS:
    _OP_IMPLS = {op: tuple(i for i in impls if _enabled(i)) or ("nccl",)
                 for op, impls in _OP_IMPLS.items()}



def _next_pow2(n: int) -> int:
    """Snap ``n`` up to the next power of two (1, 2, 4, ...)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# Own-kernel preference gate: OUR implementations must beat the best
# external (torch / flashinfer / trt / nccl / seq) candidate by this
# margin to win a bucket — otherwise ship the external one (less
# custom kernel surface in production for a sub-5% gain).
_OWN_PREFIXES = ("b10_", "col_", "cutedsl")
_OWN_WIN_MARGIN = 0.05


def _pow2_buckets(max_tokens: int) -> list[int]:
    """1, 2, 4, ... up to the smallest 2^k >= max_tokens."""
    out: list[int] = []
    b = 1
    while b < max_tokens:
        out.append(b)
        b <<= 1
    out.append(b)
    return out


class Planner:
    """Selection and autotuning bound to one Collectives facade."""

    def __init__(self, owner) -> None:
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def pick(self, op: str, shape) -> str:
        """The impl ``impl="auto"`` will choose for an operand of this
        shape ("local" at world size 1). Shape-only: no allocation, no
        collective — safe anywhere, including inside graph capture.
        Producers that see a symm-staged all_reduce pick can write into
        :meth:`symm_input` directly and skip the stage-in copy."""
        if self.world == 1:
            return "local"
        probe = torch.empty(shape, device="meta", dtype=self.dtype)
        return self._resolve(op, probe, "auto", lazy=False)

    # ------------------------------------------------------ autotune map

    def autotune(
        self,
        *,
        dims: tuple[int, ...] | None = None,
        max_tokens: int | None = None,
        ops: tuple[str, ...] = ("all_gather", "reduce_scatter",
                                "all_reduce", "all_to_all",
                                "allreduce_norm"),
        warmup: int = 3,
        iters: int = 20,
        log: bool = True,
        skip: tuple[str, ...] = (),
        clear: bool = True,
    ) -> dict[tuple[str, int, int], tuple[str, float]]:
        """Profile every candidate; fill the ``auto`` dispatch map.

        Buckets tokens to ``1, 2, 4, ..., 2^k >= max_tokens``. Timing
        uses CUDA events (median over interleaved windows); ranks
        ``all_reduce(MAX)`` so the winner minimizes the WORST rank.
        Near-tied candidates are re-measured, residual ties resolve by
        candidate order — the pick is stable across sessions. ``skip``
        drops backend names/families from the candidate sets.

        ``clear=False`` EXTENDS the existing map instead of rebuilding
        it: cells already present are kept (not re-profiled), so a
        cheap follow-up pass can add e.g. large-token buckets with
        fewer iters without discarding the primary pass's picks."""
        if self.world == 1:
            return {}
        if dims is None:
            dims = (self.max_hidden,)
        if max_tokens is None:
            max_tokens = max(1, self.max_numel // max(dims))
        buckets = _pow2_buckets(max_tokens)
        if clear:
            self._auto_map.clear()

        for op in ops:
            candidates = self._op_candidates(op, skip)
            if not candidates:
                continue
            for dim in dims:
                for bucket in buckets:
                    if not clear and (op, dim, bucket) in self._auto_map:
                        continue
                    need = bucket * dim
                    if op in ("all_gather", "reduce_scatter",
                              "all_to_all"):
                        need *= self.world
                    if need > self.max_numel:
                        continue
                    winner, us, rows = self._profile_cell(
                        op, dim, bucket, candidates, warmup, iters)
                    if winner is None:
                        continue
                    self._auto_map[(op, dim, bucket)] = (winner, us)
                    if log and self.rank == 0:
                        detail = " ".join(
                            f"{name}={t:.1f}" for name, t in rows)
                        print(
                            f"[collectives.autotune] {op} dim={dim} "
                            f"tokens={bucket:>5} -> {winner:<22} "
                            f"{us:7.1f}us  ({detail})",
                            flush=True,
                        )

        dist.barrier(self.group)
        if log and self.rank == 0:
            self.print_auto_map()
        return dict(self._auto_map)

    def export_auto_map(self) -> list[dict]:
        """The tuned lookup table as JSON-able rows (op, dim, tokens,
        impl, us) — pair with :meth:`import_auto_map` to persist the
        map across sessions instead of re-profiling."""
        return [
            dict(op=op, dim=dim, tokens=bucket, impl=impl, us=us)
            for (op, dim, bucket), (impl, us) in sorted(self._auto_map.items())
        ]

    def import_auto_map(self, rows: list[dict]) -> None:
        """Load rows produced by :meth:`export_auto_map` (or the
        lookup-table generator). Imported cells satisfy ``auto``
        lookups directly, so lazy autotune never re-profiles them.
        COLLECTIVE when a row names the trt impl (the workspace is
        built here so those winners are dispatchable)."""
        for row in rows:
            self._auto_map[(row["op"], row["dim"], row["tokens"])] = (
                row["impl"], float(row["us"]))
        if any(row["impl"] == "trt" for row in rows):
            self._trt_state()

    def print_auto_map(self) -> None:
        """Print the cached ``(op, dim, tokens) -> impl`` map (rank 0)."""
        if self.rank != 0:
            return
        if not self._auto_map:
            print("[collectives.autotune] map empty "
                  "(call autotune() first)", flush=True)
            return
        print("[collectives.autotune] === auto map "
              "(argmin max_rank_us, tokens=2^n) ===", flush=True)
        print(f"{'op':<20} {'dim':>6} {'tokens':>7} {'impl':<22} "
              f"{'us':>8}", flush=True)
        for (op, dim, bucket), (impl, us) in sorted(self._auto_map.items()):
            print(f"{op:<20} {dim:>6} {bucket:>7} {impl:<22} "
                  f"{us:>8.1f}", flush=True)

    def ensure_tuned(
        self,
        op: str,
        x: torch.Tensor,
        *,
        warmup: int = 3,
        iters: int = 20,
        log: bool = True,
        skip: tuple[str, ...] = (),
    ) -> str:
        """Lazily profile the ONE ``(op, dim, bucket)`` cell this
        operand maps to, if the auto map misses it; return the impl.
        COLLECTIVE + not capture-safe: every rank must call together,
        outside CUDA graphs. Only the operand's SHAPE is read."""
        if self.world == 1:
            return "local"
        tokens = x.shape[0]
        if op in ("reduce_scatter", "all_to_all"):
            tokens = x.shape[0] // self.world
        dim = (x.shape[1] * self.world
               if op == "all_gather_col_quant" else x.shape[1])
        bucket = _next_pow2(tokens)
        key = (op, dim, bucket)
        hit = self._auto_map.get(key)
        if hit is not None:
            return hit[0]
        need = bucket * dim
        if op in ("all_gather", "reduce_scatter", "all_to_all"):
            need *= self.world
        if need > self.max_numel:
            # oversize for the symm staging: don't profile (autotune
            # skips these cells too); resolve to the guarded fallback
            return self._guard(op, self._fallback(op, x), x)
        candidates = self._op_candidates(op, skip)
        if not candidates:
            return self._guard(op, self._fallback(op, x), x)
        winner, us, rows = self._profile_cell(
            op, dim, bucket, candidates, warmup, iters)
        if winner is None:
            return self._guard(op, self._fallback(op, x), x)
        self._auto_map[key] = (winner, us)
        if log and self.rank == 0:
            detail = " ".join(f"{name}={t:.1f}" for name, t in rows)
            print(f"[collectives.ensure_tuned] {op} dim={dim} "
                  f"tokens={bucket:>5} -> {winner:<22} {us:7.1f}us  "
                  f"({detail})", flush=True)
        return winner

    def _op_candidates(self, op: str,
                       skip: tuple[str, ...] = ()) -> list[str]:
        """The profile-able impls for ``op`` on this instance, minus
        the instance-level and per-call ``skip`` names/families."""
        skip = self._skip_impls + tuple(skip)
        configured = list(_OP_IMPLS[op])
        if self._cutedsl_enabled and op in (
                "allreduce_norm", "gemm_allreduce"):
            configured.append("cutedsl")
        cands = [c for c in configured
                 if c not in skip and c.partition(":")[0] not in skip]
        if self._symm is None:
            cands = [c for c in cands
                     if not c.startswith(("torch_symm",
                                          "torch_low_contention",
                                          "nccl_symm"))]
        if self._flashinfer is None and not (
                op == "allreduce_norm" and self._fi_norm is not None):
            cands = [c for c in cands if not c.startswith("flashinfer")]
        if not self._b10_ready():
            cands = [c for c in cands
                     if not c.startswith("b10_copy_engine")]
        if self._col_ag is None:
            cands = [c for c in cands if not c.startswith("col_")]
        if "trt" in cands and not self._trt_state():
            cands.remove("trt")
        if any(c.partition(":")[0] == "sgl" for c in cands) \
                and not self._sgl_state():
            cands = [c for c in cands if c.partition(":")[0] != "sgl"]
        if any(c.startswith("b10_multimem") for c in cands) \
                and not self._multimem_state():
            cands = [c for c in cands
                     if not c.startswith("b10_multimem")]
        if "cutedsl" in cands:
            cd = self._cutedsl_state()
            if not cd or not hasattr(cd, op):
                cands.remove("cutedsl")
        return cands

    def _agree(self, names: list[str]) -> list[str]:
        """Rank 0's view of a name list, broadcast to every rank (the
        MAX-reduced timings agree, but candidate sets could differ if
        an impl failed on one rank only)."""
        obj = [names]
        dist.broadcast_object_list(obj, src=0, group=self.group)
        return obj[0]

    def _profile_cell(
        self,
        op: str,
        dim: int,
        bucket: int,
        candidates: list[str],
        warmup: int,
        iters: int,
    ) -> tuple[str | None, float, list[tuple[str, float]]]:
        """Time each candidate; return (winner, max_us, [(name, us)]).

        Candidates the capacity gate rejects are skipped; the rest are
        timed interleaved (median over event windows, MAX over ranks).
        Near-ties are re-measured with heavier sampling and a residual
        tie resolves to the earlier candidate in ``candidates`` order,
        so the pick is stable across sessions. A window whose BEST
        time is off-scale (>10x a pessimistic 50 GB/s wire estimate,
        floor 500 us) was contaminated by concurrent load: re-measure
        once, then fall back to the plain impl rather than trust it."""
        if op == "all_gather_col_quant":
            if dim % self.world:
                raise ValueError(
                    "all_gather_col_quant output dim must divide world size")
            rows = 1
            probe = torch.empty(
                (bucket, dim // self.world), device="meta", dtype=self.dtype)
        else:
            rows = self.world if op in ("reduce_scatter", "all_to_all") else 1
            probe = torch.empty((bucket * rows, dim), device="meta",
                                dtype=self.dtype)
        fns: dict[str, Callable[[], None]] = {}
        for name in candidates:
            try:
                if self._fits(op, name, probe):
                    fns[name] = self._make_bench_fn(op, name, dim, bucket)
                    fns[name]()  # smoke: drop impls that cannot run here
            except Exception:  # noqa: BLE001
                fns.pop(name, None)
        if not fns:
            return None, float("nan"), []
        us = self._time_windows(fns, warmup, iters, reps=5)
        order = {n: i for i, n in enumerate(candidates)}

        itemsize = torch.empty((), dtype=self.dtype).element_size()
        nbytes = bucket * dim * itemsize * rows
        bound = max(500.0, 10.0 * nbytes / 50e3)  # contamination guard
        if min(us.values()) > bound:
            us = self._time_windows(fns, max(warmup, 4), iters, reps=5)
            if min(us.values()) > bound:
                plain = _PLAIN_IMPL[op]
                if self.rank == 0:
                    print(f"[collectives.autotune] WARNING: implausible "
                          f"({op}, dim={dim}, tokens={bucket}) window "
                          f"even after re-measure (best "
                          f"{min(us.values()):.1f}us > {bound:.0f}us) — "
                          f"storing plain {plain!r}", flush=True)
                return plain, us.get(plain, min(us.values())), \
                    sorted(us.items(), key=lambda kv: order[kv[0]])

        best = min(us.values())
        tie = self._agree([n for n in sorted(us, key=order.get)
                           if us[n] - best <= max(1.0, 0.05 * best)])
        tie = [n for n in tie if n in fns]
        if len(tie) > 1:  # re-measure near-ties with heavier sampling
            us.update(self._time_windows(
                {n: fns[n] for n in tie}, max(warmup, 4), iters * 2,
                reps=9))
            # own-kernel margin gate (winner pick only; the reported rows
        # keep every timing): our impls must beat the best external
        # candidate by _OWN_WIN_MARGIN or the external one wins
        pick = us
        ext = {n: t for n, t in us.items()
               if not n.startswith(_OWN_PREFIXES)}
        if ext:
            gate = min(ext.values()) * (1.0 - _OWN_WIN_MARGIN)
            pick = {n: t for n, t in us.items()
                    if not n.startswith(_OWN_PREFIXES) or t <= gate}
        best = min(pick.values())
        winner = self._agree([min(
            (n for n in pick if pick[n] - best <= max(0.3, 0.015 * best)),
            key=order.get)])[0]
        return winner, us.get(winner, best), \
            sorted(us.items(), key=lambda kv: order[kv[0]])

    def _make_bench_fn(
        self, op: str, impl: str, dim: int, bucket: int,
    ) -> Callable[[], None]:
        """A zero-alloc timed closure for one (op, impl, shape)."""
        def t(*shape):
            return (0.02 * torch.randn(*shape, device=self.device)) \
                .to(self.dtype)

        if op in ("all_gather", "all_reduce", "reduce_scatter",
                  "all_to_all"):
            rows = (bucket * self.world
                    if op in ("reduce_scatter", "all_to_all") else bucket)
            x = t(rows, dim)
            return lambda: getattr(self, op)(x, impl=impl)
        if op == "quantized_all_reduce":
            x = t(bucket, dim)
            return lambda: self.quantized_all_reduce(x, impl=impl)
        if op == "all_gather_col_quant":
            if dim % self.world:
                raise ValueError(
                    "all_gather_col_quant output dim must divide world size")
            x = t(bucket, dim // self.world)
            return lambda: self.all_gather_col_quant(x, impl=impl)
        if op == "allreduce_norm":
            x, g, r = t(bucket, dim), t(dim), t(bucket, dim)
            return lambda: self.allreduce_norm(
                x, g, 1e-6, residual=r, impl=impl)
        if op == "gemm_allreduce":
            k = self._boundary_gemm_k
            x, w = t(bucket, k), t(dim, k)
            return lambda: self.gemm_allreduce(x, w, impl=impl)
        if op == "allreduce_norm_gemm":
            n = self._boundary_gemm_n
            x, r, w = t(bucket, dim), t(bucket, dim), t(n, dim)
            g = torch.ones(dim, device=self.device, dtype=self.dtype)
            return lambda: self.allreduce_norm_gemm(
                x, r, g, w, impl=impl)
        raise ValueError(op)

    def _time_windows(
        self,
        fns: dict[str, Callable[[], None]],
        warmup: int,
        iters: int,
        reps: int,
    ) -> dict[str, float]:
        """Time every candidate INTERLEAVED: ``reps`` round-robin
        rounds, one CUDA-event window of ``iters`` calls per candidate
        per round; per-candidate MEDIAN over rounds, then MAX across
        ranks (us). The interleaving hands time-varying noise to every
        candidate's same-numbered window; the median then discards the
        affected windows outright."""
        names = list(fns)
        for name in names:
            fn = fns[name]
            for _ in range(warmup):
                fn()
        torch.cuda.synchronize()
        dist.barrier(self.group)
        wins: dict[str, list[float]] = {n: [] for n in names}
        for _ in range(max(reps, 1)):
            for name in names:
                fn = fns[name]
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(iters):
                    fn()
                e.record()
                e.synchronize()
                wins[name].append(s.elapsed_time(e) * 1000.0 / iters)
            dist.barrier(self.group)
        med = torch.tensor(
            [statistics.median(wins[n]) for n in names],
            device=self.device, dtype=torch.float64)
        dist.all_reduce(med, op=dist.ReduceOp.MAX, group=self.group)
        return dict(zip(names, med.tolist()))

    # ------------------------------------------------- resolve / guard

    def _fits(self, op: str, impl: str, x: torch.Tensor) -> bool:
        """Can ``impl`` carry this operand? Symm-staged impls are
        bounded by the ``max_numel`` staging, FlashInfer by its
        workspace; nccl / torch_low_contention / trt / seq are
        unbounded. Shape-only (meta-safe, no collective — an unbuilt
        lazy backend reads as unavailable)."""
        n = x.numel()
        family, _, variant = impl.partition(":")
        if x.dtype != self.dtype and impl != _PLAIN_IMPL[op]:
            return False
        if family == "torch_symm" and self._symm is None:
            return False
        if family == "torch_low_contention" and self._symm is None:
            return False
        if family == "nccl_symm" and self._symm is None:
            return False
        if family == "b10_copy_engine" and not self._b10_ready():
            return False
        if family == "trt" and not self._trt:
            return False
        if family == "flashinfer" and self._flashinfer is None and not (
                op == "allreduce_norm" and self._fi_norm is not None):
            return False
        if family == "sgl":
            # unbuilt lazy backend reads as unavailable (shape-only)
            return bool(self._sgl) and self._sgl.supports(op, x, variant)
        if op == "all_gather":
            if family == "b10_copy_engine":
                return self._b10.supports(op, x, variant or "dma")
            if impl == "torch_symm:multimem":
                # staged input + gathered [world*B, D] output slot
                return n * self.world <= self.max_numel
            if impl == "nccl_symm":
                return n * self.world <= self.max_numel
            return True  # nccl, torch_low_contention
        if op == "all_gather_col_quant":
            if impl in ("col_quant", "col_seq"):
                return self._col_ag_fits(x)
            return impl == "nccl_seq"
        if op in ("reduce_scatter", "all_to_all"):
            if family == "b10_copy_engine":
                return self._b10.supports(op, x, variant or "dma")
            if impl == "nccl_symm":
                return n <= self.max_numel
            return True
        if op == "all_reduce":
            if impl.startswith("b10_multimem"):
                return bool(self._mm_ar) and self._mm_ar.supports(op, x)
            if family == "flashinfer":
                return (self._flashinfer is not None
                        and self._flashinfer.fits(*x.shape)
                        and (variant != "2shot"
                             or x.shape[0] > self.world))
            if family == "torch_symm":
                return n <= self.max_numel
            if impl == "nccl_symm":
                return n <= self.max_numel
            if family == "b10_copy_engine":
                return self._b10.supports(op, x, variant or "dma")
            return True  # trt, nccl
        if op == "quantized_all_reduce":
            return (
                impl in ("vllm_int8", "vllm_fp8")
                and self._quantized is not None
                and n % 8 == 0
                and n >= 8192
            )
        if op == "allreduce_norm":
            if impl == "b10_multimem":
                return bool(self._mm_ar) and self._mm_ar.supports(op, x)
            if impl == "cutedsl":
                cd = self._cutedsl
                return bool(cd) and cd.supports(op, x.shape[0], x.shape[1])
            if family == "flashinfer":
                ok_shared = (self._flashinfer is not None
                             and self._flashinfer.fits(*x.shape))
                ok_norm = (variant == "1shot"
                           and self._fi_norm is not None
                           and x.shape[0] <= self._fused_ar_max_rows)
                if variant == "2shot" and x.shape[0] <= self.world:
                    return False
                return ok_shared or ok_norm
            return True  # trt, seq
        if op in ("gemm_allreduce", "allreduce_norm_gemm"):
            if impl == "cutedsl":
                cd = self._cutedsl
                return bool(cd) and cd.supports(op, x.shape[0], x.shape[1])
            return True  # seq
        return True

    def _guard(self, op: str, impl: str, x: torch.Tensor) -> str:
        """Capacity gate on every AUTO resolution: an impl that cannot
        carry the operand resolves to the plain unbounded impl instead
        (graceful oversize fallback, never a raise). During CUDA-graph
        capture, capture-unsafe impls swap to their safe twin. Explicit
        ``impl=`` requests are NOT guarded (autotune's bench closures
        rely on forced impls failing loudly)."""
        if not self._fits(op, impl, x):
            impl = _PLAIN_IMPL[op]
        if (impl in _CAPTURE_UNSAFE.get(op, ())
                and torch.cuda.is_current_stream_capturing()):
            impl = _CAPTURE_SWAP[(op, impl)]
            if not self._fits(op, impl, x):
                impl = _PLAIN_IMPL[op]
        return impl

    def _resolve(self, op: str, x: torch.Tensor, impl: str,
                 *, lazy: bool | None = None) -> str:
        impl = _IMPL_ALIAS.get(impl, impl)
        if impl != "auto":
            return impl
        tokens = x.shape[0]
        if op in ("reduce_scatter", "all_to_all"):
            # operand is world*B; map keys use the local/logical B
            tokens = x.shape[0] // self.world
        dim = (x.shape[1] * self.world
               if op == "all_gather_col_quant" else x.shape[1])
        bucket = _next_pow2(tokens)
        hit = self._auto_map.get((op, dim, bucket))
        if hit is not None:
            return self._guard(op, hit[0], x)
        # Lazy autotune (default): profile exactly this cell on first
        # use. COLLECTIVE — sound because TP layers issue the same
        # shapes on every rank; skipped inside graph capture (and by
        # pick(), which passes lazy=False to stay shape-only).
        lazy = self._lazy_autotune if lazy is None else lazy
        if lazy and not torch.cuda.is_current_stream_capturing():
            return self._guard(op, self.ensure_tuned(op, x), x)
        # nearest larger profiled bucket for this (op, dim), else smaller
        keys = sorted(
            (b for (o, d, b) in self._auto_map if o == op and d == dim))
        for b in keys:
            if b >= bucket:
                return self._guard(op, self._auto_map[(op, dim, b)][0], x)
        if keys:
            return self._guard(
                op, self._auto_map[(op, dim, keys[-1])][0], x)
        return self._guard(op, self._fallback(op, x), x)

    def _fallback(self, op: str, x: torch.Tensor) -> str:
        """Static heuristics when the auto map has no answer (inside
        graph capture, or with lazy_autotune off). Selection depends
        only on operation and the current tensor/context properties."""
        if op == "all_gather":
            if self._fits(op, "torch_symm:multimem", x):
                return "torch_symm:multimem"
            return "nccl"
        if op == "all_gather_col_quant":
            return "col_quant" if self._col_ag_fits(x) else "nccl_seq"
        if op in ("reduce_scatter", "all_to_all"):
            return "nccl"
        if op == "all_reduce":
            if self._fits(op, "flashinfer:1shot", x):
                return "flashinfer:1shot"
            if self._fits(op, "torch_symm:2shot", x):
                return "torch_symm:2shot"
            return "nccl"
        if op == "quantized_all_reduce":
            return "vllm_int8"
        if op == "allreduce_norm":
            return ("flashinfer:1shot"
                    if self._fits(op, "flashinfer:1shot", x) else "seq")
        if op in ("gemm_allreduce", "allreduce_norm_gemm"):
            return "seq"
        raise ValueError(op)

