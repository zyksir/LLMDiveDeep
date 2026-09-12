"""CuTeDSL standalone gated-RMSNorm (KDA/GDN output o_norm) — black-box, importable.

IDENTICAL TO (same end-to-end function, just faster):
  - SGLang Triton ``FusedRMSNormGated`` / ``layer_norm_fwd_kernel``
    (sglang/srt/layers/attention/fla/fused_norm_gate.py, activation="sigmoid")
  - the same kernel as vendored by vLLM / FLA
Same math and I/O: ``o = r * rsqrt(mean(r^2)+1e-5) * w * sigmoid(z)``.

DROP-IN API? ALMOST — free-function twin of rms_norm_gated / FusedRMSNormGated.
  Triton: rms_norm_gated(x, g, weight, bias=None, activation="sigmoid", ...)
           or FusedRMSNormGated(...).forward(x, g)
  This:   gated_rmsnorm(r, z, w)  # names differ; always sigmoid; no bias/residual
  One-line: o = gated_rmsnorm(x, g, weight)  if you rename args and pass
           bias=None / activation="sigmoid" on the Triton side equivalently.


Extracted from CuTeDSLGen champion gen_gated_rmsnorm_claude_r2_0729_1603/best/run.py.
Computes o = (r * rsqrt(mean_d(r^2)+1e-5)) * w * sigmoid(z), RMS over the 128 head dim
(matches sglang FusedRMSNormGated(activation="sigmoid") to bf16 precision). Uses the
TRT-derived recipe: one-warp-per-row + warp-shuffle reduction (no SMEM/barrier), 128-bit
vectorized loads, single HBM pass, fold rstd*w, fast-math rsqrt/rcp_approx-sigmoid, no
training scratch. WIN: 3.7-4.1 us vs SGLang/vLLM's launch-bound FLA Triton norm (~20 us)
= 5-6x faster, at N in {4..1024}, H=16, D=128.

    from kda.b10.b10_kda_gated_rmsnorm_cutedsl import gated_rmsnorm
    o = gated_rmsnorm(r, z, w)   # r,z [N,H,128] bf16; w [128] fp32 -> o [N,H,128] bf16
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as T
import cuda.bindings.driver as cuda

D = 128                 # head dim (row width) -- structural, see design.md §7
EPS = 1.0e-5            # matches FusedRMSNormGated(eps=1e-5)
RMEAN = 1.0 / D         # mean over the head dim is a constant multiply, not a divide

# ---------------------------------------------------------------------------
# Compile-time levers. Defaults are the measured optimum (see optimization.md).
# ---------------------------------------------------------------------------
# Threads per CTA. 128 = 4 warps = 8 independent rows per step.
NT = int(os.environ.get("GRN_NT", "128"))
# bf16 values per thread. 8 == 16 B == one LDG.128/STG.128, and makes a row
# exactly 16 lanes (a half warp), which is what removes SMEM and barriers.
# 16 is the other legal setting (8 threads/row, 3 shuffle steps, 2 LDGs/row).
VPT = int(os.environ.get("GRN_VPT", "8"))
# Rows per thread (steps per CTA). MEASURED: 1 is fastest at every spec shape,
# and the sweep is monotonic (N=1024: 2.86 / 3.13 / 3.31 / 3.72 / 4.03 us for
# RPT = 1 / 2 / 4 / 8 / 16). Batching rows per thread amortizes the two `w`
# loads, but it divides the CTA count by RPT, and CTA count is what balances the
# grid over 148 SMs: 2048 CTAs land 13 or 14 per SM (1.2% quantization), while
# 512 CTAs land 3 or 4 (15.6%). Granularity beats amortization by ~14%.
_RPT_ENV = os.environ.get("GRN_RPT")
RPT_DEFAULT = 1
# `__launch_bounds__(NT, MINCTA)`. 0 = let ptxas pick (a forced bound is a
# promise ptxas must keep even by spilling; this kernel has no register
# pressure, so there is nothing to buy).
MINCTA = int(os.environ.get("GRN_MINCTA", "0"))
# Sigmoid implementation. The gate is the ONLY expensive arithmetic in this
# kernel: at 128-bit vector width the memory side is ~3 instructions per row,
# while `1/(1+exp(-z))` naively costs TWO MUFU ops per *element* (ex2 + rcp),
# and MUFU runs at 16/SM/cycle against FP32's 128/SM/cycle. See optimization.md.
#   0 = exp + per-element rcp        2.000 MUFU/elem  (accurate, the obvious form)
#   1 = exp + batch-inverted rcp     1.125 MUFU/elem  (accurate: one rcp per VPT
#                                     elements via the stable u=exp(-|z|) form)
#   2 = tanh.approx.f32              1.000 MUFU/elem  (approximate)
#   8 = ABLATION: rcp only, no ex2   1.000 MUFU/elem  (INCORRECT: measurement only)
#   9 = ABLATION: no MUFU at all     0.000 MUFU/elem  (INCORRECT: measurement only)
SIG = int(os.environ.get("GRN_SIG", "0"))
# ABLATION lever, measurement only -- every non-zero value produces WRONG results
# on purpose, to price one part of the kernel (see optimization.md). The point is
# that a memory-bound kernel is only "done" when the parts you can delete are
# worth nothing.
#   bit 0 (1) = don't load w    (use w == 1.0)
#   bit 1 (2) = don't reduce    (use rstd == 1.0; drops the shuffles)
#   bit 2 (4) = don't load z    (use sigmoid(z) == 1.0; drops one whole stream)
ABL = int(os.environ.get("GRN_ABL", "0"))
# Critical-path diet. At the spec shapes this kernel is LATENCY-bound, not
# bandwidth-bound (see optimization.md: it already runs at 95.8% of HBM peak once
# the working set exceeds L2, and its ablations are non-additive). So what is
# left to win is the length of the dependency chain between `r` arriving in
# registers and the store issuing.
#   0 = the straightforward form: sequential sum-of-squares, then r*sc*w*sig
#   1 = both of the below
#   2 = (a) only: balanced binary tree for the local sum of squares (3 dependent
#           FADDs instead of a VPT-deep serial FFMA chain)
#   3 = (b) only: exactly ONE multiply downstream of `sc`. r*w*sigmoid(z) does
#           not depend on the reduction, so it is hoisted above the butterfly and
#           only the final `* sc` waits on it.
# MEASURED (N=1024): 0 -> 2.835us, 1 -> 2.900us, 2/3 -> see optimization.md. Both
# halves lengthen the live range of VPT extra fp32 values, and 32 registers is
# exactly the 16-blocks-per-SM occupancy cliff, so shortening the ALU chain of a
# MEMORY-latency-bound kernel buys nothing and costs occupancy.
CPD = int(os.environ.get("GRN_CPD", "0"))

assert D % VPT == 0 and VPT in (4, 8, 16), "VPT must divide 128 and be 4/8/16"
assert NT % (D // VPT) == 0, "NT must be a multiple of the threads-per-row"


def pick_rpt(rows: int, rstep: int) -> int:
    """Rows per thread. Shape-independent (see RPT_DEFAULT): one row per thread
    maximizes the CTA count, which is what makes the grid divide evenly over the
    148 SMs. Depends only on static metadata, never on input values."""
    if _RPT_ENV is not None:
        return int(_RPT_ENV)
    return RPT_DEFAULT


_LOG2E = 1.4426950408889634


def _selp_lt0(x, a, b):
    """Branchless `a if x < 0 else b` for Float32 (setp.lt + selp, 2 ops)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(x).ir_value(), cutlass.Float32(a).ir_value(),
         cutlass.Float32(b).ir_value()],
        "{ .reg .pred %p1; setp.lt.f32 %p1, $1, 0f00000000; "
        "selp.f32 $0, $2, $3, %p1; }",
        "=f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


def _tree_sum(vals):
    """Balanced binary tree over a Python list of Float32 SSA values.

    Lives at module level on purpose: inside a `@cute.kernel` body the AST
    preprocessor rewrites `for`/`while` into *dynamic* loops, which would make
    the shape-only index `n` an ArithValue. Here it is ordinary Python, so the
    tree is fully unrolled at trace time.
    """
    acc = list(vals)
    n = len(acc)
    while n > 1:
        n //= 2
        acc = [acc[j] + acc[j + n] for j in range(n)]
    return acc[0]


def _sigmoid_group(fz, m, vpt, sig):
    """Return the `vpt` gate values sigmoid(z) for one row, as a Python list of
    Float32 SSA values. `fz[m]` holds the row's bf16 gates.

    Everything here is about MUFU count, because MUFU is 1/8 of FP32 throughput
    and the gate is the only transcendental in the kernel.
    """
    if sig == 0:
        # sigmoid(z) = 1/(1+exp(-z)). rcp.approx.ftz.f32 is exact to 2^-23 and
        # 1+exp(-z) >= 1 so ftz is irrelevant; both tails are right
        # (rcp(inf)=0, rcp(1)=1). Two MUFU per element.
        out = []
        for j in range(vpt):
            zf = fz[m][j].to(cutlass.Float32)
            out.append(cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-zf, fastmath=True)))
        return out
    if sig == 1:
        # One rcp for the whole group instead of `vpt` of them (batch inversion:
        # 1 rcp + 3*(vpt-1) multiplies), which is exact to the same 2^-23.
        #
        # The naive product prod_j (1 + exp(-z_j)) would OVERFLOW to inf for a
        # row with a few strongly negative gates (exp(45)^2 > 3.4e38) and then
        # zero out the WHOLE group, including elements whose true gate is 1. So
        # use the sign-stable form: u = exp(-|z|) in (0,1], den = 1+u in (1,2],
        # so the product is bounded by 2^vpt and can never overflow, for any
        # input. num = u for z<0, else 1.
        den = []
        num = []
        for j in range(vpt):
            zf = fz[m][j].to(cutlass.Float32)
            # -|z|*log2(e), then a single ex2: u = exp(-|z|) in (0, 1]
            u = cute.math.exp2(
                cute.arch.fmin(zf, -zf) * cutlass.Float32(_LOG2E), fastmath=True)
            den.append(cutlass.Float32(1.0) + u)
            # z<0: sigmoid = u/(1+u);  z>=0: sigmoid = 1/(1+u).
            # (At z = 0 both give 0.5, so the boundary is exact either way.)
            num.append(_selp_lt0(zf, u, cutlass.Float32(1.0)))
        # batch inversion: one rcp, 3*(vpt-1) multiplies
        pre = [den[0]]
        for j in range(1, vpt):
            pre.append(pre[j - 1] * den[j])
        inv = cute.arch.rcp_approx(pre[vpt - 1])
        rec = [None] * vpt
        for jj in range(vpt - 1):
            j = vpt - 1 - jj
            rec[j] = inv * pre[j - 1]
            inv = inv * den[j]
        rec[0] = inv
        return [rec[j] * num[j] for j in range(vpt)]
    if sig == 2:
        # sigmoid(z) = 0.5*(1+tanh(z/2)): one MUFU (tanh.approx.f32) per element.
        out = []
        for j in range(vpt):
            zf = fz[m][j].to(cutlass.Float32)
            t = cute.math.tanh(zf * cutlass.Float32(0.5), fastmath=True)
            out.append(cutlass.Float32(0.5) + cutlass.Float32(0.5) * t)
        return out
    if sig == 8:
        # ABLATION (INCORRECT): drop the ex2, keep one rcp per element.
        out = []
        for j in range(vpt):
            zf = fz[m][j].to(cutlass.Float32)
            out.append(cute.arch.rcp_approx(cutlass.Float32(1.0) + zf * zf))
        return out
    # sig == 9: ABLATION (INCORRECT): no MUFU at all, but the z load and its
    # conversion stay live so the memory traffic is unchanged.
    out = []
    for j in range(vpt):
        zf = fz[m][j].to(cutlass.Float32)
        out.append(cutlass.Float32(1.0) + zf * cutlass.Float32(0.0))
    return out


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
def make_launcher(rows: int, nt: int = NT, vpt: int = VPT, rpt: int | None = None,
                  mincta: int = MINCTA, sig: int = SIG, abl: int = ABL,
                  cpd: int = CPD):
    """Build a `cute.jit` launcher specialized to a row count.

    `rows` = N*H. Everything below is a Python int at trace time, so all the
    index arithmetic collapses to masks and shifts -- no division in the kernel.
    """
    tpr = D // vpt                 # threads per row (16 at vpt=8)
    rstep = nt // tpr              # rows per CTA per step (8 at nt=128, vpt=8)
    if rpt is None:
        rpt = pick_rpt(rows, rstep)
    rows_per_cta = rstep * rpt
    grid = (rows + rows_per_cta - 1) // rows_per_cta
    exact = (rows % rows_per_cta) == 0
    nshfl = int(math.log2(tpr))    # butterfly steps: masks 1,2,...,tpr/2

    @cute.kernel
    def blackwell_bf16_gated_rmsnorm_kernel(
        mR: cute.Tensor,      # (rows, 128) bf16   pre-norm input
        mZ: cute.Tensor,      # (rows, 128) bf16   gate
        mO: cute.Tensor,      # (rows, 128) bf16   output
        mW: cute.Tensor,      # (128,)      fp32   RMSNorm weight
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        # tpr and rstep are compile-time powers of two, so these are an AND and
        # a SHR -- there is no integer division in this kernel.
        cg = tidx % tpr                      # column group: owns [cg*vpt, +vpt)
        rl = tidx // tpr                     # row slot inside the step
        row0 = bidx * rows_per_cta + rl

        # `w` is the only reused input (512 B for the whole grid, so it is an
        # L1/L2 hit). Load it ONCE, before the row loop, into vpt registers;
        # every row this thread owns then reuses it. That is the whole reason
        # rpt > 1 exists -- see design.md 3.
        fw = cute.make_fragment(vpt, cutlass.Float32)
        if cutlass.const_expr((abl & 1) == 0):
            cute.autovec_copy(cute.local_tile(mW, (vpt,), (cg,)), fw)
        else:
            for j in cutlass.range_constexpr(vpt):    # ABLATION: no w load
                fw[j] = cutlass.Float32(1.0)

        # --- issue every load first ---------------------------------------
        # rpt independent (r, z) pairs in flight before any arithmetic. bf16
        # stays packed, so this costs rpt*vpt/2 registers (32 at rpt=4) and
        # therefore no occupancy.
        fr = [cute.make_fragment(vpt, cutlass.BFloat16) for _ in range(rpt)]
        fz = [cute.make_fragment(vpt, cutlass.BFloat16) for _ in range(rpt)]
        for m in cutlass.range_constexpr(rpt):
            row = row0 + m * rstep
            if cutlass.const_expr(exact):
                cute.autovec_copy(
                    cute.local_tile(mR[row, None], (vpt,), (cg,)), fr[m])
                if cutlass.const_expr((abl & 4) == 0):
                    cute.autovec_copy(
                        cute.local_tile(mZ[row, None], (vpt,), (cg,)), fz[m])
            else:
                # Row-granular predication: a row is always exactly `tpr` whole
                # threads, so the tail never splits a 128-bit vector.
                if row < rows:
                    cute.autovec_copy(
                        cute.local_tile(mR[row, None], (vpt,), (cg,)), fr[m])
                    cute.autovec_copy(
                        cute.local_tile(mZ[row, None], (vpt,), (cg,)), fz[m])

        # --- reduce + scale + store, one row at a time ---------------------
        frf = cute.make_fragment(vpt, cutlass.Float32)
        fo = cute.make_fragment(vpt, cutlass.BFloat16)
        for m in cutlass.range_constexpr(rpt):
            row = row0 + m * rstep
            for j in cutlass.range_constexpr(vpt):
                frf[j] = fr[m][j].to(cutlass.Float32)
            if cutlass.const_expr(cpd in (0, 3)):
                ss = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(vpt):
                    ss = ss + frf[j] * frf[j]
            else:
                # Balanced tree: log2(vpt) = 3 dependent FADDs instead of vpt = 8
                # serial FFMAs. The vpt squares are mutually independent, so they
                # all issue back to back while the first pair is still in flight.
                ss = _tree_sum([frf[j] * frf[j] for j in range(vpt)])
            # XOR butterfly across this row's `tpr` lanes. With tpr <= 32 and
            # masks < tpr, the exchange never crosses a row boundary, so no SMEM
            # and no barrier are needed -- the rows in a warp are independent.
            if cutlass.const_expr((abl & 2) == 0):
                for e in cutlass.range_constexpr(nshfl):
                    ss = ss + cute.arch.shuffle_sync_bfly(ss, 1 << e)
                # mean(r^2) = sum * (1/128): a constant multiply, never a divide.
                sc = cute.math.rsqrt(
                    ss * cutlass.Float32(RMEAN) + cutlass.Float32(EPS),
                    fastmath=True)
            else:
                sc = cutlass.Float32(1.0)             # ABLATION: no reduction
            if cutlass.const_expr((abl & 4) == 0):
                sgv = _sigmoid_group(fz, m, vpt, sig)
            else:
                sgv = [cutlass.Float32(1.0)] * vpt    # ABLATION: no z
            if cutlass.const_expr(cpd in (0, 2)):
                for j in cutlass.range_constexpr(vpt):
                    fo[j] = (frf[j] * sc * fw[j] * sgv[j]).to(cutlass.BFloat16)
            else:
                # `pre` depends only on r, w and z -- NOT on the reduction -- so
                # these two multiplies per element overlap the shuffle butterfly,
                # and the only thing waiting on `sc` is one FMUL per element.
                pre = [frf[j] * fw[j] * sgv[j] for j in range(vpt)]
                for j in cutlass.range_constexpr(vpt):
                    fo[j] = (pre[j] * sc).to(cutlass.BFloat16)
            if cutlass.const_expr(exact):
                cute.autovec_copy(
                    fo, cute.local_tile(mO[row, None], (vpt,), (cg,)))
            else:
                if row < rows:
                    cute.autovec_copy(
                        fo, cute.local_tile(mO[row, None], (vpt,), (cg,)))

    @cute.jit
    def launch(pR: cute.Pointer, pZ: cute.Pointer, pO: cute.Pointer,
               pW: cute.Pointer, stream):
        lay = cute.make_layout((rows, D), stride=(D, 1))
        mR = cute.make_tensor(pR, lay)
        mZ = cute.make_tensor(pZ, lay)
        mO = cute.make_tensor(pO, lay)
        mW = cute.make_tensor(pW, cute.make_layout(D))
        blackwell_bf16_gated_rmsnorm_kernel(mR, mZ, mO, mW).launch(
            grid=[grid, 1, 1], block=[nt, 1, 1], stream=stream,
            min_blocks_per_mp=mincta,
        )

    launch._grn_cfg = (nt, vpt, rpt, rows_per_cta, grid, exact, sig, abl, cpd)
    return launch


# ---------------------------------------------------------------------------
# Empty-kernel node floor.
#
# At the spec shapes this kernel is latency-bound, and a large slice of every
# measurement is the DEVICE-side cost of a dependent kernel node -- a CUDA graph
# of N sequential kernel nodes serializes each node's launch and drain. Timing a
# kernel that does nothing, in exactly the same graph harness, says how much of
# the reported microseconds no kernel of any kind could avoid. Without this
# number a "% of HBM peak" figure at 12 MB is unreadable.
# ---------------------------------------------------------------------------
@cute.kernel
def _empty_kernel():
    tidx, _, _ = cute.arch.thread_idx()


@cute.jit
def _empty_launch(stream):
    _empty_kernel().launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)


_EMPTY = {}


def empty_node_ms(warmup, iters):
    """Per-node device time of a do-nothing kernel in the same graph harness."""
    dev = torch.cuda.current_device()
    rs = _raw_stream(dev)
    comp = _EMPTY.get((dev, rs))
    if comp is None:
        comp = cute.compile(_empty_launch, cuda.CUstream(rs))
        _EMPTY[(dev, rs)] = comp
    return _graph_ms(lambda: (lambda: comp(cuda.CUstream(_raw_stream(dev)))),
                     warmup, iters)


# ---------------------------------------------------------------------------
# Host driver: compile once per row count, launch with ~2 us of host work.
# ---------------------------------------------------------------------------
# `torch._C._cuda_getCurrentRawStream` is the cheap stream query (0.05 us);
# `torch.cuda.current_stream()` costs ~2 us because it builds a Stream object.
try:
    _raw_stream = torch._C._cuda_getCurrentRawStream        # type: ignore[attr-defined]
except AttributeError:                                      # pragma: no cover
    def _raw_stream(dev):
        return torch.cuda.current_stream(dev).cuda_stream

_PTR_DTYPES = (cutlass.BFloat16, cutlass.BFloat16, cutlass.BFloat16,
               cutlass.Float32)


class _Launch:
    """One compiled+bound launch site.

    Holds a pre-packed ctypes argument array for the compiled program's C entry
    point. A launch rewrites only the four pointer slots, so the per-call host
    cost is a handful of stores plus one C call -- everything shape-derived is
    cached, and nothing input-value-derived is.
    """

    __slots__ = ("comp", "ptrs", "descs", "stream", "adapted", "exe_args",
                 "packed", "capi", "cures", "raw_stream", "rows", "dev", "fast")

    def __init__(self, comp, ptrs, stream, adapted, exe_args, raw_stream, rows, dev):
        self.comp = comp
        self.ptrs = tuple(ptrs)
        self.descs = tuple(p._desc for p in ptrs)
        self.stream = stream            # keep alive: exe_args references it
        self.adapted = adapted          # keep alive: adapter-owned buffers
        self.exe_args = exe_args
        self.raw_stream = raw_stream
        self.rows, self.dev = rows, dev
        self.fast = False
        try:
            ex = comp.to(None)
            src = ex._get_invoke_packed_args(exe_args)
            packed = (ctypes.c_void_p * len(src))()
            for i in range(len(src)):
                packed[i] = src[i]
            self.packed = packed
            self.capi = ex.capi_func
            self.cures = ex.cuda_result
            self.comp = ex               # hold the executor alive
            self.fast = self.cures is not None
        except Exception:                # pragma: no cover
            self.packed = self.capi = self.cures = None

    def fire(self):
        if self.fast:
            self.capi(self.packed)
            err = self.cures.value
            if err:
                raise RuntimeError(f"CUDA error {err} launching gated rmsnorm")
        else:                            # pragma: no cover
            self.comp.run_compiled_program(self.exe_args)


class GatedRMSNorm:
    """Host driver. Compiles once per (rows, device); a call is launch-only."""

    def __init__(self):
        self._comp = {}      # (rows, dev) -> compiled program
        self._sites = {}     # (rows, dev, raw_stream) -> _Launch
        self._last = None    # single-entry fast lookup
        self._cfg = None

    def _compiled(self, rows, dev, args, stream):
        key = (rows, dev)
        comp = self._comp.get(key)
        if comp is None:
            launcher = make_launcher(rows)
            self._cfg = launcher._grn_cfg
            comp = cute.compile(launcher, *args, stream)
            self._comp[key] = comp
        return comp

    def _site(self, rows, dev, rs):
        site = self._sites.get((rows, dev, rs))
        if site is None:
            ptrs = [make_ptr(dt, 0, cute.AddressSpace.gmem, assumed_align=16)
                    for dt in _PTR_DTYPES]
            stream = cuda.CUstream(rs)
            comp = self._compiled(rows, dev, ptrs, stream)
            exe_args, adapted = comp.generate_execution_args(*ptrs, stream)
            site = _Launch(comp, ptrs, stream, adapted, exe_args, rs, rows, dev)
            self._sites[(rows, dev, rs)] = site
        self._last = site
        return site

    @staticmethod
    def _check(r, z, w):
        if r.shape[-1] != D:
            raise ValueError(f"gated_rmsnorm requires head dim {D}, got {r.shape[-1]}")
        if z.shape != r.shape:
            raise ValueError("z must have the same shape as r")
        if w.shape != (D,):
            raise ValueError(f"w must be [{D}]")
        if r.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
            raise ValueError("r and z must be bfloat16")
        if w.dtype != torch.float32:
            raise ValueError("w must be float32")
        if not (r.is_contiguous() and z.is_contiguous() and w.is_contiguous()):
            raise ValueError("gated_rmsnorm requires contiguous inputs")

    def prepare(self, r, z, w, out=None):
        """Return (fire, o): a closure that launches on these exact tensors with
        no Python argument work at all. Used for pure-kernel and CUDA-graph
        timing. The kernel still reads the LIVE contents of r/z on every launch;
        only the pointers are frozen."""
        self._check(r, z, w)
        rows = r.numel() // D
        o = torch.empty_like(r) if out is None else out
        dev = r.device.index
        site = self._site(rows, dev, _raw_stream(dev))
        for slot, t in zip(site.descs, (r, z, o, w)):
            slot.value = t.data_ptr()
        return site.fire, o

    def __call__(self, r, z, w):
        rows = r.numel() // D
        dev = r.device.index
        rs = _raw_stream(dev)
        site = self._last
        if (site is None or site.rows != rows or site.dev != dev
                or site.raw_stream != rs):
            self._check(r, z, w)
            site = self._site(rows, dev, rs)
        o = torch.empty_like(r)
        d = site.descs
        d[0].value = r.data_ptr()
        d[1].value = z.data_ptr()
        d[2].value = o.data_ptr()
        d[3].value = w.data_ptr()
        site.fire()
        return o


_ENGINE = GatedRMSNorm()


# ---------------------------------------------------------------------------
# MANDATORY importable API
# ---------------------------------------------------------------------------
def gated_rmsnorm(r, z, w):
    """Gated RMSNorm over the last (size-128) dim.

    r, z : [..., 128] bf16 contiguous     w : [128] fp32
    -> o : [..., 128] bf16, freshly allocated

    o = r * rsqrt(mean(r^2, dim=-1) + 1e-5) * w * sigmoid(z)
    """
    assert r.shape[-1] == 128 and w.numel() == 128, (
        f"this CuTeDSL kernel is compiled for a size-128 last dim, got {r.shape[-1]}"
    )
    return _ENGINE(r, z, w)
