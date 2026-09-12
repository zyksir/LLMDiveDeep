# Mirage (MPK) megakernel study — B200 feasibility for mega-MoE / mega-KDA

Date: 2026-08-19
Qualification: build/run findings QUALIFIED (executed on B200); latency
numbers below are Mirage's own test shapes (GLM-4.6-family), anchors only —
no Kimi-K3-shape benchmark yet.
Source: github.com/mirage-project/mirage (cloned @ HEAD, submodules), built
from source for sm_100 only, venv at /workspace/mirage-venv in the
llmdd-route-bench container.

## What Mirage is, in one paragraph

Two subsystems. (1) A superoptimizer/transpiler (`mi.new_kernel_graph` +
threadblock graphs, z3-verified) that generates single fused kernels.
(2) **MPK — the Mirage Persistent Kernel**: one persistent worker+scheduler
kernel per decode step; every layer op (rmsnorm, linear, router, per-expert
GEMV, attention, sconv...) is a TASK (a templated `__device__` function in
`include/mirage/persistent_kernel/tasks/<arch>/*.cuh`) scheduled with
intra-kernel dependencies — zero kernel launches inside a step. This is the
strongest form of the megakernel idea, and (2) is the part relevant to us.

## What runs on B200 (all executed today)

| test | result | note |
|---|---|---|
| MPK persistent kernel launch (worker+scheduler, 205 KB smem) | **PASS** | glm_moe_router testmode |
| `glm_moe_router` task (sigmoid+bias top-k, DSv3 family = Kimi-K3's math) | **PASS**, weights err 3e-8 | 160 experts/top-8 shape |
| `moe gate_topk / w13_linear / silu_mul / w2_linear / weighted_sum` tasks | **PASS**: 5.1 / 22.6 / 9.6 / 19.6 / 12.3 us | standalone launches; MPK removes the gaps |
| `fp8_moe_gemm` (grouped FP8 expert GEMM) | **PASS** | FP8, not MXFP4 |
| transpiler demo (`demo_blackwell/rmsnorm.py`) | **SEGFAULT** (blocker, root-caused below) | MPK path unaffected |

Build path (PyPI is blocked from the containers): z3 built from the
submodule (ninja, ~2 min at 256 cores) and shimmed into the venv in
z3-solver wheel layout + `libz3.so.4.13` SONAME symlink +
`LD_LIBRARY_PATH`; `pip install -e . --no-build-isolation --no-deps` with
`CMAKE_CUDA_ARCHITECTURES=100` (88 s). Rust components (abstract_subexpr,
formal_verifier) built by cargo despite flaky egress.

**Transpiler segfault root cause (rule 19):** crash inside
`mirage::threadblock::Graph::matmul` during graph construction. Their own
pyproject pins `z3-solver==4.16.0.0` and warns the extension is linked
against that exact libz3 SONAME; the submodule provides 4.13.1, and with
PyPI blocked the 4.16 wheel cannot be installed — a 4.13-vs-4.16 z3 C++ API
behavior difference in the abstract-expression layer is the remaining
suspect after eliminating unresolved-symbol causes (fixed the missing
SONAME + Rust libs first; crash site unchanged). Fix path: obtain the
z3-solver 4.16 wheel (mirror/offline) and rebuild. MPK does not use this
layer at runtime, so the megakernel work is unblocked.

## Mega-MoE assessment (Kimi-K3 on MPK)

- The GLM-4.6 builder already assembles a full MoE megakernel: router task
  -> per-EXPERT w13 GEMV task -> silu -> per-expert w2 GEMV -> weighted-sum
  epilogue, one task-graph node per expert (grid=(NUM_EXPERTS,1,1)) — this
  is precisely the b10 expert-GEMV design our expert-region report marked
  PROCEED-worthy, already implemented as schedulable tasks.
- Router: same sigmoid+bias top-k family, BUT the task's algorithm is one
  THREAD per token with two float[NUM_ROUTED] local arrays and a serial
  TOPK x NUM_ROUTED argmax loop — fine at GLM's 160/8 (5.1 us measured),
  rejected by inspection at Kimi-K3's 896/16 (7 KB/thread local-memory
  spill + 14k serial compares/token; our routing study measured even the
  16-round-reduction trt_noaux at 9-10 us on this shape, and sgl_radix's
  byte-counting at 2.1 us). A K3 MPK port therefore needs a new
  radix-counting router task (sgl_radix's algorithm reshaped to one
  128-thread task block per token) — well-scoped, same registration path
  as the KDA task.
- Quantization: **bf16 + FP8 exist (fp8 linears, fp8 grouped MoE GEMM);
  MXFP4 does not** — as the user expected. K3's W4A8 MXFP4 weights would
  need either a new fp4 task (large) or an fp8-weights variant for the
  study (2x weight bytes vs production: honest caveat in any comparison).
- Decode-only, world_size 1 in current builders (inkling notes both); TP8
  needs their allreduce task (exists: `allreduce.cuh`) plus sharded
  builders — the mla_prefill_tp8 test shows TP is landing.

## Mega-KDA assessment

- Inkling's `sconv` task IS KDA's conv stage (depthwise k=4, fp32 state
  updated in place, per-channel task partitioning) — template ready.
- Missing piece: ONE new task for the gated-delta recurrence
  (state [12,128,128] fp32 in place). Registration surface is small and
  mapped: TaskType enum (`runtime_header.h`) + `task_register.cc` + a
  `persistent_kernel.py` wrapper + the task `.cuh` (~250 lines; port of our
  CuTeDSL decode kernel's math, simpler here because MPK grants a whole
  task per (head, token) with runtime-applied pointer offsets).
- The full mega-KDA layer then composes existing tasks: linear (in_proj) ->
  sconv -> linear (f_b) -> NEW kda_recurrence -> linear (o_proj), with MPK
  dependencies instead of kernel launches — the fusion ladder's step-3
  goal without hand-writing one monolithic kernel, and without the
  critical-path mistake that sank our FUSE_FB attempt (MPK schedules the
  f_b GEMV as a task that overlaps other tasks, rather than inlining it
  into a prologue).

## Verdict and next steps

The idea works on B200 today at the task level; the leverage for us is MPK
as a **scheduling substrate** for the decode megakernel (KDA layer bound
~24-25 us, currently 36; MoE expert region bound ~14-48 us, currently
22-92). Ranked next steps:
1. ~~Kimi-K3-shape router instantiation~~ — answered by inspection:
   direct instantiation rejected (thread-per-token argmax does not scale
   to 896/16); replace with a radix-counting router task port.
2. KDA recurrence task (~250-line .cuh + 3 registration touchpoints) and a
   single-layer mega-KDA task graph vs our b10_kda 36 us @B=1.
3. FP8-weights K3 expert pipeline on MPK vs trt_run_moe (bf16/fp8 caveat).
4. Blocked/deferred: MXFP4 tasks; the transpiler path (z3 4.16 wheel).

## Reproduce (container llmdd-route-bench; each step < 5 min warm)

```bash
source /workspace/mirage-venv/bin/activate && cd /workspace/mirage
python tests/runtime_python/blackwell/sm100_glm4_moe/test_glm_moe_router_testmode.py  # ~30 s
cd tests/runtime_python/blackwell/sm100_moe && python setup.py build_ext --inplace  # 41 s once
python test_gate_topk.py && python test_w13_linear.py && python test_silu_mul.py \
    && python test_w2_linear.py && python test_weighted_sum.py               # ~10 s
```
