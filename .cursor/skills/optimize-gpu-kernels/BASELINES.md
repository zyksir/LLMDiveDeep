# Open-source baseline policy

Last updated: 2026-08-18.

## Standalone-first policy

The first benchmark for a kernel task must invoke only one target kernel and
its unchanged open-source/native baseline. Decompose layers and models into
kernel-sized subtasks and solve correctness, latency, profiling, and
optimization for each target before integration.

Default to one GPU, one process, preallocated direct inputs/outputs, and no
model weights, MoE expert forward, MPI/NCCL startup, or unrelated kernels. If a
library exposes the target only through a layer API, add a thin standalone
launcher for the exact internal kernel. A full layer is not an acceptable
substitute for a standalone routing, permutation, activation, or GEMM
benchmark.

Use multiple GPUs only if the target kernel is intrinsically distributed, and
use the smallest topology that exercises its contract. Run containing-pipeline,
layer, and end-to-end tests only after the standalone kernel reaches its
acceptance or stop criterion. Those tests validate integration; they are not
the per-edit development loop.

Before building the harness, define the kernel's model/pipeline meaning, direct
inputs and outputs, exact semantics, side effects, consumers, and timed
boundary. If the name covers a pipeline such as gate projection, route
selection, and permutation, split it into separate standalone contracts first.

## Qualification

A performance statement is qualified only when all answers are yes:

- Were current SGLang, TensorRT-LLM, vLLM, FlashInfer, and model-author paths
  checked and traced to their actual executed kernels or marked not applicable
  with evidence?
- Were all applicable same-contract implementations included in the
  method-by-shape matrix, with concrete `N/A` reasons for any that could not run?
- Is at least one directly executed baseline an unchanged open-source/native
  GPU kernel?
- Is it independent of the candidate rather than another local adaptation?
- Does it implement the same model-specific math and observable output ABI?
- Is it the strongest relevant implementation found after searching current
  upstream code and results?
- Were commit, file, license, hardware, software, and launch mode recorded?
- Were candidate and baseline timed with identical inputs, output obligations,
  graph/eager mode, warmup, and timing method?
- For a decomposed operation, was the compatible cross-product of strongest
  stage implementations executed, including mixed upstream/local compositions?
- Did candidate outputs pass direct comparison with the open-source kernel on
  normal and adversarial inputs?
- Is there a concrete, falsifiable reason for the speedup, mapped to candidate
  code and supported by an ablation or measured mechanism?
- Was a documented SOL estimate and stop/proceed decision made before candidate
  implementation?

If any answer is no, label the result `UNQUALIFIED` and state the missing gate.
Lower latency without causal attribution is not enough to qualify an
optimization.

## Baseline roles

Use separate baselines when needed:

- **Correctness baseline:** unchanged open-source/native kernel implementing the
  same contract. Compare actual outputs directly.
- **Performance baseline:** fastest relevant unchanged open-source/native kernel
  for the same contract and target workload.
- **Migration baseline:** current LLMDiveDeep implementation. This only
  estimates integration impact and cannot support an optimization claim.
- **Supplemental oracle:** PyTorch or a simple CPU/GPU reference. This catches
  shared bugs but cannot replace the correctness baseline.

## What does not qualify

- another kernel authored in LLMDiveDeep;
- a candidate copied and modified from upstream;
- a benchmark number quoted from a README without running its kernel;
- PyTorch, unless the task is specifically optimizing a PyTorch implementation;
- a DeepSeek-V3/DeepV3 router used as a Kimi-K3 baseline without evidence that
  math, expert count, top-k, ABI, and workload match and that it is strongest;
- a baseline that writes fewer outputs or performs less accurate math;
- comparing eager candidate timing with graph baseline timing or vice versa.

## Adapters and forks

A baseline starts as a pinned copy of upstream source files. Keep those files
unchanged and record their commit and hashes. Do not reimplement the baseline,
copy its algorithm into a new kernel, redeclare private ABI structs, alter its
template dispatch, or specialize its launch path and still call it a baseline.
Use the upstream package/build system whenever available.

A thin adapter may reshape inputs or normalize outputs only when:

- the underlying kernel body is byte-for-byte unchanged;
- adapter work is outside the timed region or equally charged to both sides;
- the report names the adapter and its exclusions.

Keep the adapter in a separate file from the copied baseline. If the unchanged
source cannot be built or invoked without reconstructing private ABI, mark the
baseline unavailable and report the blocker; do not manufacture a substitute
baseline.

Once the kernel body, launch decomposition, math, or output writes change, call
it a candidate. Keep an untouched upstream copy for the baseline.

### Backend-module boundary

Baseline and candidate adapters implement the benchmark runner's common
backend API; they do not add backend-specific branches to the runner. A backend
module must declare provenance, supported contracts and layouts, setup, the
operation callable, outputs, and charged compatibility work. Unsupported cases
return a structured capability reason rather than requiring runner edits.

Adding a baseline should therefore be additive: copy the pinned upstream source
unchanged, add one separate adapter/backend module, register it, and run the
existing pipeline. If that requires changing common correctness or timing code,
stop and determine whether the operation contract truly changed or the backend
interface is incomplete.

## Strongest composition rule

For a pipeline decomposed into stages, the performance target is the fastest
compatible composition, not the all-local-candidate pipeline. After standalone
stage measurements:

- retain serious unchanged and candidate winners for every stage;
- execute the ABI-compatible cross-product across adjacent stages;
- include casts, repacks, adapters, intermediate writes, and synchronization in
  the timed boundary;
- compare mixed compositions with unchanged fused and unfused pipelines;
- choose by measured pipeline latency regardless of code ownership.

Do not infer the winner by adding isolated latency numbers; interactions,
launch overlap, PDL, cache state, and conversion costs require direct
measurement. A final report that omits a plausible stronger mixed composition
is `UNQUALIFIED`.

## Same-contract rule

Performance comparison requires equal obligations. Match:

- all output tensors and initialization guarantees;
- precision and exact/approximate math;
- tie ordering and deterministic requirements;
- strides, alignment, batching, rank partition, and padding;
- allocations, synchronization, PDL, and CUDA graph behavior.

If upstream lacks a required output, either add an untimed postprocess to the
baseline and charge it, or benchmark a candidate mode with the common ABI.
Disclose the choice.

## Fast-loop rule

Benchmark harness latency is part of the optimization system. Record:

- cold setup time, including baseline compilation;
- cached quick-loop wall time;
- full correctness/benchmark wall time.

The cached quick loop should execute unchanged open-source and candidate
kernels in one process, reuse compiled artifacts and allocations, and test a
small representative shape set. Diagnose the harness before kernel tuning when
imports, compilation, synchronization, model loading, distributed startup, or
excessive repetitions dominate feedback time.

Never report quick-loop numbers as the final matrix. Run full shapes,
adversarial correctness, containing-layer timing, and independent baselines
before qualification or shipping.

Use deterministic synthetic device tensors for standalone work when they can
faithfully reproduce shape, dtype, stride, alignment, and value distribution.
Do not load a model merely to produce random kernel inputs. Keep one process
alive, preallocate all buffers, compile/cache unchanged baselines once, and
optimize the harness before tuning kernels when feedback is slow.

## SOL and pre-candidate stop rule

Estimate a lower-bound latency for every required shape before writing a
candidate. Take the maximum applicable bound from launch/dispatch floor,
mandatory traffic over attainable bandwidth, required arithmetic over
attainable throughput, and irreducible serial reduction or synchronization.
State assumptions and uncertainty; nominal FLOPs alone are insufficient for
top-k, scan, permutation, or other irregular kernels.

Every SOL term must be measured or derived, never asserted:

- **Launch floor:** time an empty kernel in the same graph/eager mode.
- **Bandwidth term:** time a streaming probe kernel over the same tensor, not
  a nominal-peak fraction.
- **Dependency term (mandatory for selection/scan/permutation and any
  small-batch kernel):** bound the serial critical path explicitly. Two
  measured components dominate:
  1. *Per-warp instruction issue.* A kernel whose shape leaves few active
     warps is issue-limited: estimate (serial instructions per warp) /
     (~1 instruction/cycle/warp). Count per-lane work items times per-item
     instruction cost (an accurate `expf` plus rounded division is ~100+
     instructions, not 2 flops). This is why K-items-per-lane designs lose to
     designs that spread one token across many threads.
  2. *Synchronization-phase chain.* Count dependent `__syncthreads`/cluster/
     grid barriers and multiply by a measured phase cost from an N-phase probe
     kernel.

A latency-versus-floor ratio computed without the dependency term is not
"headroom"; label such a bound as loose and do not use it to justify a
candidate. A candidate is justified only when the gap is attributed to
removable work (a launch, a byte stream, an idle-parallelism deficit, an
oversized serial phase) that the candidate concretely eliminates, and the
report's ablation must confirm that mechanism.

Bound and measurement sanity checks: per-shape latency that is non-monotonic
in the shape indicates an internal dispatch switch — profile the executed
kernel names per shape (thresholds are often tuned for another model's top-k
or expert count) before trusting any single-shape conclusion. Per-operation
bounds are not interchangeable: never compare one stage's latency against
another stage's bound in a shared table; label every SOL column with its
operation. Run-to-run clock drift on idle GPUs is ~±0.2 us at decode sizes:
accept or revert a sub-microsecond variant only from a same-process A/B in
one run, never by comparing numbers across separate invocations.

If the best unchanged implementation is already close to this bound and no
profiled launch, byte, instruction, barrier, occupancy, or dependency
opportunity is material, stop and report that result. Do not manufacture a
candidate.

## Report template

Every attempt follows [REPORTING.md](REPORTING.md). The compact structure is:

```
Date: YYYY-MM-DD
Qualification: QUALIFIED | UNQUALIFIED | REJECTED

Operation contract:
- meaning / inputs / outputs / exact semantics / timed boundary
- decomposition when the named operation contains multiple kernels

Pre-candidate benchmark:
- all applicable unchanged methods across all required shapes
- SOL model and lower bound
- STOP or PROCEED decision

Idea:
- measured bottleneck, optimization mechanism, and unchanged semantics

Open-source correctness baseline:
- repository / commit / file / license
- unchanged kernel body: yes/no
- direct output comparison: pass/fail/not run

Open-source performance baseline:
- repository / commit / file
- same contract and launch mode: yes/no

Results:
- generated method-by-shape table with absolute latency columns for every
  unchanged implementation and every candidate variant
- fastest unchanged baseline per shape, candidate speedup, SOL bound, and
  achieved SOL/headroom

Secondary migration data:
- current LLMDiveDeep latency

Missing gates or limitations:
- ...

Alternatives and decision:
- variants and variables explored
- evidence for selection, rejection, or deferral
- chosen variant and measured reason

Dev-loop cost:
- cold setup seconds
- cached quick-loop seconds
- full validation seconds

Integration:
- containing pipeline / layer / end-to-end after standalone acceptance
```
