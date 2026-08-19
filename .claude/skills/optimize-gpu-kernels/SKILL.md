---
name: optimize-gpu-kernels
description: Optimizes LLMDiveDeep GPU kernels with correctness and performance gates against unchanged, strongest relevant open-source or native kernels. Use for CUDA, CuTe DSL, Triton, fusion, and megakernel work.
---
# Optimize GPU kernels

Last updated: 2026-08-18.

Read [BASELINES.md](BASELINES.md) and
[REPORTING.md](REPORTING.md) before implementing or reporting results.

## Mandatory task-confirmation gate

Before editing code, compiling, or running a benchmark, explicitly ask the user
to confirm the task boundary, even when the request appears clear. Restate:

- included kernels or stages and whether their composition/fusion is included;
- excluded adjacent kernels, containing layers, deployment, and integration;
- input/output dtypes, layouts, shapes, math, and observable outputs;
- required baselines, deliverables, and stopping point.

Do not infer this boundary from a model name, directory name, downstream ABI,
earlier broader request, or likely integration path. Start only after the user
confirms it, and treat later corrections as a new frozen boundary.

## Required five-stage process

Do not skip ahead to candidate code.

1. **Define the operation.** State what the kernel means in the model or
   pipeline, its direct inputs and outputs, exact math, side effects, precision,
   tie/order behavior, strides, supported shapes, and consumers. Draw the
   operator boundary: identify which surrounding work is outside this kernel.
2. **Survey implementations.** Trace every major current open-source and native
   implementation to the symbol that executes for the target model, shape,
   dtype, hardware, and backend. Record algorithms, ABIs, fusion boundaries,
   and provenance before selecting baselines.
3. **Benchmark before writing.** Build one fast standalone harness that executes
   every applicable same-contract implementation unchanged, checks outputs,
   measures latency across the full shape matrix, and estimates a defensible
   speed-of-light bound. Stop if the best implementation is already near that
   bound and no material, falsifiable improvement mechanism remains.
4. **Design and implement a candidate.** Derive concrete ideas from measured
   bottlenecks and the surveyed algorithms, or state the new algorithmic idea.
   Only then write candidates, run controlled variants, and use the
   CuTeDSLGen workflow when CuTe DSL is appropriate.
5. **Report the standalone kernel.** Produce the required method-by-shape
   result tables, correctness and SOL evidence, idea attribution,
   alternatives, and explicit qualification. Integration is a separate task
   and starts only when the user explicitly requests it.

## Benchmark architecture gate

The standalone benchmark must be a stable, backend-driven pipeline. Its central
runner generates common inputs, invokes an operation API, checks common
outputs, times that API, stores structured results, and renders tables. It must
not encode each implementation as new backend-specific control flow.

Put every implementation in a separate backend module with metadata for
provenance, capabilities, layout/dtype/shape support, output ABI, setup, the
callable operation, and explicitly charged conversions. Register or select
backends through data/configuration. Adding a new backend should require no
change to the shape loop, correctness logic, timing helper, composition loop,
or result serializer.

Treat the runner as frozen after its contract is validated. Edit it only to
change the shared operation contract, measurement method, structured result
schema/output table, or to fix a runner bug. Backend discovery, build issues,
private APIs, unsupported strides, output normalization, and backend-specific
work belong in that backend's module.

During a focused backend investigation, filter the stable runner to that
backend and the minimum reference methods and shapes needed for correctness.
Do not repeatedly rebuild or rerun already-settled backends. Rerun the complete
matrix only when shared benchmark semantics changed or at a final reporting
checkpoint.

## Highest-priority decomposition rule

Split a large kernel, layer, MoE, fusion, or end-to-end objective into
kernel-sized subtasks. Before layer initialization or end-to-end timing, create
a standalone harness for exactly one target kernel and its strongest unchanged
open-source/native baseline.

The default standalone harness uses one GPU, one process, preallocated direct
inputs/outputs, and no model weights, experts, MPI, NCCL, or unrelated kernels.
If the target is hidden behind a library or layer API, write a thin launcher for
the exact internal kernel. Do not launch a full MoE layer merely to reach a
local routing, permutation, activation, or GEMM kernel.

Work sequentially through:

1. one-kernel contract and baseline;
2. one-kernel correctness;
3. one-kernel latency and profile;
4. isolated optimization and stop decision;
5. standalone kernel report and handoff;
6. smallest containing pipeline, only after explicit user authorization;
7. layer and end-to-end integration, only after explicit user authorization.

Use multiple GPUs only when the target kernel is intrinsically distributed,
and then use the minimum rank count that exercises its contract. Serialize GPU
experiments; decomposition into subtasks does not authorize concurrent
benchmarks that cause contention.

Containing-layer and end-to-end runs are integration gates after standalone
success. They are not the normal edit loop.

### Explicit scope and phase gate

A request to optimize, benchmark, or report one kernel authorizes only that
standalone kernel. Split larger goals into small sequential tasks and finish
the current kernel's contract, correctness, performance, and report before
proposing the next task.

Do not begin, modify, benchmark, or optimize a containing pipeline, adjacent
kernel, layer, model, deployment path, or end-to-end integration unless the
user explicitly requests that phase. Do not change the standalone kernel's
precision or output contract based on an assumed downstream consumer. Record
possible integration work as out of scope rather than implementing it.

### Strongest-composition gate

Apply this gate only when the explicitly requested task is a multi-stage
pipeline or composition. It does not expand a standalone-kernel task into an
integration task.

Decomposition is not complete after choosing a winner independently for each
stage. Before calling a multi-stage result final, enumerate and benchmark the
compatible cross-product of serious stage implementations. A pipeline named
"ours" must not be fixed to locally authored kernels when an unchanged
open-source stage is faster.

At minimum:

1. retain the fastest unchanged implementation and fastest candidate for each
   stage;
2. compose every ABI-compatible combination across adjacent stages;
3. include required casts, repacks, adapters, synchronizations, and intermediate
   materialization in the timed region;
4. compare the compositions with the strongest unchanged fused and unfused
   pipelines;
5. select by measured end-to-end boundary latency, not implementation
   authorship.

Prune a combination only when it is provably dominated or ABI-incompatible,
and record that reason. If the strongest compatible composition has not been
measured, the containing-pipeline conclusion and final report are
`UNQUALIFIED`.

### Composite-front example: MoE routing

“Routing” may name a pipeline rather than one kernel. For Kimi-K3, first split
and contract at least:

1. **gate projection:** hidden states and gate weights to routing logits;
2. **route selection:** logits and correction bias to selected expert IDs and
   weights, including sigmoid, top-k, and renormalization;
3. **permutation/dispatch metadata:** selected IDs and weights to expert-major
   mappings, padding, counts, and grouped-GEMM descriptors.

Give each stage its own unchanged baselines, synthetic-input standalone
harness, correctness checks, latency matrix, SOL estimate, and stop decision.
Only after the user explicitly requests the multi-stage phase, benchmark the
route-by-permutation cross-product, including current SGLang route plus a
candidate permutation when that is the strongest measured stage combination.
Likewise, gate+route, route+permute, the full front, and fusion are separate
integration tasks. Do not call a route-only candidate faster than a fused
routing front; report them as different scopes.

## Non-negotiable gate

Every kernel written by this repository needs both:

1. an independent unchanged open-source/native kernel for correctness; and
2. the strongest relevant unchanged open-source/native kernel for performance.

An LLMDiveDeep kernel, another kernel produced in this project, or a
DeepSeek-V3/DeepV3 kernel does not satisfy either gate for Kimi-K3 unless
evidence first establishes that it implements the same Kimi-K3 contract and is
the strongest available Kimi-K3 implementation.

PyTorch may be a supplemental mathematical oracle. It does not replace the
open-source correctness kernel and is never a performance baseline.

If either gate is missing, mark the result `UNQUALIFIED`. Do not claim that the
candidate is faster, better, near SOL, ready to ship, or an optimization.

## Ecosystem implementation survey

Before choosing a baseline or writing a candidate, inspect all major current
open-source implementations relevant to the model and operation. At minimum
check SGLang, TensorRT-LLM, vLLM, FlashInfer, and the model author's repository
when available; add other active libraries found during search.

For each implementation, record:

- repository, current commit/tag/version, file, symbol, and license;
- whether the path is actually selected for the target model, shape, dtype,
  hardware, and serving mode;
- algorithm, launch topology, fusion boundary, input/output ABI, and precision;
- standalone versus production path and any backend dependency;
- correctness compatibility and why it is or is not a valid performance
  baseline;
- measured latency when it can be executed unchanged.

Do not assume that frameworks with the same routing math share a kernel. Trace
Python/model dispatch through backend selection to the actual CUDA, Triton,
CuTe DSL, or external-library symbol. Distinguish bundled code from FlashInfer,
DeepGEMM, CUTLASS, or another dependency.

Create an ecosystem comparison section in the report before implementation.
If SGLang, TensorRT-LLM, vLLM, or another major current implementation has not
been checked, the baseline search is incomplete and the optimization remains
`UNQUALIFIED`.

Every applicable same-contract implementation found by this survey must appear
in the standalone benchmark matrix. If one cannot be executed, record `N/A`,
the concrete blocker, and whether that prevents qualification. At minimum the
strongest implementation must execute unchanged for performance and one
independent implementation must execute unchanged for correctness.

## Baseline benchmark and speed of light

Use directly generated synthetic tensors for standalone work. Prefer
deterministically seeded device tensors with the real shapes, dtypes, strides,
alignment, value distribution, and changing-input behavior. Model loading is
unnecessary unless tensor values or persistent model state are part of the
kernel contract. Add boundary and adversarial inputs for ties, NaNs/Infs when
defined, skew, empty/local-expert extremes, padding, and graph replay.

For every required shape:

- execute all applicable unchanged implementations in one process when
  compatible;
- preallocate inputs, outputs, and workspaces outside timing;
- use identical inputs, warmup, graph/eager mode, synchronization, repetitions,
  and output obligations;
- report absolute latency for every method, not only relative speedup;
- identify the fastest unchanged method per shape rather than assuming one
  baseline wins every regime.

Estimate SOL before candidate implementation. State the model and assumptions.
Use the maximum of applicable lower bounds: measured launch/dispatch floor,
mandatory bytes divided by attainable bandwidth, required arithmetic divided
by attainable throughput, and irreducible serial reduction/synchronization
depth. For irregular selection and scan kernels, nominal FLOPs alone are not a
credible SOL model; include instruction, dependency, barrier, occupancy, and
launch evidence. Report both the lower-bound latency and achieved fraction of
that bound, with uncertainty when the bound is approximate.

If all strongest implementations are close to the defensible bound and
profiling reveals no material removable work, write a qualified stop report and
do not create a candidate merely to complete the workflow.

## Causal attribution and anti-cheating gate

State a falsifiable reason the candidate should beat the baseline before
accepting its result. Identify the exact eliminated work or shortened critical
path: launches, bytes, barriers, instructions, synchronization, spills,
occupancy loss, redundant computation, or algorithmic complexity.

Map that reason to the candidate's code difference and verify equal work. When
an idea has multiple changes, add controlled ablations that enable one material
change at a time. Profile or count the mechanism when practical; latency alone
does not prove causality.

Reject hidden advantages such as omitted outputs, stale buffers, cheaper math,
different precision, favorable inputs, untimed conversion, different graph
mode, or different synchronization. Preserve the unchanged baseline source
hash. If a measured speedup has no clear mechanism or attribution evidence,
mark the optimization `UNQUALIFIED` rather than inventing an explanation.

## Workflow

1. Freeze the exact contract: shapes, strides, dtypes, math, tie behavior,
   output ABI, graph/eager mode, and target hardware.
2. Complete the ecosystem survey and trace each framework to its executed
   kernel symbol.
3. Search additional current open-source/native implementations before writing
   code.
4. Record repository, commit, file, license, and why each implementation is or
   is not an applicable baseline.
5. Build the synthetic-input standalone one-kernel harness on the minimum
   required GPU count.
6. Execute every applicable unchanged baseline for correctness and performance.
   A thin input/output adapter is allowed outside the timed region; changing
   the kernel body creates a candidate, not a baseline.
7. Estimate SOL, profile the fastest methods, and make the pre-candidate
   stop/proceed decision.
8. If proceeding, write the falsifiable optimization hypothesis, equal-work
   check, candidate variants, and expected shape regimes.
9. Implement candidates and compare their outputs directly with the
   open-source correctness kernel on random, adversarial, boundary, and
   changing-input CUDA-graph cases.
10. Benchmark candidates and all applicable open-source/native methods with
    identical inputs, output obligations, warmup, graph mode, clocks, and
    timing method.
11. Run controlled ablations and verify the expected mechanism changed.
12. Treat local/current LLMDiveDeep results only as secondary migration data.
13. Run the quick baseline-gated loop after every material change and the full
    matrix at checkpoints and before reporting.
14. For decomposed operations, benchmark the compatible cross-product of stage
    winners and select the strongest measured composition.
15. Integrate into the containing pipeline and run layer/end-to-end validation
    only after the isolated kernel reaches its acceptance or stop criterion.

## Dev-loop speed is infrastructure

Measure benchmark wall time before tuning the kernel. If one correctness plus
performance iteration is slow enough to limit experimentation, stop and
optimize the harness first.

Provide three tiers:

- **smoke:** compile/launch and one open-source output comparison;
- **quick:** standalone single-kernel representative boundary shapes, unchanged
  open-source correctness, short graph timing, and cached artifacts;
- **full:** every required shape, adversarial cases, independent native
  correctness, and standalone profiles; containing-layer and end-to-end timing
  are subsequent integration gates.

Quick mode must retain an executed open-source correctness and performance
baseline. It may reduce shapes, warmup, iterations, and repeats; it may not
replace open source with a local kernel or PyTorch.

Investigate wall time by separating:

- container/process startup and imports;
- JIT/AOT compilation and extension loading;
- model/weight initialization and distributed startup;
- input allocation and host/device synchronization;
- correctness normalization;
- graph capture, warmup, iterations, and repeats.

Cache unchanged baselines by source hash, prebuild them during environment
setup, reuse allocations and graph objects, vectorize correctness checks, and
avoid redundant nested repetition. Prefer a persistent benchmark process when
startup dominates. Serialize GPU benchmark/compile jobs when concurrent runs
would contend for memory or invalidate timing. Print quick-loop wall time and
keep it in the report.

Do not optimize measurement cost by weakening semantic checks, changing the
timed contract, or timing candidate and baseline differently.

## Candidate design and CuTeDSLGen

Before coding, write an idea ledger: for each surveyed implementation list the
useful algorithmic ideas, measured strengths, measured bottlenecks, and contract
tradeoffs. Then state whether the candidate combines known ideas, changes the
launch/dataflow, or introduces a new algorithm. An unexplained rewrite is not
an optimization idea.

For CuTe DSL candidates, read and follow
`/root/CuTeDSLGen/.claude/skills/kernel-cute-writing/SKILL.md` and the relevant
CuTeDSLGen references/scripts. Start from the closest architecture-matched
example, precompile with `cute.compile`, keep allocation and compilation out of
the timed wrapper, inspect generated MLIR/PTX when attribution needs it, and
run the LLMDiveDeep open-source baseline gates in addition to CuTeDSLGen's
verification and benchmark scripts. Generated code is a candidate, never a
correctness or performance baseline.

Different candidates for different shape regimes are allowed when measurements
justify them. Report each variant, dispatch boundary, compile/cache cost, and
dispatcher overhead; benchmark around every boundary. Keep the variant count
minimal and choose by input metadata only—never by benchmark identity or
expected test data.

## Semantic safety

Do not silently change precision, sigmoid/exp/reciprocal accuracy, reduction
order requirements, tie behavior, quantization, sparsity, padding, or output
ABI. Propose semantic changes separately and implement them only with explicit
user approval and an agreed error budget.

## Reporting

Every optimization attempt must produce the report defined in
[REPORTING.md](REPORTING.md), including slower, failed, rejected, and
unqualified attempts. A kernel task is not complete with code and raw benchmark
output alone.

Lead with a two-to-four-sentence optimization idea, then the unchanged
open-source comparison. Report:

- the bottleneck, mechanism, and unchanged semantic contract;
- exact baseline provenance and whether its kernel body was unchanged;
- candidate-versus-baseline algorithm, source-lineage, launch-topology, ABI,
  and fusion-boundary differences;
- correctness versus that kernel, plus any supplemental oracle;
- candidate and baseline latency for every required shape;
- speedup versus open source, not versus LLMDiveDeep;
- a result matrix whose rows are shapes/messages/images and whose method
  columns include every executed open-source/native implementation and every
  candidate variant;
- fastest-baseline-per-shape and SOL/headroom columns;
- output-contract or environment differences;
- an explicit qualification status;
- alternatives, explored variables, and the measured reason for the final
  choice when more than one implementation is plausible;
- containing-pipeline and layer results only after standalone acceptance.

Do not visually present an unqualified result as a positive speedup. Keep local
baseline numbers in a clearly labeled secondary section.
