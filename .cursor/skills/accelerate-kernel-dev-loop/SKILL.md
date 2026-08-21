---
name: accelerate-kernel-dev-loop
description: Diagnoses and reduces GPU kernel correctness-and-performance benchmark wall time without weakening open-source baseline gates. Use when kernel, layer, distributed, CUDA graph, or profiling feedback loops are slow.
---
# Accelerate the kernel development loop

Last updated: 2026-08-18.

Read [the baseline policy](../optimize-gpu-kernels/BASELINES.md) and
[the report policy](../optimize-gpu-kernels/REPORTING.md). A faster loop must
still execute unchanged open-source correctness and performance baselines.
Do the ecosystem dispatch survey once before optimizing the loop, then cache
its source/version evidence and compiled baseline artifacts; never replace it
with an assumed framework label.

## Mandatory task-confirmation gate

Before setup, compilation, or timing, ask the user to confirm the exact task:
included standalone kernels, included compositions or fusions, excluded
integration boundaries, dtype/layout/output contracts, shape matrix,
baselines, and requested artifacts. Ask even when the initial wording appears
clear. Do not optimize harnesses or compile adjacent stages until this boundary
is explicitly confirmed.

## The canonical LLMDiveDeep -> TRT-LLM loop

For any kernel or strategy headed to production, run this sequence in order —
each step gates the next:

1. **Per-kernel bench** (`kimi_k3_layer/kernels/bench_*.py`,
   `communication/kernel_benchmarks/bench_*.py`,
   `communication/bench_comm_graph.py`): reproduce the report's number for
   every kernel the change touches, including the communication kernels.
2. **No report? Produce one.** A kernel without a recorded number on the
   current node/arch/width gets a report row before it is used anywhere.
3. **Beat the baseline.** Most kernels must beat their unchanged open-source
   or native baseline in the same bench; one that does not needs an explicit
   recorded reason to exist.
4. **Layer bench** (`kimi_k3_layer/bench_b10_kimi_k3_moe_layer.py`): the whole
   MoE layer, matched against the report table (correctness-gated).
5. **Migrate to TRT-LLM** (`kimi_k3_optim/`) and get the end-to-end serving
   number. Serving must re-establish every invariant the bench provided
   (autotuned comm maps, tuned col-AG, fused dispatches) — a layer win does
   not compose by default.

## Small-setting dev loop, full-setting validation

Always develop on the smallest configuration that exercises the code path,
then validate on the full configuration before reporting or shipping. The two
runs answer different questions and neither substitutes for the other:

* **Dev loop (small):** shrink every axis that does not change the code path
  under test -- layer count (e.g. a 4-layer Kimi-K3: 3 KDA + 1 MLA + MoE keeps
  every layer type), TP degree (TP2 locally before TP8), batch/size lists
  (`--sizes 32` before `--sizes all`), iterations (`--iters 5 --n-inputs 1`),
  random/dummy weights (`load_format: dummy`) instead of checkpoints. Target:
  seconds-to-minutes per iteration, on one node, so failures localize fast.
* **Validation (full):** the real degree (TP8/EP8), the full size list, the
  real layer pattern and activation (SiTU, not a stand-in), table-grade
  timing (locked clocks, 100 iters, correctness gates). Only this run
  produces reportable numbers -- a small-setting result is a smoke signal,
  never a table entry.

Do not skip the small stage to "save time": a config or shape bug found at
TP8 with a full warmup costs a multi-minute cycle per attempt; the same bug
at TP2/4-layers costs seconds. And do not ship from the small stage: shard
widths, thresholds, and collective winners all change with the degree.

## Highest priority: isolate one kernel first

Decompose a layer, model, or fused pipeline into explicit kernel-sized
subtasks before running experiments. The first executable benchmark must invoke
only the target kernel and its applicable unchanged open-source/native
implementations.

- Use one GPU and one process unless the target kernel itself is distributed.
- Generate deterministic representative random tensors directly on the device;
  do not initialize a model solely to obtain kernel inputs.
- Do not initialize model weights, MoE experts, communicators, MPI/NCCL ranks,
  or a full layer merely to reach a local kernel.
- Preallocate the target kernel's direct inputs and outputs. Keep source
  compilation and thin ABI adapters outside the timed region.
- If an internal library kernel has no Python entry point, expose that exact
  kernel with a thin standalone launcher. Do not substitute a layer benchmark.
- For a genuinely inseparable fused or distributed operation, isolate the
  smallest semantically complete region and use the minimum required ranks.

Solve the subtasks in order:

1. freeze the single-kernel contract and applicable unchanged baselines;
2. build a standalone correctness harness;
3. build a standalone performance and profiling harness;
4. optimize the isolated kernel until it passes or reaches the stop criterion;
5. report and hand off the isolated kernel;
6. integrate and run containing-region, layer, or end-to-end gates only after
   the user explicitly requests that separate phase.

The containing layer is a checkpoint and integration test, not the per-edit
development loop. For example, never launch a TP8 MoE layer just to benchmark a
local routing kernel.

## Stable benchmark runner and backend API

Build the standalone benchmark as a stable runner plus backend modules. The
runner is infrastructure, not an experiment scratchpad.

The runner owns only shared policy:

- shape/dtype/layout matrix and deterministic input generation;
- the operation contract, oracle invocation, and common output checks;
- warmup, CUDA graph/eager timing, synchronization, repeats, and statistics;
- capability filtering, result schema, persistence, and table rendering;
- composition enumeration and charging of declared conversion stages.

Each backend lives in its own module and exposes one small common interface:

1. immutable identity and provenance;
2. supported contracts, dtypes, layouts, shapes, and output formats;
3. one-time build/load and buffer preparation;
4. one callable operation API;
5. returned outputs plus any explicit adapters, casts, copies, or postprocesses
   that must be charged.

Adding or removing SGLang, TensorRT-LLM, vLLM, FlashInfer, a branch, or a
candidate must normally mean adding or removing one backend module and selecting
it through registry/configuration. It must not require editing the benchmark
loop, correctness loop, timing helper, shape loop, or report serializer. Do not
hard-code backend-specific branches into the central runner.

Modify the runner only when the common operation contract, shared measurement
method, result schema, or output table intentionally changes, or when fixing a
runner bug. If adding a backend requires a runner edit, first treat that as an
interface-design failure: extend backend metadata or hooks narrowly and add a
regression test showing that the next conforming backend can be registered
without another runner change.

## Prebuilt-baseline and zero-compilation runner gate

The benchmark command must not discover external source trees or compile
extensions. Do not require runtime arguments such as `--sglang-root` or
`--vllm-root`.

Prepare each unchanged baseline once, in this order of preference:

1. load the exact operator from an installed package, as with a registered
   `torch.ops` implementation;
2. vendor the minimal unchanged upstream source and required headers locally,
   with repository, commit, license, and checksums;
3. use an external checkout only in a separate setup/import command that copies
   immutable source or produces a cached artifact, never in the benchmark run.

Never rewrite a vendored baseline kernel body. Build only for the target GPU
architecture, persist artifacts by source/compiler/framework/architecture hash,
and reuse them across processes and containers where ABI-compatible. Separate
`setup-baselines` from `benchmark`: setup may compile, while benchmark must
fail fast with a clear missing-artifact error instead of starting JIT
compilation. Report cold setup time separately; correctness and latency runs
start only after all backends are ready.

Do not repeatedly rerun settled backends while debugging one backend. Preserve
their pinned full results, use the common runner's backend and shape filters for
the active investigation, and rerun the full matrix only at a report checkpoint
or when shared inputs, contracts, timing, hardware state, or runner semantics
change.

## Stop and measure first

Before tuning a kernel, time one complete **standalone single-kernel**
edit-loop command. Report cold and cached wall time, then separate:

- container/process startup and Python imports;
- baseline and candidate compilation;
- extension loading and CUDA context creation;
- accidental model weights, MPI/NCCL, and rank startup that should be removed
  unless required by the target kernel;
- allocations, graph capture, synchronization, and correctness normalization;
- warmup, iterations, repeats, profiles, and result rendering.

If harness overhead limits experiments, optimize it before the kernel.

## Required tiers

- **Smoke:** one launch and one direct open-source output comparison.
- **Quick:** one current target shape by default; unchanged open-source
  correctness and short graph-timed performance; cached compilation and reused
  buffers. Add boundary shapes when the edit can affect dispatch boundaries.
- **Full:** all required shapes, adversarial inputs, independent native
  correctness, changing-input graphs, containing-layer timing, and profiles.

Run smoke/quick after edits. Run full at checkpoints and before reporting or
shipping. Never present quick measurements as the final matrix.

## Optimize the harness

1. Cache immutable open-source baselines by source hash and prebuild them during
   environment setup.
2. Keep adapters untimed and load compiled extensions once per process.
3. Reuse inputs, outputs, CUDA graphs, communicators, and model weights.
   Baseline graphs may be captured once, but replay and remeasure them every
   round so machine drift and contention remain visible.
4. Vectorize GPU correctness normalization; avoid `.item()` or Python loops over
   CUDA tensors.
5. Remove nested timing repetition. One benchmark helper should own warmup,
   iterations, repeats, and median selection.
6. Default quick mode to one representative target shape and short timing.
7. Use a persistent benchmark process when imports or distributed startup
   dominate.
8. Serialize memory-heavy compilation and GPU timing; concurrent benchmarks can
   OOM, contend, and invalidate latency.
9. Keep expensive Nsight, all-rank, and full-model runs out of the per-edit
   path.

## Guardrails

Do not gain loop speed by dropping the open-source baseline, changing output
obligations, reducing math accuracy, using stale candidate outputs, comparing
different graph modes, or excluding required work from only one side.

## Completion criteria

Record:

- a concise explanation of the harness optimization idea;
- cold setup seconds;
- cached quick-loop seconds;
- full correctness and performance seconds;
- what dominated before and after;
- the exact quick and full commands;
- which checks moved to checkpoint-only execution;
- alternative caching, persistence, graph, repetition, or process designs and
  the measured reason for the chosen design.

The loop is ready when its cached duration supports repeated experiments and
the full gate remains reproducible.
