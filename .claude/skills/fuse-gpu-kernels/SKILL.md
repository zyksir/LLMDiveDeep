---
name: fuse-gpu-kernels
description: Evaluates and implements GPU kernel fusion only after correctness and end-to-end performance comparisons with unchanged open-source/native fused or unfused pipelines. Use for adjacent-kernel fusion, PDL, and megakernels.
---
# Fuse GPU kernels

Last updated: 2026-08-18.

Read [the baseline policy](../optimize-gpu-kernels/BASELINES.md) and
[the report policy](../optimize-gpu-kernels/REPORTING.md).

## Mandatory task-confirmation gate

Before evaluating or implementing fusion, ask the user to confirm the exact
region: component kernels, whether unfused composition and fusion are both in
scope, direct inputs/outputs and precision, required baselines, and excluded
layer/model/deployment integration. Ask even when the request appears clear.
Do not expand an explicitly confirmed region based on downstream consumers or
an earlier broader goal.

## Highest-priority decomposition rule

Fusion work still starts from standalone kernels. Split the proposed region
into its component kernels and, on the minimum required GPU count:

1. benchmark and verify each unchanged open-source/native component alone;
2. benchmark and verify each separate candidate alone;
3. benchmark the smallest complete unfused region;
4. implement and benchmark the smallest useful fusion boundary;
5. integrate into the layer and end-to-end workload only after isolated gates
   pass.

Use preallocated direct inputs/outputs and one process for local kernels. If an
internal stage lacks a public entry point, expose it through a thin untimed
launcher instead of initializing a model, full MoE layer, or TP group. Multiple
GPUs are justified only when the fused region itself communicates.

The containing layer determines whether isolated gains survive integration,
but it is a checkpoint—not the per-edit fusion loop.

## Fusion baseline

Complete the ecosystem survey required by the baseline policy before choosing
a fusion boundary. Trace current SGLang, TensorRT-LLM, vLLM, FlashInfer, and
model-author dispatch through routing, permutation, and expert execution;
record which stages are separate, fused, or delegated to a backend.

The baseline is the fastest unchanged open-source/native implementation of the
entire region being fused. Compare against the best available fused pipeline
and the best unfused partition. Build the best unfused partition by executing
the compatible cross-product of serious stage winners; it may combine an
unchanged upstream stage with a local candidate. Do not assume either the
all-upstream or all-local sequence is strongest. Charge casts, repacks,
intermediate writes, and synchronization to the composition that requires
them. A sequence of LLMDiveDeep kernels is only a migration baseline.

Do not use a DeepSeek-V3/DeepV3 pipeline as the Kimi-K3 baseline unless exact
contract equivalence and strongest-known performance are demonstrated.

## Workflow

1. Freeze the full region's inputs, outputs, precision, tie behavior, padding,
   side effects, synchronization, and graph dependencies.
2. Run the unchanged open-source pipeline and compare every observable output.
3. Benchmark each open-source stage and the complete region.
4. Measure separate optimized stages before fusion.
5. Benchmark mixed stage compositions and select the strongest unfused region.
6. Evaluate launch removal, eliminated traffic, register/shared-memory growth,
   occupancy, parallelism, overlap, and library-kernel loss.
7. Implement the smallest useful fusion boundary.
8. Recheck direct output equality/tolerance against open source on adversarial
   inputs and changing-input CUDA graphs.
9. Accept fusion only when the complete region and containing layer improve
   against open source.

One launch is not an acceptance criterion. PDL or a two-stage sequence may be
faster than a megakernel.

## Fast fusion loop

Before exploring fusion boundaries, make one open-source-versus-candidate
region A/B fast enough for repeated use. Cache compilation and graph capture,
reuse buffers, vectorize correctness normalization, and use representative
boundary shapes in quick mode. Measure and print harness wall time.

Do not run full model initialization, all shapes, or Nsight on every edit when
a smaller baseline-gated region replay answers the current question. Run the
full containing-layer matrix at checkpoints and before accepting fusion.

## Reporting gate

Write the mandatory optimization report for every fusion attempt. Its
alternatives section must compare the meaningful unfused partition, partial
fusion boundaries, full fusion or megakernel, PDL/overlap option, and important
launch or tile variables. State the measured reason for the selected boundary;
if evidence is incomplete, record that no fusion decision is qualified. Use
controlled region ablations to show that eliminated launches or memory traffic,
not omitted work or a contract difference, caused the gain.

Without an executed unchanged open-source correctness and performance baseline
for the same region, label results `UNQUALIFIED`. Do not claim fusion speedup,
readiness, or superiority. Report local/current comparisons separately.
