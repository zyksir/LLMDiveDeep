---
name: profile-gpu-kernels
description: Profiles LLMDiveDeep GPU kernels and validates conclusions against unchanged model-relevant open-source/native kernels. Use for Nsight Systems, Nsight Compute, CUDA graph, overlap, and SOL analysis.
---
# Profile GPU kernels

Last updated: 2026-08-18.

Read [the baseline policy](../optimize-gpu-kernels/BASELINES.md) and
[the report policy](../optimize-gpu-kernels/REPORTING.md).

## Highest priority: profile the isolated kernel first

Decompose the workload and build a standalone benchmark that launches exactly
the target kernel before profiling a layer or model. Use one GPU and one process
for a local kernel, preallocate direct inputs/outputs, and exclude model
initialization, experts, collectives, and unrelated kernels.

If a kernel is only exposed through an internal library path, create a thin
launcher for that exact kernel. Do not run a full MoE layer or TP group merely
to obtain its routing or permutation profile. Use multiple ranks only when the
target kernel itself communicates, and then use the minimum valid topology.

Profile in this order:

1. all applicable standalone unchanged baselines, then the candidate;
2. the compatible cross-product of serious stage winners for a decomposed
   operation, including mixed upstream/local compositions;
3. smallest containing pipeline after isolated conclusions are stable;
4. containing layer and end-to-end integration at checkpoints.

Do not profile only the all-local pipeline when a faster upstream stage can be
reused. Include required casts, repacks, and synchronization in each composed
region. A containing-pipeline profile is incomplete until the strongest
compatible composition has been executed rather than inferred from isolated
latencies.

The layer profile validates integration and overlap. It is not the per-edit
kernel profiling loop.

## Baseline before profile

Complete the ecosystem survey required by the baseline policy first: trace
current SGLang, TensorRT-LLM, vLLM, FlashInfer, and model-author dispatch to
the kernel that actually runs for the target configuration. Framework names
alone do not identify kernels.

Do not profile only an LLMDiveDeep kernel and call the resulting opportunity an
optimization target. First execute an unchanged, strongest relevant
open-source/native kernel with the same contract and workload.

For every profile, preserve:

- baseline repository, commit, file, and license;
- exact input/output contract and model-specific arguments;
- GPU, clocks, software versions, rank topology, and graph/eager mode;
- identical warmup and representative changing inputs.

DeepSeek-V3/DeepV3 is not a valid Kimi-K3 baseline by name alone. Prove contract
and workload equivalence and strongest-known status before using it.

## Procedure

1. Freeze the one-kernel meaning, inputs, outputs, exact contract, and timed
   boundary; create its standalone launcher with representative synthetic
   tensors.
2. Measure cold, cached-quick, and full standalone benchmark wall time.
3. If feedback is slow, profile the harness first: imports, compilation,
   distributed/model startup, synchronization, graph capture, and repetitions.
   Remove distributed/model startup unless intrinsic to the target kernel.
4. Verify candidate outputs directly against the unchanged open-source kernel.
5. Use Nsight Systems on the isolated launch to identify launch and dependency
   behavior, then on the containing path to locate launch gaps,
   serialization, communication overlap, and CUDA graph behavior.
6. Use Nsight Compute on candidate and all serious open-source kernels for launch,
   occupancy, instruction, memory, stall, and SOL metrics.
7. Build a per-shape SOL estimate from the maximum applicable launch,
   bandwidth, compute, and serial dependency/reduction lower bounds. Measure
   every term (empty-kernel floor, streaming probe, N-phase barrier probe,
   per-warp instruction-issue estimate for few-warp shapes); never assert it.
   State assumptions; do not use nominal FLOPs alone for selection, scan, or
   permutation kernels, and never present a launch-floor-only ratio as
   headroom. If latency is non-monotonic across shapes, profile the executed
   kernel names per shape — it is an internal dispatch switch, often mis-tuned
   for the target model. Label every SOL column with its operation; stage
   bounds are not interchangeable.
8. Before candidate work, stop if the fastest unchanged methods are already
   close to the bound and no material profiled mechanism remains.
9. Attribute gains to measured differences rather than local-baseline latency.
10. Profile the containing layer only after standalone measurements and a clear
   isolated pass or stop decision.

Keep a cached quick path with representative shapes and short graph timing.
Run expensive Nsight and full-layer paths at checkpoints, not after every edit.
Quick mode must still execute the unchanged open-source baseline.

## Reporting gate

Write the mandatory optimization report even when profiling disproves the
idea. Begin with a concise explanation of the bottleneck and proposed
mechanism. Record alternative profile-driven hypotheses, variables, evidence,
and why the chosen direction won or why all variants were rejected. Require the
profile to confirm the claimed mechanism changed; a latency delta with no
corresponding launch, traffic, instruction, stall, occupancy, or critical-path
evidence remains `UNQUALIFIED`.

If the unchanged open-source kernel was not run for correctness and
performance, label the report `UNQUALIFIED`. Do not use positive speedup/SOL
language or recommend shipping. PyTorch and the current LLMDiveDeep kernel may
appear only as supplemental oracle and migration data.
