# Kernel optimization report policy

Last updated: 2026-08-18.

Every optimization attempt must leave a concise report, including rejected,
slower, incorrect, and unqualified ideas. Code and benchmark output alone are
not a completed kernel task.

## Attribution and anti-cheating gate

Write the optimization hypothesis before or with the first candidate. It must
be falsifiable: name the instruction, memory traffic, launch, synchronization,
occupancy, parallelism, or algorithmic critical path expected to improve.

An accepted speedup needs all of:

- **equal work:** identical math, precision, inputs, output writes,
  initialization guarantees, launch mode, and timed boundaries;
- **idea-to-code traceability:** identify the code changes that implement the
  stated mechanism rather than presenting an unexplained rewrite;
- **attribution evidence:** use an ablation or controlled variant that disables
  the idea while holding other material changes fixed when practical;
- **measured mechanism:** show the expected counter changed, such as fewer
  launches, bytes, barriers, instructions, spills, or shorter critical path;
- **baseline integrity:** retain the unchanged baseline source hash and keep
  all adapters outside timing or charge them equally.

Treat an unexplained speedup as suspicious. If the candidate performs less
work, silently changes semantics, excludes required output or setup, or cannot
connect the gain to a credible mechanism, label the optimization
`UNQUALIFIED` even when latency is lower.

## Required report order

### 0. Operation contract

Explain what the kernel means in the model or pipeline. Record direct inputs,
outputs, exact math, dtypes, strides, alignment, supported shapes, side
effects, tie/order behavior, output consumers, and the work explicitly outside
the timed boundary. If a user-facing name covers multiple kernels, show the
stage decomposition and give each stage a separate contract.

### 1. Ecosystem implementation survey

Before the idea, map the current SGLang, TensorRT-LLM, vLLM, FlashInfer, model
author, and other relevant implementations to their actually executed kernel
symbols. Record version/commit, source path, algorithm, production dispatch,
fusion boundary, ABI, precision, and whether each implementation qualifies as
a correctness or performance baseline.

Do not continue to candidate optimization while a major current framework is
unchecked. Same math does not imply shared implementation.

### 2. Pre-candidate benchmark, SOL, and stop decision

Before presenting a candidate, report the standalone latency of every
applicable unchanged implementation across the required shape matrix. Include
the fastest baseline per shape and the defensible SOL lower bound with its
launch, bandwidth, compute, and serial-dependency assumptions.

State `STOP` when the best methods are already close to the bound and no
material removable work is visible. A stop report is a successful result and
does not need candidate code. Otherwise state `PROCEED` and identify the
measured headroom that motivates the candidate.

Record benchmark-harness wall time here. If the harness is slow, report the
measured CPU/setup costs and the changes made before kernel iteration.

Record the stable runner revision/configuration and the backend modules used.
If the central runner changed during the experiment, state exactly which shared
contract, timing, result-schema/table, or bug-fix reason required it. Adding a
backend is not by itself a valid reason to edit the runner. Report focused
backend-only runs separately from final full-matrix checkpoints so settled
baselines are not repeatedly rerun without cause.

### 3. Idea

Explain the optimization in two to four sentences:

- what measured bottleneck or unnecessary work it targets;
- what execution, dataflow, or algorithm change it makes;
- why that change should improve the target workload;
- which semantics and output obligations remain unchanged.

The explanation must distinguish the candidate from the baseline. “Specialized
kernel,” “better tiling,” or “faster implementation” is insufficient without
the concrete work or critical-path difference. Do not begin with implementation
history or a long kernel walkthrough.

For a stop report, replace this section with “No candidate: stop gate passed.”

### 4. Baseline

Name the unchanged strongest relevant open-source/native correctness and
performance baselines. Record repository, commit/version, file or symbol,
hardware, software, exact workload, output ABI, and launch mode. State
explicitly when a baseline is only supplemental or migration data.

Add a concise algorithm and lineage comparison:

- whether candidate and baseline share source, algorithm, or kernel lineage;
- route-only versus fused-pipeline scope;
- launch topology, threads/warps, tiling, and synchronization;
- input/output dtype and ABI;
- work fused, removed, added, or moved outside timing.

If the candidate is materially the same kernel, say so directly and do not
claim novelty. Distinguish a convenient standalone baseline from the actual
production kernel when they differ.

### 5. Results

Report:

- direct correctness result and tolerance;
- baseline and candidate latency for every required shape;
- speedup or slowdown versus the unchanged performance baseline;
- standalone benchmark wall time;
- qualification status: `QUALIFIED`, `UNQUALIFIED`, or `REJECTED`;
- missing gates, contention, contract differences, or other limitations.
- attribution ablation and the measured mechanism change.

Lead with the open-source/native comparison. Never present an internal
LLMDiveDeep comparison as the optimization result.

#### Mandatory result matrices

Generate tables from structured benchmark output such as CSV or JSON rather
than manually transcribing terminal text.

The primary latency matrix must use:

- one row per batch size, message size, image size, or tensor shape;
- one latency column, with units in the heading, for every applicable
  unchanged method and every candidate variant;
- explicit `N/A` entries with reasons for unsupported or blocked methods;
- fastest unchanged baseline, candidate speedup/slowdown versus that baseline,
  SOL lower bound, and achieved SOL/headroom columns;
- stable method names that include framework and kernel/variant, not generic
  labels such as `baseline` or `ours`.

Use a separate matrix for each non-equivalent ABI or stage. Never place
route-only and route-plus-permutation numbers in one speedup comparison unless
the missing work is equalized and charged. Add a compact correctness matrix
when methods use different tolerances, output sets, or support ranges.

For a decomposed operation, add a composition matrix before making the final
choice. Its columns must include the serious compatible combinations of stage
winners, such as unchanged SGLang route plus a candidate permutation, not only
an all-baseline pipeline and an all-local-candidate pipeline. Charge every
required cast, repack, adapter, synchronization, and intermediate tensor to the
combination that needs it. Mark the report `UNQUALIFIED` if the strongest
compatible composition was inferred by adding stage latencies rather than
executed and measured directly.

Do not use color alone to communicate winners or qualification. Absolute
latencies, units, qualification, and comparison basis must remain readable as
plain text.

### 6. Alternatives and decision

This section is required when the idea has multiple algorithms, launch
decompositions, fusion boundaries, tile sizes, output modes, precision choices,
or other meaningful variables.

For each serious alternative, state:

- what changed;
- which variables were explored;
- correctness and performance evidence;
- why it was selected, rejected, or deferred.

For multi-stage work, alternatives include mixed-author compositions. Never
use "ours" as an implicit requirement that every stage be locally authored.
Choose the fastest measured compatible pipeline even when it combines an
unchanged upstream kernel with a local candidate.

Finish with the chosen variant and the measured reason for choosing it. If the
evidence is incomplete, say that no final choice has been made. Keep tunable
variables and regime boundaries visible instead of hiding them in code.

For multi-regime dispatch, list every V1/V2/etc. variant, its intended shapes,
measured crossover, dispatch predicate, boundary-shape results, compile/cache
cost, and runtime dispatch overhead.

### 7. Integration

After standalone acceptance, report the containing-pipeline, layer, and
end-to-end effect. Keep integration results separate from standalone kernel
results so regressions and interactions are attributable.

## Minimal template

```markdown
# <optimization> report

Date: YYYY-MM-DD
Qualification: QUALIFIED | UNQUALIFIED | REJECTED

## Operation contract
- Meaning: <model/pipeline role>
- Inputs: <shape, dtype, stride, semantics>
- Outputs: <shape, dtype, semantics and side effects>
- Timed boundary: <included/excluded work>
- Decomposition: <one kernel, or stage list>

## Ecosystem implementation survey
- SGLang: <dispatch → kernel, version, algorithm, ABI>
- TensorRT-LLM: <dispatch → kernel, version, algorithm, ABI>
- vLLM: <dispatch → kernel, version, algorithm, ABI>
- FlashInfer/model author/others: <dispatch → kernel or not applicable>

## Pre-candidate benchmark and SOL
- Decision: STOP | PROCEED
- SOL model: <launch/bandwidth/compute/dependency bounds and assumptions>
- Headroom: <fastest unchanged method versus bound>
- Harness wall time: <cold / cached / full>

## Idea
<Two to four sentences.>

## Baseline
- Correctness: <unchanged source/version/symbol>
- Performance: <unchanged source/version/symbol>
- Contract/environment: <shape, dtype, ABI, GPU, graph/eager>
- Algorithm/lineage difference: <shared or different; concrete topology/work>

## Results
- Correctness: <pass/fail and tolerance; matrix if contracts differ>

| Shape | SGLang/kernel (us) | TRT/kernel (us) | vLLM/kernel (us) | Candidate V1 (us) | Candidate V2 (us) | Fastest unchanged | Speedup | SOL bound (us) | Achieved SOL |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| <shape> | <latency> | <latency> | <latency> | <latency> | <latency or N/A> | <method> | <value> | <bound> | <value> |

- Attribution: <ablation and changed launch/byte/barrier/instruction metric>
- Limitations: <missing gates or none>

## Alternatives and decision
- <variant>: <evidence and disposition>
- Chosen: <variant and measured reason>

## Integration
- <not run until standalone acceptance, or measured layer/e2e result>
```
