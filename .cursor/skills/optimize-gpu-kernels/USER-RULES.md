# USER RULES — explicit standing instructions (read FIRST, apply ALWAYS)

These were each stated explicitly by the user (2026-08-18/19). They override
defaults and habits. Violating one is a process failure, not a style choice.

1. **FlashInfer prebuild first.** At the start of ANY task, launch the
   FlashInfer JIT module build in the background before other work.
2. **Compile cost is a product concern.** Nothing may compile per shape;
   no compilation inside timed or interactive loops; caches warmed at
   setup/deploy (applies to FlashInfer, DeepGEMM, torch extensions alike).
3. **Never name a kernel or strategy "ours"/"baseline"/"opt".** Name it by
   the kernel or idea that runs: radix, triton, trtllm, moe_sort,
   llmdd_permute_v1_pdl.
4. **SOL is always a measured, reachable number.** Every SOL cell in every
   table comes from a probe that actually executed (mandatory-work kernel,
   equivalent-bytes GEMV/clone, flops-equivalent GEMM). Never `n/a`, never
   an unreachable launch-floor/bytes-only figure. "A raw MoE is at least
   one GEMM/GEMV" — build that probe and run it.
5. **Reports = numbers + explanations + reachable SOL, nothing else.**
   Follow REPORTING.md section order. The survey must say in plain
   sentences what each implementation does differently algorithmically and
   why; a dedicated section explains why the remaining gap to SOL is or is
   not recoverable. No process history in reports.
6. **Full batch range in EVERY table**: 1, 2, 4, 8, 16, 32, 64, 128, 256,
   512, 1024, 2048, 4096, 8192, 16384 — including integration/layer tables.
   Missing sizes get measured, not omitted.
7. **Bold the per-row winner in every table** so the best implementation is
   readable at a glance.
8. **Reports are files.** Deliver the markdown file path plus a 2-3 line
   takeaway in chat; never print report contents into chat.
9. **Keep working; don't stop at checkpoints.** Continue autonomously until
   blocked on something only the user can decide.
10. **Show a to-do plan and keep steps small.** Every step is trivial
    (build/measure/decide); spending long on one step means the approach is
    wrong.
11. **Measurements drive dispatch.** Full-range per-batch tables exist to
    pick which kernel serves which batch size, and the layer implementation
    must encode that dispatch.
12. **Measure sub-microsecond A/Bs in one process** — run-to-run clock
    drift is ~±0.2 us; separate invocations cannot decide small deltas.
13. **Provenance prefixes on every kernel name.** Kernels written in this
    repo: `b10_`; SGLang kernels: `sgl_`; TensorRT-LLM kernels: `trt_`;
    vLLM: `vllm_`; DeepGEMM: `dg_`. Applied in code registries, benchmark
    columns, and report tables alike.
14. **Every table carries per-regime explanations.** Under each result
    table, state WHY the winner wins in each batch regime (the mechanism,
    one or two sentences per regime). A table without the why is
    insufficient information.
15. **Long autonomous sessions are expected.** When given a time budget,
    keep iterating experiments until every kernel decision is finalized;
    do not stop at intermediate checkpoints.
16. **Reproduce commands run in under 5 minutes each.** Every results file's
    Reproduce section must finish fast (single kernel/layer runs are
    milliseconds; a full matrix is seconds of measurement). If a reproduce
    command needs more than 5 minutes, the harness is wrong — split it,
    reduce repetitions, or cache setup, and state the expected wall time
    next to each command.
17. **Whole-layer measurements always run under CUDA graphs.** Host launch
    bubbles must never shape a layer-level conclusion; if a path cannot be
    graph-captured, report GPU-kernel-sum time alongside wall time and label
    the wall number as host-bubble-inclusive.
18. **The loop is analyze -> implement -> measure -> re-analyze, always.**
    After every result: check monotonicity (latency must grow with batch
    size — any non-monotonic row gets a root-cause analysis in the report),
    analyze every case where a b10 kernel loses to a baseline (b10 is
    expected to win; a loss is a finding to explain, not just record), and
    only then pick the next improvement and re-implement. Never deliver a
    bare number without its analysis.
19. **Explain the winner, and fix graph failures.** Every report states WHY
    the winning implementation beats each alternative (mechanism, not just
    numbers). Every CUDA-graph capture failure gets a root cause (which op
    invalidates capture) and a fix or a documented blocker — never just
    "GRAPH-FAIL".
