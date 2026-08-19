# TileRT study — B200 feasibility for Kimi-K3 mega-MoE / mega-KDA

Date: 2026-08-19
Qualification: install/run findings QUALIFIED (executed on B200, container
llmdd-route-bench, torch 2.12 nightly cu13.2); everything not in the
"executed" table is source-traced from the public repo + shipped binaries.
Latency numbers are TileRT's own DSv3.2 shapes — anchors only, never
Kimi-K3 numbers.
Source: github.com/tile-ai/tilert @ a8368a6 (v0.1.5 tag, 2026-07-14) +
prebuilt wheel `tilert-0.1.5.post2` from GitHub Releases (PyPI is blocked;
the release wheel is the same artifact), venv `/workspace/tilert-venv`.
Companion study: `mirage_megakernel_results.md` (Mirage MPK) — the verdict
section compares the two directly.

## What TileRT is, in one paragraph

An ultra-low-latency decode engine for exactly two models — DeepSeek-V3.2
and GLM-5/5.1 — on exactly one topology (8×B200, TP8, batch=1), shipped as
**closed-source prebuilt binaries** (`libtilert_dsv32.so` /
`libtilert_glm5.so`, sm_100 cubins only). The GitHub repo is a
"presentation copy" of the Python glue; its own pyproject says the wheel
is built in a private dev repo and deliberately omits a `[build-system]`
block. The runtime model (from binary symbols + host API imports, all
verified below): a compiler decomposes each fused layer-op into a static
tile-task instruction schedule executed by ONE `ExecutorImpl` kernel per
op (`tilert::core::executor::piped_prefetch::<Op>ExecutorImpl<DefaultSchedule,…>`
taking `tilert::core::vm::GlbArgs<8,2>` — pointers to all 8 GPUs'
buffers); a decode step is a pre-instantiated CUDA graph of these
executors per device (`cudaGraphInstantiate/Launch` +
`cudaGraphKernelNodeSetAttribute`, i.e. PDL-style chaining), driven by ONE
host call (`dsa_show_hands(token)`) for all 8 GPUs, with sampling and
custom low-latency P2P allreduce (exchanged `ll_buf` device pointers)
fused in. Their 2026-06 blog describes a further consolidation into a
single "Persistent Engine" with heterogeneous workers — that iteration is
NOT in the public wheel and could not be verified. Compiler techniques are
promised "gradually" via TileLang/TileScale; nothing is published yet.

## What ran on B200 (all executed today)

| test | result | note |
|---|---|---|
| wheel install (GitHub release, `--no-deps`) + `load_backend` on torch 2.12 nightly | **PASS** | pinned torch 2.11.0+cu130 ABI tolerated; sm_100 cubins confirmed via cuobjdump |
| `tilert_init_op` (8-GPU context/P2P init) | **PASS** | required before any op |
| `topk_accurate_op` (sparse-index top-2048 of 4096) | **PASS**, index set == torch.topk | 210.6 us/iter host-launched (wrapper allocs per call — anchor only) |
| `expert_select_up_gate_silu_op` fp8mma, DSv3.2 shape (256+1 experts, top-8, 7168→256/dev), dev 0 of 8 | **PASS**, cos 0.999304 vs golden on kernel-selected experts; routing == group-limited sigmoid+bias top-8 | 24.0 us/iter host-launched; **10.4 us GPU self-time, exactly 1 kernel + 1 small HtoD per call** (torch.profiler) |
| same op fed Kimi-K3 MoE shape (896 experts, top-16, inter 384) | **FAIL** (expected, root-caused below) | shapes are compile-time baked |
| outer `torch.cuda.graph` capture around a TileRT op | **FAIL** (root-caused below) | TileRT owns graph capture internally |
| e2e `dsa_show_hands` decode step | **NOT RUN** (root-caused below) | needs ~700 GB converted checkpoint |

Failure root causes (rule 19):
- **K3-shape probe:** host weight converter crashes at
  `reshape(8, 1, 0, 56)` — moe_inter 384/8 = 48 rows/device < block_size
  128 makes `scale_m_dim` 0. Upstream of that, every executor kernel is a
  C++ template with expert count/dims/SM count (144) baked at compile time
  in the closed dev repo; no runtime shape path exists.
- **Outer graph capture:** each op call runs the library's own
  capture/instantiate/launch machinery
  (`register_expert_select_up_gate_silu.cu:532`); nesting that inside a
  torch capture (or invoking it on the legacy default stream) raises
  `operation not permitted when stream is capturing`. Fix used by their own
  generator and by our probes: call `tilert_init()` first and launch ops on
  a non-default stream; never wrap TileRT ops in an external graph.
- **e2e not run:** requires the DSv3.2/GLM-5 checkpoint resharded by their
  converter (~700 GB — not present on this host), and the public copy's
  random-weights fallback is broken: `init_random_weights()` calls
  `_init_weights(None)` which asserts `model_path is not None`
  (`end2end.py`). Op-level runs above stand in as the executed evidence.
- **nsys captured no CUDA events** even for a plain torch matmul in this
  container (CUPTI/environment limitation, not TileRT); torch.profiler
  worked and was used for the kernel-count/GPU-time numbers.

## Feature map (source-traced)

- **Models/shapes:** DSv3.2 (61L, 256 experts top-8 group-limited, dim
  7168) and GLM-5/5.1 (92L-class, 256 experts top-8 sigmoid, dim 6144).
  Two hardcoded backends; one per process.
- **Dtypes:** fp8 e4m3 with DeepSeek 128×128 blockwise scales (weights
  swizzled host-side into a fused weights+scales page layout), bf16/fp16
  MMA fallbacks. FP4 appears in exactly one shipped kernel
  (`RmsnormProjQWqbFP4HMMA`, DSv3.2 q_b proj). **No MXFP4/NVFP4 and no FP4
  MoE in the public wheels** (their MiMo blog deployment uses FP4 experts —
  newer than what ships).
- **Op inventory (dsv32, 278 kernel entries):** fused rmsnorm+proj ops for
  the whole MLA/DSA path, `TopkAccurate/Approximate`,
  `ExpertSelectUpGateSiLU` (routing + shared+top-8 expert up/gate GEMV +
  silu·mul in ONE kernel), `ExpertDownAllreduce` (down proj with allreduce
  epilogue), `FusedMoe`, `FlashSparseMla`, sampling (`ExecuteTopP`), MTP
  verify. Attention family only — **nothing conv/SSM/linear-attention
  shaped, so no KDA-relevant op exists**.
- **MoE layout:** all 257 experts on every GPU, sharded along moe_inter
  (TP-in-expert, 256 rows/device); routing recomputed per device; no
  expert-parallel dispatch/combine at all — decode-batch-1 specialization.
- **TP/multi-GPU:** single process drives 8 GPUs; P2P device-pointer
  exchange at init; allreduce fused into GEMM executors as LL P2P — no
  NCCL in the step path.
- **Decode vs prefill:** decode-only, batch=1 (asserted). Non-MTP "prefill"
  is literally one token per step through the decode engine; MTP mode does
  4-token chunks; real prefill is delegated to vLLM via their PD
  disaggregation (vLLM prefill → NIXL/Mooncake KV handoff → TileRT decode).
- **Authoring/registration:** none public. Ops are torch custom ops
  registered inside the closed .so; the Python layer only converts weights
  and calls them.

## Why the fused executor is fast (mechanism, per their own shapes)

The 10.4 us `ExpertSelectUpGateSiLU` kernel does what our expert-region
pipeline does in several launches: routing (group-limited sigmoid+bias
top-8 over 256), then 9 dual GEMVs (7168→256 fp8, ~33 MB of touched
weights ≈ 5 us of mandatory bytes at B200 HBM speed) and silu·mul, in one
144-SM kernel whose tile schedule prefetches expert weights while routing
resolves — the same critical-path idea as our PDL/megakernel ladder, here
compiler-generated. The step-level design (pre-instantiated graph + PDL +
in-kernel LL allreduce + GPU-side sampling, one host call per token for 8
GPUs) is the strongest production evidence yet for the direction our
graph-bucketed serving probe and MPK study point at.

## Kimi-K3 assessment: mega-MoE and mega-KDA on TileRT

- **Mega-MoE (896 experts, top-16, W4A8 MXFP4/MXFP8, hidden 7168, inter
  384, TP8):** the op that exists is the right *shape of idea* (their
  ExpertSelectUpGateSiLU ≈ our route+expert region as one kernel, and
  Kimi-K3's routing is the same sigmoid+bias family), but every relevant
  parameter is compile-time baked in a closed compiler: 896/16 vs 256/8,
  inter 48/dev vs their 256/dev (breaks even their host converter), MXFP4
  vs their fp8-blockwise. There is no kernel/task authoring surface to
  port anything into. Effort to use TileRT = reimplement the closed
  compiler+VM, or wait for TileLang/TileScale drops.
- **Mega-KDA (in_proj → conv k=4 → f_b → gated-delta recurrence → o_proj;
  decode bound ~24-25 us measured):** zero ingredients — no conv op, no
  recurrent-state op, no fp32-state anything in either backend; both
  backends are MLA/DSA-attention models. Everything would be new, inside a
  compiler we cannot see.
- **What TileRT *does* give us:** (1) executed proof that the
  per-op-fused-executor + graph/PDL + fused-LL-allreduce decode design
  works in production on 8×B200 at DeepSeek scale (README: up to ~600
  tok/s DSv3.2 ⇒ ~1.7 ms/step for 61 layers, their number, not measured
  here); (2) a measured anchor that a single fused route+expert kernel at
  a DSv3.2-like shape lands at ~10 us GPU time on one B200; (3) design
  patterns worth stealing openly: TP-in-expert layout for batch-1 decode
  (kills dispatch/combine), allreduce as a GEMM-epilogue task, GPU-side
  sampling in the same graph, weights+scales fused page layout.

## Verdict and ranked next steps

**TileRT is not a usable substrate for Kimi-K3 megakernels — the compiler
and runtime that would have to be extended are closed source, and the
shipped binaries are hardwired to two models' shapes.** Mirage MPK remains
the actionable megakernel route: its gaps (radix-counting router task,
~250-line KDA recurrence task, no MXFP4, world_size-1 builders) are all
open-code, well-scoped work; TileRT's gap is "no source". As between the
two runtime designs, TileRT's shipped model (per-fused-op executor kernels
chained by graph+PDL) is actually the *nearer* target for us — it is our
existing graph-bucketed + PDL direction taken to its limit — while MPK's
single persistent worker+scheduler kernel is the further, more invasive
step.

1. Proceed on the MPK track (KDA recurrence task, radix router task) as
   ranked in `mirage_megakernel_results.md` — unchanged by this study.
2. Adopt TileRT's fusion boundaries as targets for our own kernels:
   route+up/gate+silu as one launch for the K3 expert region (its 10.4 us
   DSv3.2 anchor vs our 22-92 us expert region), and allreduce-in-epilogue
   for the TP8 tail.
3. Watch TileLang/TileScale releases for the executor/VM compiler; if the
   tile-instruction VM lands open, it leapfrogs both our hand fusion and
   MPK porting.
4. Blocked/deferred: TileRT e2e measurement (needs ~700 GB checkpoint);
   any TileRT extension work (closed source).

## Reproduce (container llmdd-route-bench; each < 5 min warm)

```bash
source /workspace/tilert-venv/bin/activate && cd /workspace/tilert-probes
python probe_topk.py               # topk PASS + timing, ~20 s
python -u probe_expert_upgate.py   # fused expert op: cos, 24 us host / 10 us GPU, ~30 s
python -u probe_k3_shape.py        # K3-shape rejection (converter reshape error), ~15 s
python -u probe_routing_and_graph.py  # routing rule + outer-capture failure demo, ~20 s
# one-time setup that produced the venv (wheel from GitHub Releases, PyPI blocked):
#   python3 -m venv --system-site-packages /workspace/tilert-venv
#   wget https://github.com/tile-ai/TileRT/releases/download/v0.1.5.post2/tilert-0.1.5.post2-cp312-cp312-manylinux_2_28_x86_64.whl
#   pip install --no-deps --no-index tilert-0.1.5.post2-*.whl
```
