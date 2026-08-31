# Autonomous stretch summary (2026-08-28 → 08-30)

## Sunday weekend-loop addendum (see BUG_ROOTCAUSES.md for full chains)

1. **fc1-shard segfault root-caused + fixed** (device-less host pointer
   tables under the ambient device ctx; 4-site device="cpu" fix).
   **tp_te + fc1shard + real gate = new best MoE decode config:
   91.1/118.3/155.2 us at B=8/32/64 (+4.8/+4.7/+3.8% over tp_te)**,
   correctness clean.
2. dsv3_router_gemm CUBLAS failure: not reproducible anymore (proven
   across eager/autotune/graph in-module) — bench gate UNSTUBBED, gate
   GEMM back in timing.
3. fc1-shard run_moe bypass skipped the perfect-router rewrite (bench
   routing unfairness with real gates) — fixed; this had inflated the
   stubbed-gate shard gains (+19% -> honest +5%).
4. Helix B>=32 envelope: prebuilt libTrtLlmGenFmhaLib picks a
   64-heads/CTA tile for 96 heads and hard-errors; exit = tp2 x dcp4
   hybrid (48 heads/rank).
5. b10 comm autotune structurally mis-ranks capture-destined rungs
   (eager sweep vs graphed reality flips sm/nccl); torch_symm one-shot
   gather rung added (best at tiny payloads).
6. **fc2-shard: built, correct, and REFUTED** — the size-dependent BW
   curve (6.4MB reads at ~0.8 TB/s vs 51MB at 2-3.5) eats the byte
   saving, and the terminal gather has no overlap partner. Default OFF,
   kept as documented negative result.
7. MPK: exact-semantics `kda_sconv` task (silu, per-seq state) built,
   registered (502), tests PASS; exact-semantics chain + 3-block
   assembly re-validated (~49us/block unchanged).

Scope: "focus on B<=64, apply TileRT ideas, keep trying ideas in order."
All numbers CUDA-graph (or in-megakernel) timed; nothing here is eager.
Nothing is committed — both `/node-storage/LLMDiveDeep` and
`/node-storage/mirage` carry uncommitted work.

## 1. MoE decode B<=64: solved by the TileRT layout (tp_te)

TileRT itself is closed-source (compiler unusable directly), but its layout
idea — keep every expert local, shard *inside* the expert, kill the A2A —
ported to the production `KimiK3MoE` as `moe_tp_size=8, moe_ep_size=1`:

- **tp_te wins every decode batch 8–256** (92–212 us for the MoE layer);
  production TP (EP + fused finalize-AR) and CP both lose there.
- Block4 E2E decode: tp_te best ≤256 (0.607 ms @ B=8); CP takes over ≥512
  and all prefill.
- Byte-floor check (measured achievable-BW controls): tp_te MoE is at
  ~1.0–1.2x the weight-traffic floor from B>=16 — **MoE is done**; only
  B=1–8 has ~3x latency-floor headroom, which no layout fixes.

Idea verdicts along the way: autotune bucket-matching REFUTED (no effect);
`ENABLE_B10_SHARD_FC1=1` segfault isolated (flag off, issue open);
"can CP beat TP at B=32 if all comm were free?" — **NO** (CP compute alone
1.257 ms > TP total 1.098 ms; tiny-M kernel geometry, not comm).
Tables/receipts: `parallel_strategy_search/moe/`, traces per case;
byte-floor scripts in `debug/`.

## 2. Attention region: MPK (Mirage megakernel) track

With MoE at floor, the 4-block attention region (~0.35 ms/replay at B=32,
~85 small kernels at 0.5–3.5 TB/s) is the remaining ~0.2–0.25 ms prize.
KDA is 3 of 4 blocks; the missing megakernel ingredient was the gated
delta-rule recurrence.

**Built it end-to-end** (details in `kernels/mpk_kda_task_plan.md`):

- Kernel `tasks/blackwell/kda_recurrence_sm100.cuh`: warp-per-row float4,
  fp32 128x128 state in place, gated-RMSNorm epilogue.
  **14.4 us / 3.51 TB/s at 32 tokens x 12 heads = at the byte floor**
  (v1 thread-per-row was 115 us there).
- Full MPK registration (enum 501, task_register, graph dispatch,
  runtime.cc name map, generic `kda_recurrence_layer` Python API packed
  into the 7-input/3-output limit) + standalone test + test-mode pipeline
  test: PASS (out diff 0.0, state 4.8e-7).
- Chain test (3x sconv -> recurrence -> o_proj in one graph): PASS.
  Profiled 8 serialized blocks: worker_kernel 128 us GPU →
  **~16 us per KDA block inside the megakernel** (conv + recurrence +
  o_proj, zero gaps). Re-measuring the eager block4 TP trace: the fork's
  `_fused_kda_decode_kernel` is also 16.0 us/block — and already fuses
  conv+recurrence+norm. Honest read: the recurrence was already at floor
  in eager; the megakernel's win in the attention region is absorbing the
  projection GEMMs, small norms/elementwise ops, and inter-kernel gaps
  (~0.35 ms region vs ~0.10–0.15 ms fused bound), which needs the full
  block assembly to demonstrate. Per-invocation wall is launch-bound
  (~130–155 us) — per-launch comparisons are meaningless; production MPK
  launches once per decode step.

Traps found (in memory + plan doc): `NUM_THREADS=128` constant vs
256-thread Blackwell workers (stride by `blockDim.x`); new tasks need the
`runtime.cc` `task_type_to_name` entry too; `setup.py build_ext` doesn't
track `libmirage_runtime.a` (touch `_cython/core.cpp` after C++ rebuilds).

- **Faithful-shape full-block assembly** (all K3 projections + conv +
  recurrence + o_proj, 3 blocks serialized in one graph):
  **~49 us per KDA block** inside the megakernel at tokens=32
  (~3.3 TB/s effective on ~162MB/block), vs ~87 us/block for the eager
  KDA-block region incl gaps → **~1.7–1.8x**, with headroom left in the
  linear tasks. Caveats: conv activation semantics approximated
  (inkling sconv), beta GEMM (0.17MB) skipped, no o_proj AR.

## 3. Next steps

1. MLA + MoE region in-MPK for the honest 4-block megakernel: MLA decode
   substrate EXISTS (mla_mtp_decode_tp8 16-head tile; pad K3's 12 -> 16,
   ~free since KV-bandwidth-bound); MoE needs MXFP4 expert tasks (the
   remaining kernel gap). KDA side is complete and correctness-grade.
2. Sunday resolutions: SHARD_FC1 segfault FIXED (device-ctx trap);
   dsv3_router_gemm cublas NOT reproducible (bench unstubbed); helix
   B>=32 root-caused to the prebuilt trtllm-gen tile heuristic (exit =
   tp2 x dcp4 hybrid, needs new plumbing — fork helix is pure-CP only);
   fc2-shard built and REFUTED (BW-vs-size curve). Parked (outside the
   small-batch mandate): cp_mm B256-512 weak window.
3. Nothing committed in LLMDiveDeep, mirage, or trt-llm. trt-llm delta:
   b10 device="cpu" x4 (segfault fix), perfect-router bypass fix,
   torch_symm gather rung, fc2_shard (default OFF, documented negative),
   bench gate unstub.
