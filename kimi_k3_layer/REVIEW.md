# REVIEW INDEX — decode small-batch campaign (for the 2026-08-31 review)

Everything below is CUDA-graph timed, trace-recorded, and reproducible
without GPUs from the files named. Decode focus B=1..16 per direction
(tables extend to 128 where measured). Companion docs: WHY.md (mechanism
ledger + waterfall), BUG_ROOTCAUSES.md (7 root-caused entries),
kernels/cute_w4a8/RESULTS.md (kernel tables), stretch_summary_2026-08-30.md.

## 1. Headline: 4-block E2E decode, prod baseline vs new stack

Baseline = production wiring (EP experts + TP attention/others, b10 off).
New stack = tp_te expert layout + fc1-shard (real gate, fair routing).
**CORRECTED 2026-08-31 during review: bench_block4 was missing the
`with_multi_stream(True)` wrap around graph capture (trap #6; caught by
the user in the trace — only 2 stream tracks). Fixed; table below is the
multi-stream-honest one. Mechanism + predictions in section 1b.**

| B | prod E2E | new stack | gain | trace pair |
|---|---|---|---|---|
| 1 | 0.558 ms | 0.489 | **+12.4%** | ms_decode_tp{,_te}_b1_* |
| 4 | 0.583 | 0.511 | +12.4% | |
| 8 | 0.712 | 0.534 | **+25.0%** | |
| 16 | 0.819 | 0.597 | **+27.1%** | |
| 32 | 1.030 | 0.713 | **+30.8%** | ms_decode_tp{,_te}_b32_* |
| 64 | 1.519 | 0.962 | +36.7% | |
| 128 | 2.361 | 1.375 | **+41.8%** | |

(The earlier sequential-shared table — 581/671/792/901/1118/1586/2433 vs
582/589/634/692/811/1043/1450 — is kept as the overlap-worth datum: the
production multi-stream schedule is worth ~70-100us per replay to BOTH
configs. Old block4_decode_* traces = sequential-shared variants.)

### 1b. The multi-stream bug: reason chain (user-caught in review)

- Root cause: duplicated capture logic — bench_k3_moe carried the
  `with_multi_stream(True)` wrap (trap #6), bench_block4 got the
  production MoE later and never inherited it. `do_multi_stream()` is a
  flag the harness must set; without it `maybe_execute_in_parallel`
  runs sequentially.
- Why the old traces still showed 2 streams: with the flag OFF, the
  fc1-shard code takes its explicit `_eager_fc1_shared_overlap` branch
  (manual fork onto the fc1 aux stream) — but middle-replay analysis
  shows that branch achieved only ~55us of real overlap at B=1.
- Predictions tested: (P1) stream tracks 2 -> 4-5 ✓; (P2) both configs
  faster by ~70-100us ✓; (P3) A/B direction unchanged, gains grew ✓.
- The B=1 asymmetry (prod -23us vs new -93us from the fix), attributed:
  with the flag OFF the two configs take DIFFERENT branches — prod is
  fully sequential (1 stream, wall 617us = main busy), tp_te's eager
  branch pseudo-overlaps (~55us). With the flag ON, prod hides only part
  of its 137us shared work (Amdahl: the B=1 main stream is a chain of
  short latency-bound kernels + 8 AR sync points, wall 572), while tp_te
  gets the full production schedule (shared 152 + fc1 45 off-main, wall
  493). Middle-replay walls from the four B=1 traces: 617/594 -> 572/493,
  matching the measured medians.

## 2. The waterfall (B=32): every E2E microsecond attributed

See WHY.md "THE WATERFALL". Summary: Δwall −278us = expert GEMM1 −216.2
+ expert GEMM2 −68.4 + fc1 path −26 (gather overlapped) ± noise; kernel
diff reconciles with standalone kernel measurements within 2% and with
the E2E wall within 3%. Trace pair:
block4_decode_tp_b32_isl4096_w8_graph_rank0.json vs
block4_decode_tp_te_b32_isl4096_w8_graph_rank0.json.
Artifact rule: lamport/AR rows in traces absorb rank skew under the
profiler — composition comes from traces, totals from barrier-synced
medians.

## 3. Strategy regime map, with the why per boundary (RoPE-style)

**Experts layout: tp_te wins B>=2, exact tie at B=1 (both at the 25.7us
expert-op latency floor — measured, not asserted).**
Mechanism proven two ways: equal bytes but EP concentrates them in
~distinct/8 wide groups → CTA-parallelism starvation (standalone: EP
GEMM pair 71.5us vs tp_te 23.4 at B=16, same bytes — sol_ep.log); plus
skew immunity (EP +3.6-5.6% under mild skew, tp_te <0.5% — skew_ab).
ONE strategy, no threshold. RECOMMENDED unconditionally for decode.

**fc1-shard: wins B=1..128 (+18-21% layer at B=1-2, +2-5% at 8-128),
loses past the _PAIR_MAX_TOKENS=128 fused-gather window.** ONE strategy
for the decode range; the >128 gate is the code's own rung boundary, not
a new threshold. RECOMMENDED for decode.

**Expert GEMM kernels (the one place a real regime split exists):**
- B<=8: trtllm-gen chain wins — latency-first t128x8 tiles run 2-8us per
  kernel; every CuteDSL persistent kernel carries an ~11-13us floor
  (tile scheduler + TMA warmup) regardless of work (fused halves at
  B=1-8 shapes: pair 22-24us vs whole trtllm expert-op 25-30us INCL
  routing/quant/finalize).
- B=16: tie (pair 25.9+routing ~= 39.1 wall).
- B>=32: CuteDSL fused halves win, growing to 1.64x (31.4 vs 51.5us at
  B=32) — throughput-first 128x128 tiles + launch elimination.
This is the canonical latency-first vs throughput-first split. Options:
(a) accept it via measured AutoTuner tactics (never a hardcoded B);
(b) the ONE-KERNEL path: a latency-first expert megakernel
(routing→G1→SiTU→G2→finalize in one launch, TileRT's ExpertSelect design
— their measured 10.4us at DSv3.2 shape is the existence proof; our MPK
substrate is the alternative route). Since the user's range is B<=16
where trtllm already wins, the CuteDSL kernels are NOT needed for the
current target — record them as the B>=32 upgrade + megakernel evidence.

**MLA attention:** TP wins at ISL 4096 for B<=64 (0.094ms vs helix 0.143,
bs_split 0.120 at B=8) — DCP/BS-SPLIT replicate q_b/kv_b projections (8x
bytes) and KV is small at 4K. BS-SPLIT wins at long ISL (1.93x at
32K/B=32) where KV bytes dominate. Helix-DCP additionally cannot run
B>=32 OR ISL>=32K on this stack (prebuilt trtllm-gen 64-head tile —
BUG_ROOTCAUSES 4/4b); tp2 x dcp4 is the prerequisite if prod requires
DCP. For decode B=1-16 @ 4K: PLAIN TP, one strategy.

## 4. Negatives (all mechanism-proven, none "just didn't work")

- **fc2-shard**: correct but loses (BW-vs-size curve eats the byte
  saving; terminal gather has no overlap partner) — BUG_ROOTCAUSES 6.
  B=1-4 re-check CLOSED: +2.7/-2.5/-2.2% at B=1/2/4 — wash; refuted at
  every measured B (fc2shard_tiny receipt).
- **PDL (TRTLLM_ENABLE_PDL=1)**: no effect +-1% at B=1-16 — under graph
  replay the gaps PDL hides are already minimal (pdl_on/off receipts).
- **moe_as_dense_gemm**: inapplicable at K3 scale by ledger — dense
  all-experts read = 1.2GB/rank vs 35-68MB routed at B<=16.
- **Autotune bucket matching, commless-CP, b10 SHARD_FC1 as-was**: see
  BUG_ROOTCAUSES / stretch summary.
- **MXF4-recipe ICE** in the swiglu-fusion CuteDSL kernel (nvfp4 recipe
  works) — open DSL bug, logged.
- **W4A8 mixed CuteDSL**: hardware needs fp4-in-byte containers in smem;
  packed weights require an unpack pipeline stage (multi-day) —
  cute_w4a8/RESULTS.md.

## 5. Kernel-level tables

cute_w4a8/RESULTS.md: GEMM2 per-B dispatch table (B=1..128, both kernels,
correctness PASS every row), GEMM1 at-floor table, fused halves, SOL
ledger (sol_experts2.log, sol_ep.log = EP twin).

## 6. B=1 MoE-layer decomposition (77.5us wall; b1_floor trace)

Real kernels (spin rows excluded): fc2_latent nvjet 29.6 (51MB at
~1.7TB/s small-read BW — the LARGEST single item), finalize-AR 22.5
(cross-rank sync floor for a 3.6KB payload — ~10us above a one-shot AR
floor; an epilogue-AR rework is the identified-but-unbuilt lever),
shared-expert cutlass 22.3, expert bmm pair 18.8, routing 10.0,
fc1-shard 5.2 + overlapped gather, quant 5.1. Busy sum ~130us >> 77.5
wall = heavy multi-stream overlap already. The remaining tiny-B prizes,
in order: fc2 read (29.6), finalize-AR floor (~10), routing (10) — all
three are megakernel/epilogue-fusion territory, consistent with the MPK
track. MPK grounding at THIS range (mpk_t{1,8,16}.log, 3-block faithful
KDA assembly, worker-kernel GPU per block): **42.1 / 41.6 / 49.0 us at
tokens=1/8/16** (49.3 at 32) — flat, weight-bound, no launch floors; the
persistent-kernel design holds its advantage exactly where the separate-
kernel stacks pay their latency floors.

## 7. File map

- Docs: WHY.md, BUG_ROOTCAUSES.md, REVIEW.md (this), stretch_summary,
  kernels/cute_w4a8/RESULTS.md, kernels/tilert_results.md,
  kernels/mpk_kda_task_plan.md.
- Receipts: parallel_strategy_search/moe/local_results/*.json (fc1edge,
  realgate, fc2shard_*, pdl_*, skew_ab, b8/b1/b16_floor);
  attention/local_results/gmla_*, gmla32k_*; block4_demo/local_results/*.
- Traces: block4_demo/.../traces/ (per-B pairs), moe/.../traces/
  (b1/b8/b16_floor, fc2trace pair).
- Logs of every bisect/experiment: /node-storage/var/*.log.
- Code: b10 fixes in trt-llm (device=cpu x4, router bypass, torch_symm
  rung, fc2_shard default-off, TpVariant in bench_dp_attention);
  cute_w4a8/ candidate+harness; mirage KDA tasks 501/502 + tests.
