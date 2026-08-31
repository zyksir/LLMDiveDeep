# Bug root causes (weekend debug log, 2026-08-30)

One entry per bug: symptom → bisect chain → root cause → classification
(idea-infeasible vs implementation bug) → fix → verification.

## 1. ENABLE_B10_SHARD_FC1=1 segfault (fc1-shard fast path)

**Idea under test:** shard the replicated `fc1_latent_proj` GEMM
([3584, 7168] bf16, ~49MB weight read per rank per layer) row-wise across
TP8 so each rank reads 1/8 of it, then reassemble the latent with a fused
gather(+MXFP8 quantize). Targets small-batch decode where that GEMM is
weight-bound.

**Symptom:** deterministic 8/8-rank SIGSEGV in
`async_copies(at::Tensor, at::Tensor, long)` (copy-engine dma mover) on the
first eager forward of `bench_k3_moe.py --modes tp` with
`ENABLE_B10_SHARD_FC1=1`; native-only stack, no python frame.

**Bisect chain (all under mpirun -np 8, trt-k3-bench):**
1. Explicit rungs (`seq` / `fused_col_quant` / `ce_gather_quant`, B=8):
   PASS → transports fine standalone.
2. Auto sweep on the fc1_aux pool under an aux stream, graph capture +
   replay, fork/join overlap capture: PASS → autotune/capture fine.
3. Instrumented bench (pointer logging): crash at the sweep's rows=16384
   bucket, first dma `_issue`, all pointers inside valid symm windows.
4. Module context, gather alone on main stream (V2): CRASH → overlap
   schedule and real quantizer exonerated.
5. Residency splits: 18GB memory hog PASS, trtllm AllReduce+MoEAllReduce
   PASS, ordering PASS; full module + dummy quant CRASH.
6. Build stages: **bare `KimiK3MoE.__init__` (B1, no weights) crashes**;
   then EVERY per-construction stage (even a lone shared GatedMLP)
   crashed → the armer is in the common preamble, not any construction.
7. Preamble diff vs passing isolates: `torch.device(cuda:rank).__enter__()`
   — the ambient device context the bench enters because MoE builds
   require it (see trtllm-device-ctx-build-trap).

**Root cause (implementation bug, NOT idea-infeasible):**
`LowContentionComm.__init__` allocated its host-side pointer scratch as
`torch.empty(8, dtype=torch.int64)` with no device. Under the ambient CUDA
device context that lands on GPU; `async_copies` dereferences
`data_ptr<int64_t>()` on the HOST → segfault. Proven directly:
`torch.empty(8, dtype=torch.int64).device` is cpu normally, cuda:0 inside
the ambient ctx. Standalone comm tests never enter the ambient ctx, which
is why every isolation passed and the bug masqueraded as a deep IPC issue.

**Fix:** explicit `device="cpu"` on all host-read pointer tables (4 sites:
`copy_engine.py` `_dst_ptrs`/`_src_ptrs`; `col_quant.py` `_ptrs_cpu` and
the `ce_copy2d_batch` argument).

**Verification (DONE):** isolate C1 passes; full `bench_k3_moe --modes tp
--check` with SHARD_FC1=1 passes correctness at every size
(close_fraction 1.0, cosine 0.99999). CUDA-graph A/B (only the flag
flipped, ENABLE_B10_COLLECTIVES=1 in both):

| decode B | shard off | shard on | gain |
|---|---|---|---|
| 8  | 124.2 us | 111.3 us | **+10.4%** |
| 32 | 189.8 us | 179.6 us | **+5.3%** |
| 64 | 278.0 us | 290.1 us | -4.3% |

**Verdict: the idea is VALID for its target (small-batch decode)** — the
gather cost overtakes the 8x fc1 weight-read saving as rows grow, so it
should stay gated to small T in tp mode (a batch gate at ~<=48 captures
the win and avoids the B=64 regression there).

**Stacked on tp_te (the regime-map winner), it's a clean win everywhere:**

| decode B | tp_te | tp_te + fc1shard | gain |
|---|---|---|---|
| 8  | 95.8 us | **77.3 us** | **+19.3%** |
| 32 | 120.3 us | **107.7 us** | **+10.5%** |
| 64 | 153.1 us | **140.8 us** | **+8.0%** |

correctness clean (close_fraction 1.0 all sizes). tp_te+fc1shard is the
new best-known MoE decode configuration for B<=64.

**Block4 E2E validation (4 transformer blocks, tp_te decode, graphed):**
B=8: 657 -> 628 us (+4.4%), B=32: 837 -> 815 (+2.6%), B=64: 1073 -> 1042
(+2.9%) — the MoE-layer win survives at the E2E level.

**4-layer E2E, production baseline vs full new stack (2026-08-30, same
bench state: real gate, graph-timed, B=1..128):** baseline = prod wiring
(EP experts + TP elsewhere, b10 paths off); new stack = tp_te layout +
fc1-shard (+collectives facade on).

| B | prod E2E | new stack E2E | gain |
|---|---|---|---|
| 1 | 0.580 ms | 0.583 ms | ~tie |
| 8 | 0.792 ms | 0.634 ms | **+19.9%** |
| 32 | 1.118 ms | 0.811 ms | **+27.5%** |
| 64 | 1.586 ms | 1.043 ms | **+34.2%** |
| 128 | 2.433 ms | 1.450 ms | **+40.4%** |

(new-stack points reproduce the earlier A/B within noise: 634/811/1043 vs
628/815/1042.) Logs: /node-storage/var/block4_{prod_base,new_stack}.log.
**SUPERSEDED 2026-08-31: this table was measured with the production
multi-stream overlap silently OFF (bench_block4 lacked the
with_multi_stream wrap — user-caught in review; REVIEW.md 1b). The
corrected table (both configs faster by ~70-100us; gains grow to
+12.4/+25.0/+30.8/+41.8% at B=1/8/32/128) is in REVIEW.md section 1;
logs /node-storage/var/ms_{prod,new}.log.**

**Complete regime map (tp_te + real gate, MoE layer, graphed, checks
clean at every size):**

| B | shard off | shard on | gain |
|---|---|---|---|
| 1 | 94.7 us | 77.5 us | **+18.1%** |
| 2 | 99.9 us | 79.4 us | **+20.5%** |
| 4 | 87.6 us | 81.6 us | +6.8% |
| 8 | 95.7 us | 91.1 us | +4.8% |
| 32 | 124.1 us | 118.3 us | +4.7% |
| 64 | 161.3 us | 155.2 us | +3.8% |
| 128 | 222.7 us | 217.3 us | +2.4% |
| 256 | 300.5 us | 314.8 us | -4.8% |

Wins everywhere <=128 (largest exactly where the small-batch mandate
lives, B=1-2, where the fc1 GEMM is pure latency); the crossover at
B>128 coincides with `_PAIR_MAX_TOKENS = 128` — above it the fused
quantized-pair gather rung stops and the bf16 path runs. Production
recommendation: gate fc1-shard to <=128 tokens, which the code's own
rung boundary already expresses.

**Real-gate correction (same day, after bugs 2+3 below):** with the
production gate restored and the router-fairness fix, the honest stack is
**+4.8/+4.7/+3.8% (91.1/118.3/155.2 us at B=8/32/64 on tp_te)** — the
+19% above was inflated because the stub gate made the gather's overlap
window free. Still a clean, consistent win; final numbers use the real
gate.

## 2. dsv3_router_gemm CUBLAS_STATUS_NOT_SUPPORTED — NOT REPRODUCIBLE

The bench had stubbed the production gate over a historical
CUBLAS_STATUS_NOT_SUPPORTED "in-module". Evidence chain (2026-08-30):
standalone gate forward passes with AND without the ambient device ctx;
the real gate swapped into the fully built K3 module passes eager
forwards at B=8/32/64, the autotune context, and CUDA-graph capture +
replay under multi-stream (debug/dsv3_gate_inmodule.py, mpirun -np 8).
Conclusion: not reproducible in the current environment (likely fixed by
intervening env/driver/library changes). The stub is REMOVED from
bench_k3_moe.py — the gate GEMM is back in timing with deterministic
weights.

## 3. fc1-shard run_moe bypass skipped the perfect-router rewrite

**Symptom:** with the real gate, fc1-shard flipped from winning to 2x
LOSING (124/221/313 us). **Root cause:** `_maybe_get_perfect_router_logits`
is applied inside `MoE.forward` only; the fc1-shard fast path calls
`backend.run_moe` directly and skipped it, so the shard path routed the
bench's random logits raw (imbalanced expert load) while the baseline
routed perfectly — an unfair comparison, and a contract violation of the
bypass's own "exact op invocation of stock forward" docstring.
**Classification:** bench-visible implementation gap; inert in production
(the env var is bench-only). **Fix:** apply the rewrite in
`_routed_experts` before `run_moe` (modeling_nemotron_h.py). Verified:
shard wins again with the real gate (+4-5%, table above).

## 4. Helix DCP `Internal error numHeadsQ=96, numHeadsPerCta=64` at B>=32

**Symptom:** the DP-Attention-Ring (helix DCP) variant fails at B>=32 (and
B=8 @128K) with `[E] [FmhaAutoTuner.cpp:59]: Internal error numHeadsQ=96,
numHeadsPerCta=64, numCtasForAllHeads=1`.

**Root cause (proven to the boundary we can reach):** under helix DCP the
per-rank MLA keeps ALL 96 q-heads (DCP shards the KV/sequence dim, not
heads; numKvHeads=1 for MLA). The trtllm-gen MLA generation kernel is
selected by `FmhaAutoTuner` inside the PREBUILT
`kernels/trtllmGenKernels/fmha/lib/x86_64-linux-gnu/libTrtLlmGenFmhaLib.a`
— only headers ship (`trtllmGen_fmha_export/`); `FmhaAutoTuner.cpp` is not
in the repo. Its batch-dependent heuristic switches to a
64-heads-per-CTA tile once B crosses its threshold; 96 % 64 != 0 and the
library hard-errors instead of falling back (numCtasForAllHeads=96/64=1
truncated). The shipped `checkFmhaOptions` allows >64 heads only via
multi-CTA distribution, which that tile then fails to satisfy.

**Classification: third-party kernel-envelope gap** — not our
implementation bug, and not proof the ring idea is infeasible. Two exits:
(a) hybrid geometry tp2 x dcp4 (48 q-heads/rank <= 64 fits every tile) —
the production-relevant way to run helix at B>=32; (b) an NVIDIA-side fix
in the prebuilt autotuner. (a) is benchable with MLA tp_size=2 inside the
helix variant; queued as a follow-up experiment.

## 4b. DCP-MLA as prod baseline: measured at ISL 4096 (2026-08-30 night)

User direction: prod will run DCP on MLA (sglang-style), so the E2E
baseline should too. Added a plain-TP variant to bench_dp_attention and
measured the MLA layer GRAPH-timed at ISL 4096, W=8 (the earlier eager run
was 1.0-1.3ms flat — overhead-dominated, discarded per the CUDA-graph
rule):

| B | TP (12 heads/rank + o_proj AR) | DCP/helix | BS-SPLIT |
|---|---|---|---|
| 8 | **0.094 ms** | 0.143 | 0.120 |
| 16 | ~0.105 | 0.155 | ~0.124 |
| 32 | **0.118** | envelope | 0.128 |
| 64 | **0.141** | envelope | 0.128 |
| 128 | 0.206 | envelope | **0.138** |

**Mechanism:** DCP/BS-SPLIT keep all 96 q-heads per rank, so q_b/kv_b
projection weights are REPLICATED (8x the projection bytes TP reads); at
ISL 4K the KV read is small (38MB/rank at B=8) so the projection tax
dominates and DCP loses ~50% at small B. DCP pays off at long ISL (KV
bytes dominate: ~1.2GB/rank at B=8/131K in TP) and for KV capacity —
the sglang motivation. tp2 x dcp4 would halve the projection tax AND fix
the B>=32 envelope.

Composed DCP-baseline E2E (block4 prod + [helix - tp] on the one MLA
block): B=8: 0.841ms (vs 0.792 TP-MLA) → new stack (0.634) gains +24.6%
vs the DCP baseline. Full in-block4 DCP integration = next validation
step. Receipts: attention/local_results/gmla_*_w8_isl4096.json.

**ISL 32768 (graph-timed, same harness):**

| B | TP | DCP/helix | BS-SPLIT |
|---|---|---|---|
| 8 | 0.137 ms | **envelope-FAIL (even at B=8!)** | **0.130** |
| 32 | 0.292 | envelope | **0.151 (1.93x over TP)** |

Two consequences: (a) the trtllm-gen 64-head tile heuristic binds on KV
LENGTH too, so on this kernel stack helix-DCP cannot run in exactly the
long-context regime it exists for — tp2 x dcp4 (48 heads/rank) is a hard
prerequisite for prod DCP here; (b) BS-SPLIT already delivers the
long-context attention win (2x at B=32/32K) with zero comm and no
envelope issue — the strongest available baseline/candidate for long-ISL
decode on this stack. Receipts: gmla32k_*.json.

## 5. b10 collectives autotune mis-ranks capture-destined transports

**Evidence** (all_gather_cols at [B<=64, 896] bf16, world 8):

| rung | eager (sweep's view) | CUDA-graph (production's view) |
|---|---|---|
| copy_engine:dma | 113 us | 50 us |
| copy_engine:sm | 74 us | **16-17 us** |
| nccl | **34 us** | 23-24 us |
| torch_symm (added today) | 29 us | 12-16 us |

The sweep measures EAGER first-sightings (by design — capture-safe), but
at tiny payloads eager cost is host-launch dominated and the RANKING flips
(sweep picks nccl over sm; graphed sm is 30% cheaper). **Structural bug
class: eager-measured winners are wrong for graph-replayed streams at
latency-floor payloads.** Fix direction: sweep should measure captured
replays for capture-destined streams (or at least re-rank under a
throwaway graph); until then, latency-critical callers must pin rungs.
Also added the missing one-shot rung: `torch_symm`
(multimem_all_gather_out) is the best gather at these sizes and was not
wired into `all_gather_cols` at all.

## 6. fc2-shard (ENABLE_B10_SHARD_FC2): idea REFUTED by the
## bandwidth-vs-size curve + tiny-N GEMM geometry

**Idea:** mirror fc1-shard on the output projection — fc2_latent_proj
([7168, 3584] bf16, ~49MB) runs replicated post-AR, so shard output
columns 8x + all_gather_cols([B, 896]).

**Built it** (b10_kernels/fc2_shard.py + wiring; correctness PASSES at
every size). Measured on tp_te+fc1shard baseline (91.1/118.3/155.2 us):
auto rung -19/-22/-17%; pinned sm rung -3/-10/-9%; pinned torch_symm
-0.4/-10/-8%. Trace diff at B=32 (per replay): full fc2 GEMM removed
-20.6 us; shard GEMM +24 us (!); multimem AG +10.6; stage/transpose ~+4.

**Why the shard GEMM costs 24 us for 6.4MB (~5x its naive floor), proven:**
(a) standalone probes are invalid — 51MB fits GB300 L2, so a repeated
standalone GEMM reads L2 (5.7 us) while the in-bench replay walks ~500MB
of expert weights and evicts everything; (b) the measured achievable-BW
curve (byte-floor control: ~0.5-0.85 TB/s at 5-10MB vs 2-3.5 TB/s at
25-50MB) means cutting a weight read 8x in BYTES only cuts ~2x in TIME.
The sharding gain is eaten by the size-dependent bandwidth cliff, and the
mandatory gather sits terminally on the critical path (no overlap partner
— unlike fc1's gather, which hides behind the gate).

**Related idea considered and dropped without an experiment (reason
recorded):** rerouting the single-node fused finalize-AR (23.8 us at B=8)
through the b10 moe_finalize rung — the b10 backend runs the SAME stock
compiled kernel on a fabric workspace (its win case is multi-node where
the stock IPC workspace cannot exist); on this node there is no second
kernel to race, and the 23.8 us is the kernel's own cross-rank sync floor
at tiny payloads.

**Verdict: idea-infeasible at this granularity on this memory system** —
same root curve that refuted small-M GEMM and commless-CP. It could only
be revived by a fused GEMM+gather epilogue kernel that streams the full
weight at large-read bandwidth on ONE rank... which is just the baseline.
ENABLE_B10_SHARD_FC2 stays default OFF; code kept as the documented
negative result. Tiny-B re-check (fc2 is the largest single kernel at
B=1, 29.6us): B=1 +2.7%, B=2 -2.5%, B=4 -2.2% — noise-level wash, the
verdict holds at every measured B (fc2shard_tiny receipt).
