# Kimi-K3 decode layers: TRT-LLM baseline vs b10 optimizations

Layer-level decode benchmarks on B200 inside the `trt-dev` container
(official `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23`): the REAL
TRT-LLM modules as baseline, plus a b10-optimized subclass with one
switch per optimization - every claim below is a measured ablation.

**Production alignment**: the deployed build is the fork branch
`aaryams/hybrid_cache_fixes @ 496bc01f8e` (1.3.0rc19 base; the local
checkout is that commit + a few later KDA sync fixes). The baseline
here matches it structurally AND at the kernel level - see
"Production-alignment notes" at the end for exactly what was matched
(FlashInfer op backend incl. split routing kernels, fused
latent+shared AR tail, AllReduce construction) and the two knowingly
accepted deviations (in-kernel SiTU epilogue lives in a private
FlashInfer cubin pool we don't have - Swiglu epilogue is
cost-identical; MNNVL AR does not exist on this single-node box).

```bash
# TP8 MoE (production trtllm-gen backend is the default)
docker exec trt-dev bash -c "cd /workspace/diffusion_inference/LLMDiveDeep && \
  mpirun -n 8 --allow-run-as-root python3 kimi_k3_layer/bench_moe_kimi_k3.py --ablate"
# KDA (single GPU), comm kernels, routing kernel
... bench_kda_kimi_k3.py        ... bench_comm.py        ... bench_routing_kimi_k3.py
# traces are built into the benches:  --profile 64
```

Correctness: seed-identical global weights sliced per rank; every
configuration compared against the baseline forward on the SAME
module instance. Rel-err is <=1e-2 at B<=4; at B>=8 the max-rel-err
metric reads ~0.2 because bf16 logits flip top-16 NEAR-TIES on single
tokens - the `-all` control below (all switches off ~= baseline
re-implemented) shows the same value, so it is a tie-break artifact
of random weights, not an optimization bug.

## TL;DR

TP8 B200, CUDA-graph replay, max over ranks, 100 iters; repeat runs
agree within +-2 us (`results/bench_moe_kimi_k3_tp8.{csv,md,png}`,
logs in `results/experiments/`).

**MoE layer** (production stack: `TRTLLMGenFusedMoE`,
`W4A8_MXFP4_MXFP8` = MXFP4 weights + MXFP8 activations, the published
K3 QAT recipe; SiTU shared expert):

| B | baseline us | b10 us | improvement |
|---|---|---|---|
| 1 | 108.4 | 80.7 | **+26%** |
| 2 | 114.0 | 79.3 | **+30%** |
| 4 | 118.5 | 84.2 | **+29%** |
| 8 | 125.8 | 105.3 | **+16%** |
| 16 | 133.3 | 116.8 | **+12%** |
| 32 | 149.7 | 143.3 | **+4%** |
| 64 | 163.8 | 160.6 | **+2%** |

(`a6_multistream_base_sweep.log`. These are against the
PRODUCTION-ALIGNED baseline with BOTH production fast paths on: the
FlashInfer split routing pipeline (cut the old wheel's routing stage
from 41-46 us to ~26 us) AND `with_multi_stream(True)` during graph
capture, which lets `maybe_execute_in_parallel` fork the shared
expert onto its aux stream exactly like production's
cuda_graph_runner (another ~10-15 us off the baseline). Margins are
honest, not inflated by a crippled baseline. B=128/256 rows pending
a re-run; old-baseline values were +28%/+25%.)

**KDA layer** (`b10_fused` CuTeDSL kernel vs production `trt_fused`):
tp8 B=1/8/32: 56.6->47.8 / 59.9->50.8 / 67.2->62.2 us
(**+16/+15/+7%**); tp1 B=32: 333.8->276.1 (**+17%**). A KDA+MoE
decode block at B=8/TP8 goes ~221 -> ~155 us, **~+30% end-to-end**.

Why the gain shrinks with B: the opt savings are mostly FLAT
microseconds (routing stage cut, launch/glue, comm) so they shrink
relatively as the expert group-GEMMs grow (~27 us at B=8 -> ~50 at
B=64, identical in both paths). On top of that the baseline's own
shared-expert overlap (multi-stream fork) hides more of its shared
chain as B grows, which is exactly the mechanism our overlap switch
uses - so at B>=32 the two paths converge and the remaining edge is
the sharded fc1/fc2 + routing handoff.

## Status (Aug 5): REF-overlap tail - PENDING final e2e validation

The TL;DR table above PRE-DATES the newest tail; the final numbers
come from `run_final_experiments.sh` (below). What changed since:

**New default tail from B>=4 (`ref_tail`, REF_TAIL_MIN_TOKENS=4).**
Insight: after a fused AR+RMSNorm every rank holds the IDENTICAL full
normed latent, so the FULL-weight fc2 output is identical on every
rank and needs NO output collective at all. The only reduction left
is the shared-expert partial, and that AR(7168) is independent of the
whole latent chain -> it runs on the aux stream (chained after the
shared down-GEMM inside `_fork_shared`), hidden under the expert bmms
/ AR+norm / fc2 GEMM. The shared add rides the fc2 GEMM epilogue
(`addmm`), no separate elementwise pass. Clean fair probe
(`tmp_tail_ref_overlap.py`, both arms on the SAME FlashInfer AR+norm
primitive, us, graph replay, max over ranks):

| B | AR+shard tail | REF+overlap tail | fat AR (baseline) |
|---|---|---|---|
| 1 | 22.5 | 24.1 | 25.4 |
| 2 | 22.6 | 23.2 | 32.2 |
| 4 | 22.8 | **22.2** | 31.5 |
| 8 | 25.3 | **20.9** | 31.0 |
| 16 | 30.5 | **24.2** | 34.6 |
| 32 | 35.5 | **28.7** | 41.0 |
| 64 | 44.2 | **40.7** | 55.5 |

The shard tail only holds at B<=2 and by <2 us; REF wins from B=4 up
and is also the numerically cleanest (no split-K bf16 rounding).

**Custom RS+norm is retiring.** The clean `-customcomm` ablation
(`m6_ablate_clean_gpus.log`) shows FlashInfer AR+norm at parity or
better e2e at every decode size, and the probe above confirms the
kernel-level RS edge is only 0.3-0.5 us. With ref_tail on from B=4,
the custom RS only serves B<=2 - after the pending validation it gets
deleted (along with `rs_defer`/`scale_deferred`) and the sub-crossover
tail becomes AR+norm -> sharded fc2 -> AR. The custom kernel that
STAYS is the fc1-shard AG with fused MXFP8 quantize - FlashInfer has
no equivalent and it is a clear win.

**Shared experts must stay bf16**: the `moonshotai/Kimi-K3`
checkpoint's quantization ignore-list explicitly excludes
`re:.*shared_experts.*` (plus attention / lm_head), so an fp4 shared
GEMM would break production faithfulness. The ~100 MB shared weight
read is a hard cost; overlap is the only legal mitigation.

**Pending validation (run on an idle 8-GPU node):**

```bash
docker exec trt-dev bash -c "cd /workspace/diffusion_inference/LLMDiveDeep && \
  bash kimi_k3_layer/run_final_experiments.sh" 2>&1 | tee /tmp/final.log
```

Produces, in order: (1) the fair crossover probe, (2) the tail-shard
strategy probe, (3) e2e crossover extremes (ref tail ALWAYS vs NEVER
- confirms the aux-stream shared AR does not perturb the expert
kernels in-pipeline, and fixes REF_TAIL_MIN_TOKENS), (4) base-vs-opt
traces at B=1,2,4,8,16 (`results/*.trace.json`), (5) the full
ablation LAST so `results/bench_moe_kimi_k3_tp8.{csv,md,png}` are
regenerated from the newest code. Old results/traces were deleted -
anything present was produced by the current code. To read the
traces: opt should show the merged [gate|fc1-shard] GEMM + AG(+quant)
input stage, the shared chain + its AR on the aux stream with no
exposed tail collective after fc2 (B>=4), and a shorter replay span
than base; no long spin-wait kernels outside warmup.

---

# Part 1 - how the baseline works

Baseline = real `NemotronHMOE` parameterized as `KimiK3MoE`
(`moe_trtllm_kimi_k3.py`): 896 routed experts (EP-sharded over TP8,
top-16, latent 3584, intermediate 384), TP-sharded shared expert,
Stable-LatentMoE norm, one fused TP reduction. Expert quant is the
published K3 QAT recipe, `W4A8_MXFP4_MXFP8`: MXFP4 weights (fp4 e2m1
packed 2/byte + ue8m0 per-32 block scales) x MXFP8 activations (fp8
e4m3 + ue8m0 scales, quantized at runtime). Weights are
random-initialized at a sane scale (scale byte 124 = 2^-3) - no
checkpoint needed, every kernel's cost is preserved. Expert + routing
kernels come from FlashInfer's trtllm-gen cubin pool (the production
op backend; see the alignment notes). One steady B=64 iteration
(~178 us), function -> kernel (routing/quantize/GEMM entries
re-measured from the aligned B=8 trace, scale-invariant at decode
sizes):

| # | stage (Python) | kernel | us | idea |
|---|---|---|---|---|
| 1 | `gate` (`DeepseekV3Gate` Linear) | `nvjet..._splitK` + reduce | 9+5 | router GEMM [64,7168]x[7168,896] -> fp32 logits |
| 2 | `fc1_latent_proj` Linear | `nvjet...TNT` | 18 | hidden -> latent [64,3584]; FULL 51 MB weight on every rank |
| 3 | experts: input quantize | `quantize_with_block_size` | 6 | wrapper `quantize_input`: bf16 latent -> MXFP8 (e4m3 + ue8m0 block scales) |
| 4 | experts: IN-KERNEL routing | `routingIndicesBlockScoresKernel` + `routingIndicesClusterKernel` | **13+13** | the production SPLIT pipeline: kernel 1 does sigmoid + fp32 bias + top-16 of 896 from raw logits, kernel 2 the post-top-k permute bookkeeping (the rc23 wheel's native op ran ONE monolithic 41-46 us cluster kernel instead - that is NOT what production runs) |
| 5-6 | experts: grouped GEMMs | `bmm_MxE4m3_MxE2m1MxE4m3` + `bmm_Bfloat16_MxE2m1MxE4m3` | 27+22 | the MXFP4xMXFP8 idea: fp4 weights = 4x less weight HBM read (the dominant decode cost), fp8 activations on fp8 tensor cores; GEMM 1 emits MXFP8 + SwiGLU epilogue, GEMM 2 emits bf16; PDL pipelines GEMM 2 into GEMM 1's tail |
| 7 | experts: finalize | `finalizeKernel` | 9 | scale by routing weights, un-permute -> [64,3584] partial |
| 8-10 | shared expert (`GatedMLP`) | splitK GEMM, `_situ_and_mul`, GEMM | 11+6+2+7 | gate/up -> SiTU -> down, [64,7168] partial; `maybe_execute_in_parallel` forks this chain onto the aux stream during graph capture (production's cuda_graph_runner captures under `with_multi_stream(True)`; our `capture()` replicates it), so it overlaps the routed-expert stage |
| 11 | pack | `CatArrayBatchedCopy` + DtoD | 3+2 | cat [latent \| shared] = [64, 10752] message |
| 12 | allreduce | `ncclSymkDevKernel_AllReduce...` | 26 | ONE fused TP8 AR (strategy AUTO -> NCCL_SYMMETRIC); also absorbs post-expert rank skew |
| 13 | `latent_norm` | flashinfer `RMSNormKernel` | 4 | Stable-LatentMoE norm on the reduced latent |
| 14 | `fc2_latent_proj` Linear | `nvjet...TNT` | 14 | latent -> hidden; FULL 51 MB weight, replicated on every rank |
| 15 | add | `elementwise_kernel` | 5 | shared + routed |

Trace-reading notes: skip the first ~2 iterations (the first
collective absorbs profiler-start skew); a collective's duration =
wire + waiting for the slowest rank, not pure comm.

The KDA baseline (`kda_trtllm_kimi_k3.py`) grafts the unreleased
production modules from the checkout: `trt_fused` = ONE Triton kernel
for conv + KDA recurrence + gated RMSNorm, between the in/out
projection GEMMs and AttnRes.

# Part 2 - our optimizations, one experiment per switch

`KimiK3MoEB10` (`moe_b10_kimi_k3.py`) keeps weights and semantics,
swaps the forward. Every optimization has a switch (`set_opt_flags`);
`bench_moe_kimi_k3.py --ablate` measures full-opt plus each switch
off. Production-ALIGNED sweep (`a7_multistream_ablate.log`, us;
baseline includes BOTH production fast paths - FlashInfer split
routing and the multi-stream shared-expert fork; repeat runs within
+-2 us):

| B | baseline | opt | -routing | -fc2shard | -fc1shard | -overlap | -merged | -customcomm | -shared | -all |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 108.7 | **80.7** | 90.7 | 93.8 | 85.0 | 92.9 | 83.1 | 79.5 | 77.0 | 129.3 |
| 4 | 119.5 | **84.6** | 98.4 | 106.5 | 94.4 | 98.1 | 89.4 | 84.8 | 81.2 | 144.0 |
| 8 | 127.2 | **105.5** | 114.6 | 127.4 | 109.6 | 115.7 | 111.0 | 104.9 | 91.4 | 151.3 |
| 16 | 133.6 | **118.4** | 131.1 | 131.2 | 121.9 | 130.5 | 122.9 | 117.6 | 106.8 | 157.6 |
| 32 | 149.1 | **143.3** | 148.2 | 152.1 | 143.3 | 156.1 | 145.7 | 142.2 | 130.0 | 178.4 |
| 64 | 162.0 | **161.9** | 166.0 | 171.8 | 161.6 | 177.8 | 169.5 | 166.3 | 157.6 | 185.7 |

(Earlier revisions of this table benched against baselines missing
production fast paths - first the rc23 monolithic routing kernel
(`m5_..._ablate.log`), then a serial shared expert
(`a4_fi_base_ablate.log`) - each fix shaved the baseline and
tightened our margins; this table is the honest one.)

(`-shared` = shared expert removed entirely, the critical-path
check: at B>=8 it saves 10-15 us, so the shared chain is only
PARTIALLY hidden by our overlap at those sizes - residual
scheduling headroom. `-all` = every switch off ~= the baseline
re-implemented in our forward; it is now ~15-20% SLOWER than the
real baseline because the real baseline's aux-stream fork is a
production feature our all-off fallback doesn't replicate.)

### Routing top-k handoff -> **always on, now a modest lever**

* Experiment: `-routing` hands raw fp32 logits back to the runner's
  in-kernel routing - with the production op backend that is the
  SPLIT FlashInfer pipeline (`routingIndicesBlockScoresKernel`
  ~13 us + `routingIndicesClusterKernel` ~13 us at B=8).
  Ours: one Triton program per token does sigmoid + fp32 bias +
  top-16 (order-preserving packed int64 keys) directly on the bf16
  strided logits, emitting ids+scales in `run_moe`'s exact format,
  so the runner skips its scores stage and runs only the ~10 us
  post-top-k permute; the `logits.float()` cast dies.
* Result vs the ALIGNED baseline: our route_pack (~6 us) + permute
  replaces the ~26 us split pipeline -> ~10 us saved (previously
  ~35 us against the rc23 wheel's monolithic 41-46 us kernel - that
  inflated win is gone from all tables, as it should be).
* Caveat (trace-verified): the kernel is 16 dependent reductions and
  inflates 3-6x under concurrent GEMM waves - never overlap it; the
  layer gives it a clean window. Numerically CLOSER to the fp32
  oracle than `noaux_tc_op`-on-bf16 (fp32 accumulation, ~5e-3).
* History: earlier revisions benchmarked against the rc23 wheel's
  native op (monolithic `routingIndicesClusterKernel` /
  `routingIndicesBlockKernel`, 41-46 us) - production never ran that
  kernel. The `routingIndicesSmallBsKernel` (~6 us at B=1) seen in
  some production traces is a still-newer FlashInfer cubin tier that
  this container's flashinfer 0.6.15 does not ship; at B=1 our
  route_pack path is already at ~6 us-equivalent cost, so parity, and
  the ablation isolates the non-routing levers either way.

### FC2 column shard + fused RS -> **SUPERSEDED at B>=4 by ref_tail**

> Aug 5: the REF-overlap tail (see "Status" above) replaces this from
> B>=4; the shard tail below remains the B<=2 path. The `-fc2shard`
> ablation description still applies below the crossover.

* Experiment: `-fc2shard` runs the baseline-style tail (cat
  [latent|shared] partials -> ONE fat AR -> norm -> FULL fc2).
  Sharded: column reduce-scatter with the latent RMSNorm FUSED
  in-kernel, then each rank's 7168x448 fc2 slice accumulated into the
  shared partial via cuBLAS beta=1 (exact:
  `Y @ fc2.T = sum_r Y[:,s_r] @ fc2[:,s_r].T`), final [B,7168] AR.
  Compute cut: 51 MB -> 6.4 MB fc2 weight read; comm change: fat AR
  -> RS + hidden-size AR.
* Result (aligned sweep): sharded wins +10..22 us through B=64
  (biggest at B=4-8); forcing it at 128
  (`m4_mxfp8_fc2_always.log`) LOSES 30 us (the RS fixed cost +
  second rendezvous outgrow the weight cut once the fat AR is
  bandwidth-efficient).
* Shipped: gated at `FC2_SHARD_MAX_TOKENS = 64` (at 128 the opt path
  and `-fc2shard` coincide, hence the identical 185 in the table).

### FC1 column shard + AG with FUSED MXFP8 quantize -> **on at B<=16**

* Experiment: `-fc1shard` (off everywhere) vs forced ALWAYS-on, plus
  a B=8 trace pair (`results/moe_kimi_k3_tp8_b8_opt_fc1{off,on}_
  graph_annotated.trace.json`). Sharded: rank reads 448x7168 of fc1
  instead of 3584x7168, then a [B,448] one-shot AG rebuilds the
  latent - and the AG QUANTIZES ON ITS WRITE-OUT (`ag_mxfp8`,
  bit-exact replica of `cvt_warp_fp16_to_mxfp8`), emitting the (e4m3,
  ue8m0-scales) pair `run_moe` consumes. The experts' activation
  quantize is fused into the communication: the separate
  `quantize_with_block_size` kernel disappears from the critical
  path, and 448-column shards = 14 aligned 32-element scale blocks
  per rank, so no scale block straddles ranks.
* Result: +7..9 us at B<=8 and +3 at 16 (B=8: 111.5 -> 104.1).
  Forced past the gate: wash at 32, -3 at 64, -7 at 128. The B=8
  traces show the whole story: fc1off runs routing 10.6 -> latent
  copy 6.5 (the mxfp8 op demands contiguous input, the merged GEMM's
  slice is strided) -> quantize 4.3 -> permute; fc1on runs routing
  11.5 -> permute DIRECTLY, with the 8 us fused AG+quant hidden under
  the shared gate/up GEMM in the next-input window (walls 109.5 vs
  104.8). Standalone (tmp_ag_mxfp8_test, graph-timed): fused AG
  beats AG+quantize by 2-6 us at every size, bit-exact at all of
  them.
* Shipped: gated at `FC1_SHARD_MAX_TOKENS = 16` (was 8 before the
  fusion; the cheaper AG moved the crossover). At prefill scale the
  shard returns as a real 1/world FLOP cut (below).

### Shared-expert overlap -> **on through B=64, off above**

* Experiment: `-overlap` runs the shared chain inline; on = the chain
  (gate/up GEMM -> SiTU -> down GEMM) runs on an aux stream under the
  routed path. Wins 7-17 us at B<=64; at B=128 it LOSES (~203 off vs
  ~209 on): the shared GEMMs and the expert bmms are both
  weight-read-bound, and at 128 the DRAM pipe is saturated enough
  that overlapping them just serializes in the memory system.
  Crossover moved to `B10_OVERLAP_MAX_TOKENS=64`.
* Fork placement matters as much as on/off - measured with the
  `-shared` timing probe (skip the chain entirely; opt minus -shared
  = exposed shared time). The windows differ in what they consume:
  input GEMM/routing (SM+DRAM), expert bmms (DRAM-heavy), RS
  spin-wait (nearly nothing), fc2 (small). Placement by batch:
  - B < 8 (`B10_SHARED_FORK_EXPERTS_MIN_TOKENS`): fork after
    routing - the chain hides under the bmms; only ~3.5 us exposed.
  - B = 8..32: fork after the expert kernels, so the gate-up
    (13.2 us, ~100 MB weight read) runs under the RS spin-wait and
    the down-GEMM under fc2; the tail joins with an add instead of
    an addmm. B=32: 145.0 -> 139.3, B=16: 118.7 -> 115.7. Early
    forks instead stretch the fused input GEMM 8.8 -> 10.8 us and
    routing +1.7 us; the after-routing fork still leaves 7-13 us
    exposed to bmm DRAM contention.
  - B >= 64: routed split pipeline, fork after routing (expert
    window is long enough to hide the whole chain; exposure ~0 by
    B=128).
* Why the exposure cannot reach 0 by scheduling alone: the chain and
  the routed path share DRAM bandwidth; hiding memory-bound work
  under memory-bound work only relocates the bytes. The remaining
  shared cost is the weight read itself - cutting it needs fewer
  bytes (e.g. quantized shared weights), not a better schedule.

### Merged vs split input GEMM -> **merged below 64, split from 64**

* Experiment: `-merged` (separate gate/fc1 GEMMs everywhere) plus a
  forced split-never sweep (`m3_mxfp8_split_never.log`). Merged = ONE
  [gate|fc1] GEMM (one weight pass, one launch; its strided latent
  slice costs a contiguous-copy before the mxfp8 quantize unless the
  fc1 shard's AG rebuilds it). Split = gate -> routing -> fc1 serial:
  routing in a clean window, contiguous latent, no copy.
* Result: `-merged` loses 3-9 us at B<=64 and 13 at 128; split-never
  loses 4 us at 64 and 9 at 128. Crossover at
  `ROUTED_SPLIT_MIN_TOKENS=64` confirmed under the production mode.

### Custom comm kernels -> **AG stays; RS retiring (see Status)**

* Experiment: `-customcomm` falls back to flashinfer's fused AR+norm
  for the latent reduction.
* Result: a tie as a pure AR replacement (+-1.5 us at B<=32, +3.6 us
  for ours at 64) - which is exactly why the custom RS+norm retires
  once the ref_tail validation lands (with ref_tail on from B=4 it
  only serves B<=2 anyway). The custom kernel that KEEPS earning its
  place is the one-shot AG: TRT-LLM has NO small-message one-shot AG,
  fc1 sharding is built on it, and it carries the experts' MXFP8
  activation quantize in-kernel. See Part 3.

### Prefill path (`prefill_opt`, tokens >= 192)

At prefill the latent GEMMs are COMPUTE-bound: the fc1/fc2 shards cut
FLOPs by 1/world and the latent rebuild is a COPY-ENGINE all-gather
(zero SMs) hidden under gate/routing/shared compute. Measured
(cutlass comparison mode; production backend at 256 tokens: +24%):

| tokens | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| gain | +12% | +18% | **+22%** | **+22%** | **+21%** |

### Are we at the max?

B=64 opt critical path (annotated trace,
`results/moe_kimi_k3_tp8_b64_opt_graph_annotated.trace.json`): gate
9+5 -> routing 11 -> fc1 20 -> mxfp8 quantize 5 -> permute 17 ->
expert bmms 28+22 (PDL) -> finalize 9 -> RS 20 -> fc2 addmm 6 ->
AR 25; shared chain fully hidden under the fc1/permute/bmm window.
What remains is (a) the production expert kernels (~75 us of
quantize + permute + bmms + finalize, identical in both paths - the
shared denominator), (b) the RS reading 20 us against a ~13 us
lockstep floor - the rest is absorbing genuine EP routing skew the
baseline's AR also absorbs, (c) real wire in the final AR. Further
cuts need a persistent megakernel - diminishing and unmaintainable.

**Cross-rank verification of the RS wait** (all-rank graph traces,
`BENCH_ALL_RANK_TRACES=1`, B=8, 500 iters/rank,
`results/moe_kimi_k3_tp8_b8_opt_graph_r{0..7}.trace.json`): ranks
arrive at the RS with a median 10.2 us spread, yet the RS *ends*
within 0.9 us on all 8 ranks; the last-arriving rank's RS reads
11.4 us (the true kernel cost) while the first-arriving rank's reads
21.8 us (cost + wait). The skew is NOT TP asymmetry - the dense
TP stages are identical - it is the EP-sharded expert stage: rank 2's
expert shard consistently draws more (token, expert) work (bmm1/bmm2
14.4/15.1 us vs ~13/13 on other ranks, stable across all 500
deterministic replays), so everyone waits ~10 us for it. Overlap is
NOT the cause: with `overlap_shared` off (`*_noovl_graph_r*` traces)
expert-GEMM durations change by <0.5 us and the arrival skew is the
same (10.4 us median) - the aux-stream shared chain does not
meaningfully steal SMs from the routed GEMMs at decode sizes.

Second confirmation - vary the input (`BENCH_PERTURB_INPUT=1`
rewrites the graph input with a fresh, rank-identical tensor before
each replay block): the slowest rank ROTATES per block (r4, r3, r6,
r5, r7, r3, r5, r1, r4, r1 over ten blocks) and rank 0's RS median
swings 13.2-19.8 us block to block. If the skew were anything
systematic (clocks, NVLink topology, a slow GPU) the same rank would
always lag; instead it follows wherever the routing sends more
tokens - data-dependent EP load imbalance, as expected.

---

# Part 3 - supporting kernels

## Communication (`comm.py`, `comm_cuda.py`, `bench_comm.py`)

TRT-LLM reduces everything with ALLREDUCE; there is no small-message
one-shot AG/RS. We built both as single CUDA kernels over torch
symmetric memory with flashinfer's Lamport protocol (sentinel poll =
`ld.volatile.global.v4.u32` + `__vcmpeq2`, three rotating slots,
graph-replay-safe round counters, world-templated peer unrolling,
PDL). TP8, [B,3584] bf16, us:

| tokens | AG ours | RS ours | best AR | NCCL AG/RS |
|---|---|---|---|---|
| 1 | **3.8** | **3.9** | 4.5 | ~20 |
| 64 | 12.1 | **5.5** | 13.2 | ~23 |
| 256 | **14.6** | **9.9** | 22.2 | ~30 |
| 4096 | 258 | 122 | 156 | 135 |

Conclusion: at decode sizes both sit on the ~4 us NVLink latency
floor and beat every AR while moving 1/world of the wire - that is
the enabler for the fc1/fc2 shards. Past ~1k tokens the Lamport
sentinel clear prices them out; the layer falls back (NCCL / CeComm).
Roofline: 7 KB at 900 GB/s is ~8 ns of wire - these kernels are 100%
latency, and 4 us is the observed floor across every family. Our own
AR measured 6-8x behind flashinfer's and was deleted: only primitives
that BEAT the alternatives are kept.

Because the kernels are latency-bound, fusing epilogues into them is
free: `rs_cols` carries the latent RMSNorm, and `ag_mxfp8` carries
the experts' MXFP8 activation quantize on its write-out (transport
stays bf16, so the Lamport sentinel protocol is untouched; one thread
owns one 32-element scale block). The quantize recipe is a bit-exact
replica of TRT-LLM's `cvt_warp_fp16_to_mxfp8` - verified byte-for-
byte against `torch.ops.trtllm.mxfp8_quantize` for contiguous and
strided shards (`tmp_ag_mxfp8_test.py`); fused vs AG+quantize:
14.4->10.4 us at B=1, 16.6->12.7 at B=8, 26.7->20.5 at B=64.

`CeComm` = copy-engine collectives for prefill/overlap: peer
`cudaMemcpy2DAsync` data path (zero SMs) + one arrival-sync kernel.
Findings that took profiling: local same-device 2D memcpy queues
behind a saturating GEMM (use a small SM kernel instead); side-stream
comm needs a HIGH-PRIORITY stream to get slots mid-GEMM; poll loops
must nanosleep. At 4096 tokens: AG 179 us vs oneshot 349, overlap
cost +29 vs +81. It ships as the prefill fc1 rebuild.

## MoE routing (`routing_kimi_k3.py`, `bench_routing_kimi_k3.py`)

Covered in Part 2 (first switch). Emitters for every backend:
`route_for_trtllm_gen` (int32+bf16 for `run_moe`),
`route_for_fused_moe` (int32+fp32 for CUTLASS), `route_pack`
(packed `(id<<16)|bf16(w)`). One CTA per token; above
`TRITON_ROUTE_MAX_TOKENS=4096` the cooperative builtin wins and the
layer falls back.

## KDA layer (`kda_trtllm_kimi_k3.py`, `kda_b10_kimi_k3.py`)

`b10_fused` = ONE CuTeDSL kernel for conv + recurrence + gated norm
(same fusion as production `trt_fused`, better kernel): one CTA per
(batch, head) keeps the [128,128] fp32 state resident in
registers/SMEM for the whole step - one HBM round trip instead of the
Triton kernel's spills. Gap is biggest where decode lives (small B,
few local heads). tp8 B=1: 56.6 -> 47.8 us (+16%); tp1 B=32: 333.8 ->
276.1 (+17%). Bound analysis: ~115 MB of weights = ~14 us of HBM in a
47.8 us layer - latency-bound, GEMMs at their weight roofline,
nothing structural left short of a megakernel. All backends pass
output/conv-state/ssm-state/AttnRes checks vs `trt_fused`.

---

## Production-alignment notes (1.3.0rc19 @ 496bc01f8e)

The deployed build is `aaryams/hybrid_cache_fixes @ 496bc01f8e` on the
basetenlabs fork; the local `trt-llm` checkout is a descendant of
exactly that commit. What was verified / matched:

* **MoE structure** - the fork's `KimiK3MoE` passes
  `fuse_latent_and_shared_tp_reduction=True`,
  `shared_experts_gated=True` (SiTU), `latent_moe_use_norm` -> our
  baseline forward implements the identical flow (one AR over
  `cat[latent|shared]`, latent RMSNorm + fc2 AFTER the AR). No diff
  between 496bc01f8e and the checkout HEAD in `fused_moe/`,
  `modeling_nemotron_h.py`, `modeling_deepseekv3.py`, or the routing
  kernel sources.
* **Op backend** - production K3 FORCES the FlashInfer op backend
  (its SiTU epilogue lives in a private FlashInfer cubin pool via
  `FLASHINFER_PRIVATE_CUBIN_DIR`), so production expert + routing
  kernels are FlashInfer's trtllm-gen cubins - notably the SPLIT
  routing pipeline. The rc23 wheel gates flashinfer off for
  DeepSeekV3 routing; `_force_flashinfer_op_backend`
  (`moe_trtllm_kimi_k3.py`) swaps it in anyway (default;
  `BENCH_MOE_OP_BACKEND=trtllm` reverts). Verified bit-exact vs the
  native op at B=1/8; B=64 differs only through top-16 near-tie
  flips (bf16 scores in the split pipeline - true in production
  too). One shim: flashinfer 0.6.15 asserts a 2-D activation-scale
  layout that production's private build doesn't - reshaped at the
  call boundary, same bytes.
* **AllReduce** - production constructs `AllReduce(mapping,
  strategy)` WITHOUT the dtype arg (the rc23 parent passes dtype to
  pre-build MNNVL); baseline rebuilds it production-style. On this
  single-node NVL8 box both resolve to `NCCL_SYMMETRIC` (~21 us at
  B=8) since MNNVL doesn't apply.
* **Accepted deviations** - (1) routed-expert epilogue: production
  runs in-kernel SiTU from the private cubin pool; without those
  cubins we run the Swiglu epilogue on the SAME bmm kernels
  (elementwise epilogue, cost-identical). (2) the
  `routingIndicesSmallBsKernel` tier some production traces show at
  B=1 is a newer FlashInfer artifact than 0.6.15 ships.
* **Shared-expert multi-stream** - production's cuda_graph_runner
  captures under `with_multi_stream(True)`, which is the ONLY thing
  that makes `maybe_execute_in_parallel` fork the shared expert onto
  `aux_stream_shared` (in eager it always runs serial). Our
  `capture()` in `bench_moe_kimi_k3.py` now wraps warmup+capture the
  same way; the B=16 baseline trace confirms the shared gate/up
  splitK GEMM, `_situ_and_mul`, and down GEMM on a second stream,
  matching production profiles. This took the baseline down another
  ~10-16 us across sizes (sweep `a6_multistream_base_sweep.log`).
* **KDA** - the graft list (conv, recurrence, gated norm, fused
  decode, attn_res, stochastic rounding) is byte-identical between
  496bc01f8e and the checkout HEAD, so the KDA baseline was already
  aligned. The 4 files that DO differ (`kda_mixer.py`,
  `mamba2_metadata.py`, `fla/cached_replay.py`,
  `modeling_kimi_k3.py`) are prefill/cache-metadata paths newer than
  production - they are the sync-elimination fixes. Prefill D2H
  syncs still present in production @ 496bc01f8e: (1)
  `kda_mixer.py:762` boolean-index state reset
  (`state_indices[:num_prefills][~has_initial_states[:num_prefills]]`
  on a CUDA bool tensor -> internal `nonzero()` -> blocking D2H of
  the match count, EVERY prefill batch) - this is the D2H copy
  visible in production prefill profiles; (2) `fla/index.py:21`
  `prepare_chunk_indices` does `.tolist()` on a CUDA tensor (called
  from the Triton chunk-prefill kernels `chunk_kda`/`chunk_delta_h`/
  `chunk_o`; `@tensor_cache`d, so it fires once per distinct
  `cu_seqlens` - i.e. nearly every prefill step; skipped on the
  cute/flashkda prefill backends). The metadata chunk-index path
  itself is already sync-free at that commit (`total_seqlens` /
  `extra_chunks` are passed from CPU; the `.item()` calls in
  `mamba2_metadata.py:111/123` are unused fallbacks). The decode /
  verify path has no sync.
* The opt path deliberately pins the NATIVE trtllm op for its
  `run_moe`/quantize handoff (`_run_moe_native`): the flashinfer
  routed dispatch packs `(id<<16)|scale` with two extra elementwise
  kernels and is 1-8 us slower for precomputed routing. Baseline
  keeps the production backend; the opt path takes the fastest op.

## File map

```text
config.py               Kimi-K3 dims + K3Shard (tp1/tp8 local shapes)
comm.py / comm_cuda.py  OneShotComm AG / AG+MXFP8-quantize / RS /
                        col-RS+norm, CeComm, TunedAllGather,
                        FlashInferAllReduce (CUDA JIT)
routing_kimi_k3.py      our Triton routing + per-backend emitters
moe_trtllm_kimi_k3.py   baseline KimiK3MoE (real NemotronHMOE,
                        production trtllm-gen MXFP4 config)
moe_b10_kimi_k3.py      KimiK3MoEB10 - one switch per optimization
kda_trtllm_kimi_k3.py   baseline KDA (grafts unreleased TRT modules)
kda_b10_kimi_k3.py      KimiK3KDAB10 - CuTeDSL fused decode
bench_*.py              one bench per topic; --profile writes traces;
                        MoE bench writes csv + md table + 2-panel png
annotate_graph_trace.py inject stage labels into graph-replay traces
results/                csv/md/png + annotated traces (b64 opt,
                        b8 fc1off/fc1on pair)
results/experiments/    switch-sweep logs; a1..a4 = production-
                        aligned baseline (flashinfer op backend;
                        a3 = headline sweep, a4 = ablation),
                        m1..m6 = native-op-baseline runs (m6 = clean-
                        GPU ablation), e1..e5 = earlier w4a16 runs
run_final_experiments.sh  pending validation set (see Status section)
tmp_rs_norm_probe.py    RS/AG kernel probe + shared bench_graph helper
tmp_tail_ref_overlap.py fair tail-crossover probe (4 tail variants)
tmp_tail_shard_probe.py col- vs row-shard vs REF tail strategies
```

## Appendix: the b10 opt path at B=64 (~162 us), kernel map

Open `results/moe_kimi_k3_tp8_b64_opt_graph_annotated.trace.json` in
Perfetto; `stages-*` tracks label every kernel (graph replays cannot
carry record_function).

| stage | kernel | us | vs baseline |
|---|---|---|---|
| gate_gemm | `nvjet..._splitK` + reduce | 9+5 | bf16 out, no fp32 cast |
| routing | `_kimik3_route_pack_kernel` | 11 | replaces the aligned baseline's ~13 us scores kernel (and the old wheel's 46 us monolith); clean window |
| fc1_gemm | `nvjet...TNT` | 20 | own GEMM -> contiguous latent, no copy (B<64: merged [gate\|fc1] GEMM) |
| latent_quant | `quantize_with_block_size` | 5 | bf16 -> MXFP8, same kernel the baseline pays |
| experts.permute | `routingIndicesClusterKernel` | 17 | post-top-k permute only (the aligned baseline runs the same kernel after its scores stage) |
| experts.bmm x2 | `bmm_MxE4m3_MxE2m1MxE4m3` + `bmm_Bfloat16_MxE2m1MxE4m3` | 28+22 | identical production kernels (fp8 act x fp4 weight) |
| experts.finalize | `finalizeKernel` | 9 | identical |
| rs_latent+norm | `rs_cols_lamport_kernel<8>` | 20 | RS + fused RMSNorm (norm kernel dead); ~13 floor + EP skew |
| tail_fc2_addmm | `nvjet..._badd_TNN` | 6 | fc2 column slice, beta=1 into shared partial (add kernel dead) |
| allreduce_7168 | flashinfer `allreduce_fusion` | 25 | final AR (cat/stage copies dead) |

Aux stream, hidden under the fc1/permute/bmm window: shared gate/up
(26 concurrent with fc1), SiTU (3), down (10) - forked after routing,
joined before the tail. The B=8 pair
(`..._b8_opt_fc1{off,on}_graph_annotated.trace.json`) shows the fc1
shard + fused-quantize mechanism: fc1off runs routing -> latent copy
6.5 -> quantize 4.3 -> permute; fc1on runs routing -> permute
directly, with the 8 us `ag_mxfp8` (AG + in-kernel MXFP8 quantize)
hidden under the shared gate/up GEMM at the iteration tail.
