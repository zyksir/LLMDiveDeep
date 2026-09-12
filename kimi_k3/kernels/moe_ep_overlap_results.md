# MegaMoE EP dispatch/combine overlap study (fake weights)

Date: 2026-08-19
Qualification: mechanism findings QUALIFIED (controlled same-kernel A/B,
equal per-expert load verified at runtime); latency numbers UNQUALIFIED as
absolutes (fake weights, practice inter=512, autotuner not warmed) — valid
only as a same-harness comparison, same caveats as
moe_expert_region_results.md.

Claim under test: "if we do expert parallel, the communication can be fully
overlapped by the mega kernel in DeepGEMM."

Verdict: **TRUE only for B/rank <= 256, FALSE at B/rank >= 1024.** At EP=8
the kernel exposes a near-constant 3-5 us of communication at B<=128 (>=91%
of the equivalent-collective cost hidden, and 64-77% of even the
theoretical wire time hidden at B=32-256). From B=512 the exposed
communication grows to the FULL wire time: at B>=1024 the added
dispatch+combine bytes appear ~1:1 in total latency (zero effective
overlap); mega's remaining advantage over run_moe+NCCL there is a faster
wire (~790 vs ~615-660 GB/s) and no separate phase/kernel overhead — not
overlap. This also explains the arm crossover in
moe_expert_region_results.md: mega's 1.6-3.4x wins at B<=512 are overlap
wins, its 1.2-1.5x wins at 1024-2048 are leaner-pipeline wins, and its
losses at >=8192 follow from 1.5x more wire bytes with none of them hidden.

## Operation contract

Same expert region as moe_expert_region_results.md, EP topology:
`Mapping(tp=w, moe_ep_size=w, moe_tp_size=1, enable_attention_dp=True)`,
w=8 (plus w=2/4 sweeps), 896 global experts (112/rank at w=8; the w=2/4
sweeps use 224/448 experts so experts/rank stays 112 and per-rank compute
is world-invariant), hidden 3584, practice inter 512, top-16,
W4A8 MXFP4/MXFP8, fake weights. Per rank: bf16 x `[B, 3584]` + GLOBAL
top-16 ids/scales -> combined routed bf16 `[B, 3584]`.
`dg_mega_moe = MegaMoEDeepGemm.run_moe` -> DeepGEMM `fp8_fp4_mega_moe`,
ONE kernel. All numbers CUDA-graph medians, MAX over ranks, barriers
between ranks; zero capture failures. B = tokens PER RANK.

## What the kernel actually does (source anatomy)

From `tensorrt_llm/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
(TRT-LLM 1.3.0rc23 bundle). The kernel is warp-specialized per SM:
dispatch warps + 1 TMA-load warp + 1 MMA-issue warp + epilogue warps.

Designed overlap points:
- **Dispatch pull overlaps GEMM1 at block granularity.** After a metadata
  exchange (per-expert counts + source indices pushed to owner ranks, one
  grid sync, one NVLink barrier), dispatch warps PULL each token-expert
  pair (fp8 row 3584 B + SF 112 B + weight 4 B) from the source rank's
  symm buffer via TMA and bump a per-BLOCK_M arrival counter; the GEMM1
  loader spins on that counter per block, so compute chases the pull
  block-by-block.
- **Combine push is fused into the GEMM2 epilogue.** Each L2 output row is
  written directly to the source rank's combine slot over NVLink
  (`sym_buffer.map` store) from the epilogue warps, inside the 2-stage
  TMEM accumulator pipeline.

Serial points (cannot overlap by construction):
- metadata exchange + `fetch_expert_recv_count` spin (GEMM scheduling waits
  for ALL ranks' counts) + 3 NVLink barriers (before pull, before combine
  reduce, after workspace clean);
- the final combine reduce: each rank reads B*16 bf16 partial rows
  (7168 B each) from its LOCAL buffer only after every rank's pushes passed
  the barrier, then reduces and TMA-stores `y`.

The pull does NOT dedup a token whose top-16 hits several experts on one
rank, so wire volume is per PAIR:

- dispatch: 3704 B/pair, combine: 7168 B/pair; with r of 16 pairs remote,
  per-rank wire per direction = `B * r * 10872` bytes (in: pulls + received
  pushes; out: served pulls + sent pushes). Uniform routing over 896
  experts gives r ~= 14 at EP=8.
- For comparison the non-fused arm (NCCL AG of bf16 x + ids/scales, local
  run_moe over w*B tokens, NCCL RS) moves `B * 7 * ~7264 + B * 7 * 7168`
  ~= 101 KB/token — mega moves **1.5x more wire bytes** (152 KB/token).

## Method: controlled comm-volume A/B at equal expert load

The mega path is one kernel, so phases cannot be read from a timeline.
Instead the wire fraction is varied at exactly equal per-expert load: every
variant uses the same per-token `base[t,s] in [0,112)` (same seed on all
ranks, randperm so slots are distinct) and only remaps which rank owns the
expert:

- **local**: `ids = own_rank*112 + base` — pull/push resolve to the rank's
  own symm buffer (loopback over local HBM; identical instruction path,
  wire = 0). This is the compute-side anchor t_compute(B).
- **shift4**: `ids = ((r+4)%8)*112 + base` — 16/16 pairs remote, one peer.
- **balanced**: `ids = ((r+s)%8)*112 + base` — pairs spread over all 8
  ranks, 14/16 remote = the uniform-routing wire fraction.
- **uniform**: `ids ~ U[0,896)` — the original harness distribution.

The multiset of per-expert token counts is identical across
local/shift4/balanced by construction and was verified every row via a
global bincount (`load_eq = PASS` on all 15 batch sizes, all worlds): equal
GEMM work, equal active-expert weight bytes, equal block schedule. The
measured delta `t(variant) - t(local)` IS the exposed communication.

Wire anchors (measured in the same process, CUDA-graph timed): NCCL
all_gather sized so per-rank incoming bytes equal the dispatch pull volume,
NCCL reduce_scatter sized to the combine push volume. "wire @ peak" is the
arithmetic floor at 900 GB/s per direction (B200 NVLink, strict bound only
— mega's achieved wire rate is lower, see anomalies).

## Results — EP=8 variant latencies (us)

| B/rank | dg_mega local | dg_mega shift4 | dg_mega balanced | dg_mega uniform | nccl ag anchor | nccl rs anchor |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **53.1** | 56.5 | 56.6 | 59.8 | 21.7 | 20.4 |
| 2 | **59.7** | 62.8 | 62.7 | 64.6 | 22.6 | 20.6 |
| 4 | **70.3** | 73.7 | 73.6 | 78.6 | 22.5 | 21.2 |
| 8 | **87.3** | 91.3 | 91.3 | 88.2 | 22.7 | 21.4 |
| 16 | **93.9** | 97.3 | 97.9 | 98.1 | 22.7 | 23.2 |
| 32 | **99.3** | 102.6 | 102.7 | 103.0 | 24.3 | 29.2 |
| 64 | **110.8** | 114.5 | 114.7 | 114.4 | 30.7 | 40.9 |
| 128 | **117.8** | 122.2 | 122.8 | 124.0 | 44.9 | 53.0 |
| 256 | **143.4** | 160.2 | 154.9 | 156.5 | 49.1 | 76.4 |
| 512 | **176.2** | 248.7 | 236.4 | 237.9 | 73.8 | 134.0 |
| 1024 | **240.8** | 436.4 | 413.4 | 415.5 | 135.4 | 189.2 |
| 2048 | **415.1** | 814.5 | 761.4 | 761.8 | 197.9 | 359.2 |
| 4096 | **682.1** | 1565.6 | 1448.5 | 1455.6 | 366.9 | 699.4 |
| 8192 | **1268.7** | 3092.4 | 2818.1 | 2824.2 | 715.3 | 1327.0 |
| 16384 | **2512.5** | 6127.0 | 5550.5 | 5608.9 | 1378.1 | 2500.0 |

(local is bold as the zero-wire control — it must win every row; a row
where it did not would falsify the construction.) balanced ~= uniform
everywhere (within 1.5%), so the synthetic balanced routing is a faithful
stand-in for uniform routing; shift4/balanced exposure ratios sit at
1.13-1.19 vs the 16/14 = 1.143 byte ratio (B>=1024), i.e. the delta scales
with wire BYTES, not with peer count — the attribution evidence that the
delta is communication.

## Results — overlap verdict at EP=8

exposed = balanced - local; "hidden vs anchor" = 1 - exposed/(ag+rs)
(fraction of what the same wire costs as separate NCCL collectives);
"hidden vs peak-wire" = 1 - exposed/(bytes @ 900 GB/s) (intrinsic overlap,
conservative).

| B/rank | exposed comm (us) | wire bytes/rank (MB) | wire @ 900 GB/s (us) | nccl anchor (us) | hidden vs anchor | hidden vs peak-wire |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.4 | 0.15 | 0.2 | 42.1 | **92%** | floor |
| 2 | 3.0 | 0.30 | 0.3 | 43.2 | **93%** | floor |
| 4 | 3.3 | 0.61 | 0.7 | 43.7 | **92%** | floor |
| 8 | 4.0 | 1.22 | 1.4 | 44.1 | **91%** | floor |
| 16 | 4.0 | 2.44 | 2.7 | 45.9 | **91%** | floor |
| 32 | 3.3 | 4.87 | 5.4 | 53.5 | **94%** | 38% |
| 64 | 3.9 | 9.74 | 10.8 | 71.5 | **94%** | 64% |
| 128 | 5.1 | 19.5 | 21.6 | 97.9 | **95%** | 77% |
| 256 | 11.5 | 39.0 | 43.3 | 125.6 | **91%** | 74% |
| 512 | 60.2 | 77.9 | 86.6 | 207.8 | 71% | 30% |
| 1024 | 172.6 | 155.9 | 173.2 | 324.6 | 47% | 0% |
| 2048 | 346.3 | 311.7 | 346.4 | 557.1 | 38% | 0% |
| 4096 | 766.4 | 623.4 | 692.7 | 1066.3 | 28% | -11% |
| 8192 | 1549.4 | 1246.9 | 1385.4 | 2042.3 | 24% | -12% |
| 16384 | 3038.0 | 2493.8 | 2770.9 | 3878.0 | 22% | -10% |

Per-regime mechanism:
- **B<=256 — communication effectively fully overlapped.** Exposure is a
  near-constant 3-5 us (11.5 at 256): the NVLink pull/push round-trip
  latency floor, world-invariant (2.4 us at EP=2, 2.8 at EP=4, 3.4 at
  EP=8 for B=1). "floor" rows: wire time itself is < 3 us, so the whole
  (tiny) wire is this latency, not bandwidth. The 91-95% "hidden vs
  anchor" is the honest statement of the claim: what a separate AG+RS
  would put on the critical path (42-126 us) shrinks to 3-12 us.
- **B=512-1024 — the overlap capacity is exhausted.** Wire grows ~152 KB
  per token while the hideable compute window grows only ~0.15 us/token;
  at 512 still 30% of intrinsic wire is hidden, at 1024 0%.
- **B>=1024 — serialized.** Exposed equals or exceeds the peak-rate wire
  time; marginal slopes (8192->16384): compute 0.152 us/token, exposed
  wire 0.182 us/token, total 0.334 = their SUM — the kernel ADDS wire and
  compute instead of maxing them. Wire is also intrinsically dominant
  (0.182 > 0.152 us/token), so even perfect overlap could not hide it —
  but the kernel does not reach even that max() bound.

Why the serialization at large B (source-grounded): (1) the combine push
is issued by the epilogue warps inside a 2-deep TMEM accumulator pipeline
— once a block's push time exceeds its MMA time, MMA stalls on
`tmem_empty`, inserting wire time INTO the GEMM pipeline; (2) the dispatch
pull bounces every 3584 B chunk through SMEM with a full
`tma_store_wait<0>` drain (one outstanding chunk per warp, latency-bound
per warp), and GEMM1 waits for FULL BLOCK_M arrival per block, so when the
aggregate pull rate is at its wire ceiling the GEMM idles behind arrivals;
(3) the count-finalization spin, 3 NVLink barriers, and the post-barrier
combine reduce (B*16*7168 local bytes) are strict phase serializations.
The exact intra-kernel split between (1) and (2) is OPEN — it is one
kernel and needs ncu/SASS-level profiling of a multi-rank capture; the
evidence gathered is the additive slope plus the declining marginal wire
rate below.

## Results — where mega's win over the non-fused arm comes from

Non-fused arm = unchanged TRTLLMGen `run_moe` + NCCL AG/RS DEP path, same
harness, fresh run this date ("trt wire" = its measured AG+RS probe;
"trt rest" = total minus probe = its gathered-token quantize/permute/
GEMMs/finalize).

| B/rank | trt_run_moe+nccl | trt wire | trt rest | dg_mega uniform | mega compute (local) | mega exposed wire | winner |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 99.5 | 43.7 | 55.8 | **59.8** | 53.1 | 3.4 | dg_mega 1.66x |
| 2 | 108.1 | 43.8 | 64.3 | **64.6** | 59.7 | 3.0 | dg_mega 1.67x |
| 4 | 123.8 | 43.5 | 80.3 | **78.6** | 70.3 | 3.3 | dg_mega 1.57x |
| 8 | 145.4 | 45.1 | 100.3 | **88.2** | 87.3 | 4.0 | dg_mega 1.65x |
| 16 | 174.3 | 45.5 | 128.8 | **98.1** | 93.9 | 4.0 | dg_mega 1.78x |
| 32 | 180.5 | 50.0 | 130.5 | **103.0** | 99.3 | 3.3 | dg_mega 1.75x |
| 64 | 258.6 | 63.8 | 194.8 | **114.4** | 110.8 | 3.9 | dg_mega 2.26x |
| 128 | 418.4 | 89.7 | 328.7 | **124.0** | 117.8 | 5.1 | dg_mega 3.37x |
| 256 | 439.2 | 104.0 | 335.2 | **156.5** | 143.4 | 11.5 | dg_mega 2.81x |
| 512 | 487.8 | 142.1 | 345.7 | **237.9** | 176.2 | 60.2 | dg_mega 2.05x |
| 1024 | 609.9 | 269.1 | 340.8 | **415.5** | 240.8 | 172.6 | dg_mega 1.47x |
| 2048 | 917.7 | 387.8 | 529.9 | **761.8** | 415.1 | 346.3 | dg_mega 1.20x |
| 4096 | 1482.6 | 716.6 | 766.0 | **1455.6** | 682.1 | 766.4 | tie (1.02x) |
| 8192 | **2704.2** | 1397.2 | 1307.0 | 2824.2 | 1268.7 | 1549.4 | trt 1.04x |
| 16384 | **5201.1** | 2670.0 | 2531.1 | 5608.9 | 2512.5 | 3038.0 | trt 1.08x |

Per-regime attribution of the win:
- **B<=512: the win IS overlap plus per-rank-sized routing.** Mega hides
  43-142 us of wire down to 3-60 us, AND its routing/permute metadata
  stays sized to B while trt's post-AG pipeline processes w*B = 8B tokens
  (trt rest jumps 195->329 us at B=64->128 when its gathered 512->1024
  token pipeline crosses a tile boundary; mega compute rises only
  111->118 us there).
- **B=1024-2048: overlap is gone; the win is a leaner pipeline and a
  faster wire.** Mega's exposed wire (173-346 us) is comparable to trt's
  (269-388 us) despite mega moving 1.5x MORE bytes — mega's in-kernel
  pull/push sustains <=789 GB/s (bound from the shift4 exposure at 16384:
  2.85 GB / 3614 us) vs the NCCL anchors' 616 GB/s (AG) / 658 GB/s (RS) —
  and mega's compute side (415 us at 2048) beats trt's gathered-token
  rest (530 us).
- **B>=8192: trt wins** because mega's 1.5x wire bytes are now fully
  exposed at a wire-dominant operating point and trt's grouped GEMMs are
  better at large per-expert M (same large-B mechanism as
  moe_expert_region_results.md).

## Results — world-size sweep (exposed comm, us; experts/rank fixed at 112)

Per-rank compute is world-invariant by construction and measured so: the
local column agrees across w=2/4/8 within 3.5% at every B (e.g. 2427.9 /
2460.9 / 2512.5 us at B=16384 — the small growth IS the extra multi-rank
barrier/metadata cost). Remote pairs per token: 8 (w2), 12 (w4), 14 (w8).

| B/rank | EP=2 exposed | EP=4 exposed | EP=8 exposed | EP=2 hidden vs anchor | EP=4 | EP=8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **2.4** | 2.8 | 3.4 | 90% | 91% | 92% |
| 2 | **2.8** | 3.5 | 3.0 | 89% | 89% | 93% |
| 4 | **2.5** | 2.5 | 3.3 | 91% | 92% | 92% |
| 8 | **2.4** | 3.0 | 4.0 | 92% | 91% | 91% |
| 16 | **2.7** | 3.1 | 4.0 | 92% | 92% | 91% |
| 32 | **2.4** | 3.2 | 3.3 | 94% | 93% | 94% |
| 64 | **2.4** | 3.3 | 3.9 | 96% | 96% | 94% |
| 128 | **3.0** | 4.5 | 5.1 | 96% | 96% | 95% |
| 256 | **5.6** | 9.1 | 11.5 | 94% | 93% | 91% |
| 512 | **13.3** | 41.6 | 60.2 | 91% | 77% | 71% |
| 1024 | **52.9** | 133.8 | 172.6 | 80% | 54% | 47% |
| 2048 | **111.2** | 264.7 | 346.3 | 76% | 51% | 38% |
| 4096 | **292.2** | 595.9 | 766.4 | 67% | 40% | 28% |
| 8192 | **593.2** | 1226.5 | 1549.4 | 64% | 34% | 24% |
| 16384 | **1208.2** | 2491.8 | 3038.0 | 62% | 31% | 22% |

Why: smaller worlds keep more communication hidden at every B because the
compute window is the same but the wire is (w-1)/w smaller — at EP=2,
B=512 the wire still fits 73% under compute where EP=8 fits 30%. The
exposure ratio w8/w2 at large B is 2.5, ABOVE the 14/8 = 1.75 byte ratio:
partly the fixed hideable slack (subtracting a constant from both), partly
a lower multi-peer wire rate — the shift4-vs-balanced marginal rate falls
from 838 GB/s (B=2048) to 618 GB/s (B=16384). The exact split is OPEN
(same single-kernel profiling limit as above).

## Anomalies (root-caused or open)

- **uniform vs balanced jitter at tiny B** (B=4: 78.6 vs 73.6; B=8: 88.2
  vs 91.3, uniform faster): uniform routing draws a binomial per-rank pair
  count and timing takes MAX over 8 ranks; balanced is deterministic. The
  +-5 us matches max-of-8 load fluctuation at ~2 pairs/expert. Root-caused.
- **"hidden vs peak-wire" slightly negative at B>=4096** (-10..-12%): the
  900 GB/s column is a strict floor; mega's achieved multi-peer wire rate
  is <=789 GB/s (shift4 bound), so true wire time exceeds the column and
  true intrinsic overlap there is ~0, not negative. Root-caused
  (bound arithmetic, not extra slowdown).
- **shift4/balanced ratio 1.47 at B=256** vs 1.143 expected: both
  exposures are ~12-17 us there, within a few us of the latency floor —
  denominator too small for the ratio to be meaningful. Root-caused.
- **EP=1 anchor transients** (supplemental only): two fresh EP=1 runs of
  bench_expert_region.py each showed ONE spiked row at a different B
  (97.6 us at B=1; 544.8 us at B=1024) with all other rows within ~3% of
  each other and of moe_expert_region_results.md. GPU0 had no other
  processes; a clock/throttle transient spanning >3 of the 7 graph replays
  is the suspect. OPEN — does not affect the EP verdict (the compute
  anchor used is the in-process EP=8 local variant, and stable EP=1 rows
  confirm it: EP=1 mega 2562-2580 us at B=16384 vs EP=8 local 2512.5 us).
- **World-sweep superlinearity**: see sweep section; split between fixed
  slack and multi-peer rate degradation OPEN with the evidence listed.

## Alternatives and decision

- Chosen method: same-kernel routing-locality A/B (local/shift4/balanced)
  with runtime-verified equal per-expert load — it isolates wire time
  inside a single fused kernel without touching kernel code or timing
  boundaries.
- Rejected: reading phases from a profile (one kernel — no boundaries);
  EP=2/4 at 896 experts (changes experts/rank, confounds compute — solved
  by scaling total experts to keep 112/rank).
- Consequence for the EP serving decision in moe_expert_region_results.md:
  unchanged ADOPT-track for mega at B/rank<=2048, but the reason to prefer
  it at 1024-2048 is pipeline leanness, not overlap — so a non-fused arm
  with a better collective (communication/RESULTS.md symm one-shot AG /
  copy-engine movers at 550->790+ GB/s effective) could close most of the
  1.2-1.5x gap there, while the <=512 overlap win (1.6-3.4x) cannot be
  matched by any separate-collective pipeline that keeps wire on the
  critical path — the only counter would be inter-layer overlap
  (micro-batch pipelining) outside the region.
- Improvement lever inside mega, if upstreamed: the wire is per-PAIR
  (152 KB/token vs the AG/RS path's 101 KB/token); deduplicating pulls per
  (token, rank) — a token reaching k experts on one rank needs its 3.7 KB
  once, not k times — would cut dispatch wire up to ~2x at top-16 over
  112-expert ranks and directly shrink the exposed time at B>=512.

## Reproduce (trt-dev container, 8x B200)

```bash
D=kimi_k3_layer/kernel_research/kimi_k3_ep_overlap
# EP=8 variants + anchors (~4 min):
mpirun --allow-run-as-root -np 8 python3 $D/bench_ep_overlap.py
# derived tables (instant):
python3 $D/analyze.py $D/ep_overlap.csv 8 14
# world sweep, experts/rank held at 112 (~3 min each):
mpirun --allow-run-as-root -np 2 python3 $D/bench_ep_overlap.py --experts 224 --variants local,balanced --csv $D/ep_overlap_w2.csv
mpirun --allow-run-as-root -np 4 python3 $D/bench_ep_overlap.py --experts 448 --variants local,balanced --csv $D/ep_overlap_w4.csv
# non-fused arm reference (unchanged harness, ~3 min):
mpirun --allow-run-as-root -np 8 python3 kimi_k3_layer/kernel_research/kimi_k3_megamoe_practice/bench_expert_region_ep.py
```
