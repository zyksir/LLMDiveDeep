# Kimi-K3 MoE optimization — decode + prefill strategy (TP-8) and evidence

Date: 2026-08-19 (all tables in this file re-measured today in single
runs unless marked). TP8 on 8x B200,
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23`-class containers, vs
the production-aligned reference (`KimiK3MoEReference`, same weights
and semantics). The layer of record is
`kimi_k3_layer/b10_kimi_k3_moe_layer.py`; the shipped configuration is
`measured_config(tokens)` — a frozen, per-token-count
`ExperimentConfig`; DEPLOY mode has no mutable flags.

Two structural rules of the code:

1. **The layer owns no communication.** Every collective goes through
   ONE `communication.collective.Collectives` instance (`all_reduce`,
   `allreduce_norm`, `all_gather(impl=...)`, `all_reduce_dedicated`),
   autotuned per `(op, dim, size-bucket)`, world==1 handled inside —
   no per-call-site backend picking, no `if world > 1` at call sites.
2. **Strategy = one frozen dataclass.** `ExperimentConfig` axes:
   `decode_front`, `routing`, `decode_tail`, `route_on_side_stream`,
   `prefill_fc1`, `prefill_tail`, `prefill_expert_backend`,
   `prefill_overlap_shared_branch`. Every axis value that exists is
   used by the measured plan or is a live EXP ablation; dead variants
   are deleted (Appendix A).

There is NO baseline fall-back window: decode serves tokens <= 256,
prefill serves everything above. The dispatch has exactly three
regimes with two boundaries (256, 2048) plus the expert-backend
switch at 8192.

## 1. Final results — TP-8 (Aug-19 shipping validation, one run)

CUDA-graph replay, 100 iters, 8 deterministic inputs, tie-aware
correctness gated at every size (`bench_b10_kimi_k3_moe_layer.py
--sizes all`; CSV `local_results/bench_moe_layer_tp8_shipping_aug19.csv`).

| B | opt (us) | vs reference |
|---:|---:|---:|
| 1 | 44.8 | **+40.2%** |
| 2 | 45.9 | **+42.9%** |
| 4 | 48.3 | **+39.8%** |
| 8 | 50.5 | **+37.8%** |
| 16 | 59.4 | **+34.5%** |
| 32 | 68.3 | **+34.9%** |
| 64 | 81.0 | **+30.0%** |
| 128 | 99.5 | **+23.3%** |
| 256 | 132.8 | **+21.5%** |
| 512 | 178.8 | **+16.4%** |
| 1024 | 263.4 | **+22.5%** |
| 2048 | 366.7 | **+29.6%** |
| 4096 | 624.2 | **+34.4%** |
| 8192 | 1093.3 | **+38.9%** |
| 16384 | 1980.6 | **+45.7%** |

Reproduce (repo root, ~5 min):

```bash
mpirun --allow-run-as-root -np 8 python3 \
  kimi_k3_layer/bench_b10_kimi_k3_moe_layer.py \
  --sizes all --iters 100 --n-inputs 8
```

### 1.1 Strategy contribution waterfall (Aug-19, full range)

Cumulative ladder (`ablate_moe_contributions.py`): start from the
exact reference forward, add ONE strategy at a time until the config
equals `measured_config(tokens)`. **Each step cell is the LATENCY
SAVED by adding that strategy, in us** (+2.1 = 2.1 us faster; a
negative cell = that step made the layer slower), GIVEN the strategies
before it (contributions interact; the ladder is the natural build
order). `baseline` and `final` are absolute; per row,
baseline − (sum of cells) = final. **Bold** = largest saving in the
row. Raw CSVs `local_results/ablate_moe_contributions_tp8*.csv`.

Decode (TP-8 on GB300/sm103a, authenticated rc19 production image;
unchanged TRT KimiK3MoE baseline, BF16 I/O, MXFP4/MXFP8 experts, CUDA
graph timing, 100 iterations × 8 inputs). B=1..128 follows the current
decode plan; B=256 is the same decode ladder measured with the cap
temporarily extended from the current B300 boundary of 128:

| B | baseline | skeleton (pre-route + packed AR) | +fc1 shard (col-AG+quant) | +fused front (dual-out) | +radix routing | +route side stream | +measured tail | final | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71.2 | −16.5 | +4.5 | +3.5 | **+12.3** | +1.4 | +8.5 | 57.4 | **+19.4%** |
| 2 | 80.1 | −16.8 | +4.2 | +1.3 | **+14.5** | +1.4 | +9.2 | 66.2 | **+17.3%** |
| 4 | 93.3 | −15.2 | +2.8 | −0.1 | **+16.3** | +2.6 | +8.5 | 78.3 | **+16.1%** |
| 8 | 118.4 | −10.7 | +3.4 | +1.7 | **+14.0** | +4.0 | +8.4 | 97.7 | **+17.5%** |
| 16 | 151.4 | −12.2 | +3.8 | +2.9 | **+14.1** | +3.6 | +9.3 | 130.0 | **+14.2%** |
| 32 | 198.4 | −17.4 | +5.0 | +4.3 | **+14.4** | +3.7 | +13.2 | 175.0 | **+11.8%** |
| 64 | 254.9 | −10.6 | +3.6 | +1.4 | +14.0 | +5.1 | **+15.4** | 226.0 | **+11.3%** |
| 128 | 355.7 | +4.7 | +1.8 | +2.4 | **+15.5** | +4.6 | +12.9 | 313.9 | **+11.8%** |
| 256 | 424.1 | +5.7 | +5.0 | −4.5 | **+20.4** | +2.5 | +7.7 | 387.3 | **+8.7%** |

Prefill (fresh ladder matching today's shipped plan; 512–1024 rows end
at their plan config FULL+FC2_SHARD, larger rows continue through the
sharded steps):

| B | baseline | skeleton | +overlap shared | +radix routing | +fc2-shard tail | +fc1 shard (DMA AG) | +native experts | final | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 213.4 | −4.7 | +6.0 | +12.3 | **+21.9** | (plan: full) | | 178.0 | **+16.6%** |
| 1024 | 337.9 | −10.8 | +9.6 | +10.3 | **+66.2** | (plan: full) | | 262.5 | **+22.3%** |
| 2048 | 512.4 | −10.3 | +6.9 | +17.2 | **+79.9** | +13.4 | | 405.3 | **+20.9%** |
| 4096 | 934.7 | +7.6 | −1.0 | +31.0 | **+201.9** | +55.8 | (plan: flashinfer) | 639.5 | **+31.6%** |
| 8192 | 1758.3 | +6.4 | −20.1 | +68.7 | **+436.9** | +147.1 | −17.6 | 1136.9 | **+35.3%** |
| 16384 | 3655.2 | +26.6 | −18.7 | +184.2 | **+1015.8** | +352.5 | +13.5 | 2081.2 | **+43.1%** |

(The fc1-shard rung was measured with the then-current DMA gather +
interleave copy; the shipped config later gained the multimem gather
and the fused `gather_quant` kernel — worth a further
+38/+15/+44/+101 us at 2048/4096/8192/16384, visible as §1's finals
vs this table's.)

What the waterfall says, per strategy:

- **Decode: radix routing is the largest rung at most sizes**
  (+12.3..+20.4 us). The measured tail remains substantial
  (+7.7..+15.4 us) and is the largest rung at B=64: fc2-shard at B<=8
  (1/world fc2 weight read), multimem full-fc2 at B>=16 (no output
  collective; the shared AR reduced switch-side on the overlap stream).
- **Prefill: the fc2-shard tail is the dominant win everywhere**
  (+21.9 at 512 growing to +1015.8 at 16K — it deletes 7/8 of the fc2
  FLOPs for the same wire), with the overlapped-gather fc1 shard
  second from 2048 up (+13.4..+352.5 at this rung, more after the
  multimem + gather_quant upgrades).
- **radix routing**: +12.3..+20.4 us at decode, growing to
  +184 us at 16K (vs REFERENCE noaux_tc routing).
- **fused dual-out front** −4.5..+4.3 us; **route side stream**
  +1.4..+5.1 us at decode. The skeleton is costly at B=1..64
  (−10.6..−17.4 us) but saves +4.7/+5.7 us at B=128/256.
- Rung-conditional negatives (verified 3x same-process,
  `local_debug/moe_prefill_axis_probe.py`): overlap_shared is
  −19..−20 us at 8K/16K in the SKELETON rung (full fc1 + reference
  routing saturates the SMs, nothing to hide under) yet wins +75 us
  in the shipped sharded config — a ladder step's sign depends on the
  rungs below it. native experts is marginal at exactly 8192
  (−17.6 here, ~0..+34 across runs) and clearly positive at 16K.

Correctness per row: tie-excluded error <= 1.0e-2 everywhere (raw err
at prefill sizes is dominated by benign near-tie top-16 flips,
247..8012 tie rows).

Reproduce (~2.5 min per size group):

```bash
mpirun --allow-run-as-root -np 8 python3 \
  kimi_k3_layer/ablate_moe_contributions.py --sizes 1,8,32,128
```

## 2. Decode strategy (TP-8, tokens <= 256)

`measured_config` for decode:

| axis | pick | boundary + why |
|---|---|---|
| `decode_front` | `FUSED_FC1_SHARED_GATE_CUTE` — ONE dual-output CuTeDSL GEMM `[fc1-shard \| shared g/u \| gate]` emitting (merged, logits) | **every decode size, unconditional** (Aug-19 shipped-config A/B: fused 81.2/100.2 us vs separate-sharded 89.8/111.5 at B=64/128; also wins at 256, 132.9 vs 138.8). Wide-N reads run ~4x the bandwidth of narrow-N |
| `routing` | **`RADIX`** — unchanged SGLang RouteRadixKernel (§2.1) | every size; fastest 1..16384 with bit-identical IDs |
| `route_on_side_stream` | **True** | +1.6..+5.7 us: routing, the column-AG+quantize, and the shared chain all depend only on the front, so routing leaves the main-stream critical path |
| `decode_tail` | `SHARDED_FC2_OUTPUT_REDUCE` (per-rank fc2 column shard `addmm_` into the shared partial, one output AR) | **B <= 8**: the 1/world fc2 weight read dominates at tiny B |
| | `FULL_FC2_MULTIMEM_SHARED_REDUCE` (full fc2 on every rank — identical output, NO output collective; shared partial reduced by NVLS multimem AR on the overlap stream) | **B >= 16**: multimem's switch-side reduction makes the overlapped shared AR cheap; the output collective disappears |

`DECODE_MAX_TOKENS = 256` (Aug-19, was 128): the decode path still
beats the baseline at 192/256 (+18.1/+21.7% in the shipping run), so
the cap was extended to shrink the former fall-back window; the fused
front boundary merged into the cap (its old constant equaled 128 = the
old cap, a dead conditional).

Cross-cutting mechanisms (structural, no flags):

- **Zero-copy AR staging**: producers write AR operands directly into
  `Collectives.symm_input` views when the autotuned pick is
  symm-staged (the expert finalize's `moe_output`, the shared down
  GEMM's `out=`); disjoint offsets partition one buffer between
  concurrent ARs (`_split_ar_staging`).
- **Column-slice RMSNorm** (`kernels/rmsnorm_cutedsl.py`): NOT a
  duplicate of TRT-LLM's RMSNorm — that kernel needs contiguous input
  and writes full rows; this one reads strided input in place and
  writes only the consumed column window. Measured vs TRT-LLM's at
  the layer's shapes (1 GPU, `local_debug/rmsnorm_vs_trtllm_probe.py`):
  strided full-width 1.6 vs 4.1 us at B=32, 36.3 vs 162.7 at 16384;
  window 1.5 vs 4.5 and 21.0 vs 52.3 — 2.5-4.5x, bit-exact through
  B=2048 (bf16 reduction-order ulps at 16K).
- **fc1 column shard + fused AG-quantize (decode)**: each rank
  computes `[B, 448]` of the latent; the bounded Lamport col-AG
  engine rebuilds it with MXFP8 quantize fused on the write-out.

### 2.1 Routing

The math (DeepSeekV3-style noaux_tc, n_group=1): sigmoid scores
$s_e = \sigma(\ell_e)$ over 896 experts, top-16 SELECTION on the
biased $s_e + b_e$, WEIGHTING by the unbiased
$w_i = \gamma\, s_i / \sum_{j \in \mathcal{T}} s_j$. Selection and
weighting reading different scores is why generic top-k helpers don't
fit.

`Routing` backends (both emit `run_moe`'s wire format directly):

- **`RADIX` (default)** — the unchanged SGLang `RouteRadixKernel`:
  byte-radix COUNTING top-k, no serial 16-reduction chain. Adopted
  after the full routing study
  (`kimi_k3_layer/kernels/routing_permutation_results.md`): runs at
  measured SOL (0.9–1.1x the mandatory-work probe), wins every batch
  size. Kernel A/B vs the previous CuTeDSL default: 3.4 vs 4.7 us at
  B=1, 46 vs 50 at 16384, expert IDs bit-identical. Whole-layer:
  −1.1 us at B=1; hidden by the side stream at B>=32.
- **`REFERENCE`** — the stock routing_method (in-kernel trtllm-gen
  routing; its 896-expert scores stage is a single-cluster ~47 us
  kernel at B=64, which precomputed routing skips). The in-layer A/B
  baseline.

Two former backends were REMOVED (radix dominates both at every
size): `CUTE_EXACT` (the CuTeDSL exact-top-k router, previous
default) and `TRITON` (one-CTA-per-token serial 16-pick chain). Both
stay research-only in this repo (`kernels/routing_cutedsl.py`,
`kernels/routing_triton.py`) and are not merged into production
repos.

## 3. Prefill strategy (TP-8, tokens > 256)

| tokens | plan |
|---|---|
| 257 .. 2047 | `PrefillFC1.FULL` + **`PrefillTail.FC2_SHARD`**: full fc1, then AR(latent) → column-window norm → this rank's fc2-shard `addmm_` into the shared partial → ONE output AR reducing shared + fc2 partials together. Same total wire as the old packed tail (3584+7168 = 10752), 1/8 the fc2 FLOPs. This DELETED the former ship-the-baseline window (+16.6/+22.3% at 512/1024 where the packed tail managed +2..+6%) |
| >= 2048 | `PrefillFC1.SHARDED` (boundary moved 4096 → 2048 on Aug-19): fc1 column shard + **multimem gather** on the prefill stream, feeding **`gather_quant`** — one fused Triton pass that interleaves the rank-major blocks AND quantizes to (fp8, sf), bit-exact vs the trtllm op and 12-119 us faster than interleave+quantize (kernels/gather_quant_results.md). Gather impl chosen by 3x-repeated in-layer A/B at the REAL [B,448] payloads: multimem beats the DMA mover +37/+4/+217 us at 2048/4096/16384 and is stable where DMA flaps at 16K; DMA wins only at 8192 (−20 us, conceded for one-rule simplicity — open corner). Split zero-copy AR staging, same fc2-shard tail. FULL wins at 512-1024 (gather fixed cost > the fc1 FLOP cut there) |
| >= 8192 | + `ExpertBackend.NATIVE` (marginal at exactly 8192, clearly positive at 16K — the trtllm grouped GEMMs overtake FlashInfer at large per-expert M) |

GEMM regime logic behind the boundaries (code-independent roofline):
the fixed-shape latent GEMMs go compute-bound between B=256 and 512
(analytic B* ≈ 280 on B200); fc1/fc2 sharding buys only the flat
weight-read cut at decode sizes but a real 1/world FLOP cut at
prefill. The fc2 shard needs NO extra wire (its output AR replaces
the packed AR), which is why it pays from 512; the fc1 shard needs a
gather, which only pays once (a) the FLOP cut exceeds the gather's
fixed cost AND (b) the gather stops displacing compute — the DMA
mover delivers (b), moving the boundary 4096 → 2048.

Scheduling rules (profiled, load-bearing): the copy-engine gather
(pure DMA) hides under anything; the shared chain runs EARLY,
overlapping the latency-bound routing + expert-metadata phase;
nothing may be scheduled into an SM-resident AR window — compute
there displaces the AR and costs more than it hides.

Open items, in value order:
1. **Latent-AR overlap in the FC2_SHARD tail** — that AR is the
   tail's one exposed collective; the shared down GEMM is the natural
   partner.
2. **8192 gather corner**: the DMA mover beats multimem by ~20 us
   there (only size where it does); root-cause the multimem dip
   before special-casing.
3. **gather_quant residual**: 2.6x its bytes bound (strided
   block-gather reads) — worth a pass only if the gather stops
   hiding it.

## 4. Collectives: measured picks, overlap, and the shared IPC pool

Full per-op tables: `communication/RESULTS.md`. The layer-relevant
findings:

- **all_reduce**: `trt` MIN_LATENCY one-shot wins B<=32 (5.1-9.3 us —
  flashinfer:1shot is 5-6% behind from Lamport sentinel clearing);
  NVLS multimem from B~64-128 up. This retired the old hand-picked
  `comm="fi"` — the layer states the op, the autotune supplies the
  backend.
- **all_gather** (after fixing a bench capacity-probe bug that had
  spuriously gated every non-NCCL backend at bs>=4096):
  `torch_symm:multimem` <=64, `b10_copy_engine:sm` 128-4096, pure
  DMA `b10_copy_engine` at 8192+ (1.0x SOL, 720 GB/s measured peak).
  `torch_low_contention` is never the best row.
- **AG overlapped with an independent GEMM** (the §3 design driver):
  the DMA mover hides a full compute chain nearly for free at
  1-2K tokens (+2.6-3.1 us dilation); the SM-copy mover is the WORST
  overlap partner everywhere (fights the GEMM for SMs, −3..−5
  efficiency); NCCL and multimem sit between. From 4096 even DMA
  shows +124-149 us dilation (HBM contention — §3 open item 3).
- **Shared IPC pool**: both b10 movers now borrow the torch_symm
  output staging buffer (one symm allocation instead of three;
  rendezvous is idempotent so the lazy build stays collective-free) —
  bit-exact validated 8..16384, timings unchanged, 2x max_numel of
  symm HBM returned.
- **EP is a different regime**: `moe_ep_overlap_results.md` — inside
  DeepGEMM's mega kernel at EP=8 the dispatch/combine comm is fully
  hidden only at B/rank<=256 (91-95% of the NCCL-equivalent cost);
  from B>=1024 exposure equals the full wire time (marginal slopes
  additive). Mega's mid-range win is a leaner pipeline + faster
  in-kernel wire, not overlap.

## Appendix A. Removed variants and negatives (the record)

| candidate | verdict |
|---|---|
| `Routing.CUTE_EXACT`, `Routing.TRITON` | removed: radix dominates at every size; research-only in `kernels/routing_{cutedsl,triton}.py` |
| `DecodeFront.FUSED_FC1_SHARED_GATE_TRITON` | removed: Triton dual-out GEMM slower than CuTeDSL at every size; research-only at `kernels/dual_out_gemm_triton.py` |
| `DecodeFront.FUSED_FC1_GATE_CUTE` (no shared rows) | removed: dominated by the full fuse; never selected |
| `DecodeTail.FULL_FC2_SHARED_REDUCE` (+ its chain_ar plumbing) | removed: never selected; the multimem tail dominates it |
| `prefill_fused_column_gather_quantize` axis | removed with the NCCL col AG it configured: its over-capacity fallback lost 7-24 us to plain AG + expert-side quantize (quantize alone is only 7-29 us) |
| `kernels/quant_slice.py` | deleted as OBSOLETE, not a loser: served the old merged-GEMM strided slice; the MXFP8 recipe lives on in the col-AG epilogue and the planned interleave+quantize kernel |
| PACKED prefill tail as the plan | +2..+6% at best — superseded by FC2_SHARD (+17..+43%); remains the EXP ablation tail |
| routing+permutation fusion (two designs) | rejected with mechanisms (routing study report); region at 1.3-1.6x its reachable bound |
| SM-copy mover for overlapped gathers | worst overlap partner measured (§4) — DMA only |

## Appendix B. Measurement protocol

- Idle GPUs; CUDA-graph replay (100 iters/graph), MAX over ranks,
  mean over 8 random inputs (routing/EP skew is data-dependent).
- Whole-layer numbers are ALWAYS graph-timed. Host bubbles are a
  SEPARATE serving-mode issue and never enter strategy decisions;
  the measurement on record (`local_debug/moe_prefill_host_bubble.py`):
  ~0 at 8192/16384, +137/+190 us at 2048/4096, ~+300 us at decode
  sizes (graph-served in production) — eager prefill serving at
  2048-4096 needs graph bucketing, same as the KDA conclusion.
- Same-process A/B for sub-us deltas; sub-run flap on this shared box
  is real (a one-off 2x outlier at 16K never reproduced across five
  fresh captures) — repeat captures before believing an outlier.
- Correctness gates every size; near-tie top-16 flips vs the bf16
  gate are benign and tie-aware-excluded.

## Appendix C. Changelog (evidence for the shipped defaults)

- **Aug-11 re-search**: multimem tail from B=16, fused front through
  the decode cap, route_on_side_stream.
- **Aug-12 shipping validation** (superseded by §1's Aug-19 run).
- **Aug-19 routing study + RADIX adoption**; CUTE_EXACT/TRITON
  removed here and in the trt-llm fork (`optimized/k3`).
- **Aug-19 simplification sweep**: dead fronts/tails/flags deleted;
  decode cap 128 → 256; prefill decode-fields stripped.
- **Aug-19 fc2-shard prefill tail** (`PrefillTail.FC2_SHARD`): the
  baseline window deleted; +17..+26% across 512-4096 vs the packed
  tail's +2..+6%.
- **Aug-19 overlapped gather**: sharded boundary 4096 → 2048;
  gather impl settled by in-layer A/B at the real [B,448] payloads
  (multimem; the synthetic equal-legs study favored DMA — layer-real
  shapes flipped it); enabled by the comm findings (capacity-probe
  bug fix, overlap study, shared IPC pool).
- **Aug-19 `gather_quant` kernel**: fused block-interleave + MXFP8
  quantize, bit-exact vs the trtllm op, +12..+119 us at 2048-16384
  (kernels/gather_quant_results.md); final §1 numbers include it.
- Fork commits on `optimized/k3`: radix adoption, dead-flag removals,
  unfused AG quantize, fc2-shard tail + baseline-window removal
  (DMA-gather port pending — the fork's vendored comm package has no
  mover and fork-side TP8 validation isn't possible in these
  containers).
