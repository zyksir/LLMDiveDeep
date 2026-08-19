# Kimi-K3 KDA layer — mega-kernel study and results

Date: 2026-08-19
Qualification: measurements QUALIFIED (bench_b10_kimi_k3_kda_layer.py, B200, TP8 rank
shard, CUDA-graph decode / eager prefill as shipped); fusion analysis is a
pre-candidate study — no fused kernel written yet.
Prefix key: `trt_` = TensorRT-LLM modules, `b10_` = written in this repo.

## Operation contract (whole layer, one TP8 rank)

bf16 x `[B, 7168]` -> in_proj (7168 x 6288 merged qkvgfab GEMM) -> short
conv + f_b gate proj (128 x 1536) -> KDA gated-delta recurrence
(12 heads x 128, fp32 state 823 KB/seq) -> o_proj (1536 x 7168) -> bf16 y.
Weights: 112.6 MB/layer/rank. B<=128 rows are decode (B independent
sequences, state read+write per sequence); B>=256 rows are prefill (chunked
scan, state per 64-token chunk).

## Implementations

- **trt_kda (reference)**: TRT-LLM checkout modules — separate kernels for
  conv, gating, recurrence, norm, plus the three GEMMs.
- **b10_kda**: this repo's fused decode kernel (conv+gate+recurrence fused)
  and prefill pipeline over the same GEMMs.

## SOL (measured; the layer viewed as its GEMMs plus mandatory state bytes)

SOL(B) = max of measured probes (`kimi_k3_layer/kernel_research/
kimi_k3_kda_megakernel/sol_probe.py`): the layer's three GEMMs executed with
their exact shapes (reads every weight once, does the exact GEMM flops; the
cheaper of one-big-GEMM and three-stage formulations), and a copy kernel
moving the recurrence/conv state (per sequence at decode, per 64-token chunk
at prefill) plus activations. Decode bound is flat ~24-25 us: 112.6 MB of
weights at ~4.7 TB/s. The recurrence's intra-chunk matmul flops (~20 GFLOP
at 16K tokens) are below the GEMM term and not separately bounded.

## Results (us, full range; bold = winner)

| B | trt_kda eager | b10_kda eager | b10_kda GRAPH | SOL (measured) | best x SOL |
|---:|---:|---:|---:|---:|---:|
| 1 | 41.1 | **36.0** | (= eager; decode is graphed) | 23.7 | 1.5x |
| 2 | 43.1 | **36.5** | (= eager; decode is graphed) | 25.5 | 1.4x |
| 4 | 44.0 | **37.4** | (= eager; decode is graphed) | 24.9 | 1.5x |
| 8 | 43.8 | **37.5** | (= eager; decode is graphed) | 24.8 | 1.5x |
| 16 | 46.6 | **40.3** | (= eager; decode is graphed) | 25.5 | 1.6x |
| 32 | 50.7 | **46.4** | (= eager; decode is graphed) | 25.4 | 1.8x |
| 64 | 64.3 | **55.2** | (= eager; decode is graphed) | 24.3 | 2.3x |
| 128 | 90.9 | **69.9** | (= eager; decode is graphed) | 24.4 | 2.9x |
| 256 | 804.2 (graph FAILS) | 698.8 | **116.3** | 31.0 | 3.8x |
| 512 | 802.5 (graph FAILS) | 408.1 | **151.1** | 45.2 | 3.3x |
| 1024 | 793.2 (graph FAILS) | 407.2 | **246.1** | 85.5 | 2.9x |
| 2048 | 892.9 (graph FAILS) | 972.2 | **416.6** | 159.3 | 2.6x |
| 4096 | 1245.5 (graph FAILS) | 1901.5 | **781.1** | 318.9 | 2.4x |
| 8192 | 1998.0 (graph FAILS) | 1524.4 | **1508.3** | 620.7 | 2.4x |
| 16384 | 3514.4 (graph FAILS) | 3029.4 | **3007.6** | 1314.9 | 2.3x |

Rule-17 note: decode rows were always CUDA-graph timed; prefill rows were
eager in the original bench. The GRAPH column re-times b10 prefill under
CUDA graphs; the trt reference prefill CANNOT be captured (its chunk
pipeline performs capture-invalidating host-side operations), so its honest
GPU-only reference is the profiled kernel sum (537/906 us at 2048/4096 —
still slower than b10's 417/781 graph numbers).

Why the winner wins per regime (updated with graph-mode prefill):
- **b10_kda wins EVERY size once host bubbles are removed.** The former
  B=256 "floor" (699-804 us) and the 2048-4096 "regression" were entirely
  CPU launch overhead: graphed b10 prefill is 116-3008 us, monotonic in B
  as it must be, at 2.3-3.7x the GEMM+state bound. Production consequence:
  prefill must ship graph-bucketed (confirming the repo's earlier
  graph-bucketed serving probe).

- **Decode (B<=128): b10_kda** — its fused conv+gate+recurrence removes
  kernel launches and intermediate traffic from the non-GEMM chain; the
  remaining 1.4-2.9x over SOL IS that chain (GEMMs alone would be ~24 us),
  and it grows with B because per-sequence state traffic and the recurrence
  kernel scale with B while the 112.6 MB weight read (the SOL floor) does
  not.
- **Prefill gap analysis (graph mode):** 2.3-3.7x over the GEMM+state
  bound, shrinking as B grows — the fixed non-GEMM chain (conv, chunked
  recurrence intra-chunk compute, norm) amortizes; the residual at 16K is
  the chunk recurrence itself, which the GEMM-only bound does not price.

## Mega-kernel fusion ladder (analysis; target = SOL above)

The decode gap (best 36-70 us vs 24-25 us bound) is the non-GEMM chain, so
the fusion ladder attacks it in three steps:

1. **f_b + kda — IMPLEMENTED AND REJECTED** (candidate
   `b10_kda_decode_conv_gated_fusefb_cutedsl.py`, A/B in
   `bench_b10_kda_fusefb.py`): the in-kernel GEMV runs 2.2-3.3x SLOWER than
   GEMV+kernel (12.1 vs 5.4 us at B=1; 117.8 vs 35.6 at B=128). Measured
   why: the gate value is needed in the kernel's latency-critical prologue
   (conv gates phase 0 gates the CTA barrier), so the injected 512-FMA dot
   product extends the critical path instead of hiding; and each (b,h) CTA
   re-reads its 32 KB W_fb slice (~49 MB of L2 traffic at B=128) where
   cuBLAS reads the 0.4 MB weight once for the whole batch. The original
   ~3-6 us estimate wrongly assumed the GEMV could overlap. Correct
   remaining option for this step: PDL-chain the cuBLAS GEMV with the
   decode kernel (launch-gap hiding only, ~0.5-1 us).
2. **+ o_proj** (recurrence output into the 1536x7168 GEMV): the o_proj
   weight read (22 MB) streams while the recurrence computes — removes
   another launch + the `[B,1536]` output round-trip. Expected ~4-8 us.
3. **+ in_proj** (full layer, one kernel): in_proj is 80% of the weight
   bytes; fusing it means one kernel streams all 112.6 MB once while
   computing conv/gate/recurrence in registers/shared per token — the
   whole-layer-as-one-GEMM ideal (~24 us at decode, -33..-65% vs today).
   High risk: register pressure across four stages, and the conv+recurrence
   dependency serializes against the in_proj streaming unless tokens are
   pipelined across CTAs (v1/v2 of the routing study showed exactly how
   per-warp serialization and fence costs can eat such fusions).

Recommended order: measure step 1 first (small, isolates the fusion
mechanics); step 3 only if steps 1-2 confirm the launch/round-trip
accounting. Independently, graph-capture the prefill buckets to reclaim the
B=2048-4096 wall-clock loss (pure CPU launch overhead, profiled above).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 python3 kimi_k3_layer/bench_b10_kimi_k3_kda_layer.py \
    --shards tp8 --token-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384  # ~96 s
CUDA_VISIBLE_DEVICES=0 python3 \
    kimi_k3_layer/kernel_research/kimi_k3_kda_megakernel/sol_probe.py
```
