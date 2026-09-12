# Kimi-K3 KDA layer — decode + prefill baselines, profiles, and measured wins

Single B200 (`model-performance` box, locked clocks 1965 MHz), trt-dev
container (1.3.0rc23). K3 TP8 shard: 12 local heads, head_dim 128,
hidden 7168. No MLA work (out of scope by request). All layer-level
claims should be roughly HALVED when talking end-to-end: the model
interleaves other layer types (rule of thumb agreed for this project).

## 1. Decode (`bench_b10_kimi_k3_kda_layer.py`, graph replay, B=1..16)

Baseline = the production `trt_fused` path (ONE Triton kernel for
conv4+SiLU + KDA recurrence + gated RMSNorm, between the real
projection GEMMs and AttnRes). Opt = `b10_fused` (linear_attn's
CuTeDSL kernel: same fusion, one CTA per (batch, head), state resident
in registers). All backends pass output/conv-state/ssm-state/AttnRes
correctness vs `trt_fused`.

| B | trt_fused | b10_fused | layer gain | ~e2e (halved) |
|---|---|---|---|---|
| 1 | 41.7 us | 36.1 us | **+13.5%** | ~7% |
| 2 | 44.2 | 37.2 | **+15.8%** | ~8% |
| 4 | 44.8 | 37.9 | **+15.5%** | ~8% |
| 8 | 44.7 | 38.7 | **+13.4%** | ~7% |
| 16 | 47.0 | 41.2 | **+12.2%** | ~6% |

Per-kernel profile at B=8 (one iteration):

| stage | trt_fused | b10_fused | roofline-ish |
|---|---|---|---|
| in_proj GEMM `[8,7168]x[7168,6288]` | 17.9 + 4.0 (splitK)* | 17.6 + 3.5* | ~11 us (near roofline: standalone cuBLAS runs it at 11.3-13.8 us = ~7.5 TB/s; *trace durations are PDL-inflated — kernels overlap, so per-kernel sums exceed the wall) |
| KDA core (conv+recurrence+norm) | 12.2 (Triton) | **4.7 (CuTeDSL)** | near floor |
| attn_res | 7.2 | 6.9 | small |
| o_proj (22 MB) | 6.0 | 6.1 | ~2.8 us |
| f_b_proj | 3.0 | 3.1 | — |

The b10 kernel already collapsed the core 2.6x. A standalone GEMM
swap test (optional `local_debug/kda_decode_gemm_swap.py` evidence)
DISPROVED the obvious next
lever: cuBLAS is already near-roofline on both projections (the
CuTeDSL tall-GEMM only wins at B=1, +2.7 us). The real remaining
decode levers are LATENCY-bound: `attn_res` (7.2 us for ~1.5 MB of
traffic — a fused low-latency kernel should land ~3 us; CuTeDSLGen
spec written: `attn_res_spec.md`) and folding the 3 us `f_b_proj`
GEMV into the b10 core kernel. Estimated landing: ~+25-30% layer —
the +20% decode target is reachable with those two.

## 2. Prefill (`bench_b10_kimi_k3_kda_layer.py`, eager, B=1, S=4k/8k/16k)

There was no layer-level prefill bench; built one. Baseline = the
production flow with BOTH of its blocking D2H syncs:

1. new-state reset via CUDA-bool boolean indexing
   (`state_indices[~has_initial_states]` → internal `nonzero()` →
   blocking D2H; `kda_mixer.py:762` at 496bc01f8e);
2. the Triton chunk pipeline's `prepare_chunk_indices` `.tolist()`
   (fires per distinct `cu_seqlens` — every real serving step; its
   `@tensor_cache` keys on tensor identity, and the bench hands a
   fresh cu_seqlens per call like an engine does).

Kernels: in_proj → `causal_conv1d_fn` → TRT-vendored FLA
`chunk_kda_with_fused_gate` (safe gate, fused l2norm) → gated RMSNorm
→ o_proj.

Opt = same projections/conv + sync-free reset (precomputed index
list, the checkout-HEAD form) + the b10 CuTeDSLGen chunk-prefill
champion with fused-Triton l2norm glue (`D^-0.5` folded into q, no
output pass).

| S | baseline | opt (eager) | gain | opt (graph) | gain |
|---|---|---|---|---|---|
| 4096 | 1222 us | 819 | **+33.0%** | 808 | **+33.9%** |
| 8192 | 1986 | 1588 | **+20.0%** | 1616 | +18.6% |
| 16384 | 3518 | 3140 | **+10.8%** | 3121 | +11.3% |

(The sync-free opt is CUDA-graph capturable; the baseline's D2H syncs
forbid capture — a structural advantage beyond the timings, since a
serving engine can only graph the opt form. Graph ~= eager here, so
the 16k residual is genuine kernel time, not launch gaps.)

**16k to +20%**: needs one more kernel round — a chunk-prefill respec
with the l2norm + safe-gate + scale glue fused as a kernel prologue
(est. -300-400 us; the glue is ~4 fp32 elementwise passes over
[S,1536] tensors today) and a look at the skinny f_b GEMM (K=128).
Est. landing ~+23-25%. First CuTeDSLGen attempt (gen_kdachunk_d)
produced a CORRECT fused-prologue kernel but at 1.28 ms vs the
incumbent's ~0.74 ms at 16k — fusing the prologue is not enough
without the INT21-class schedule; needs an improvement cycle.

Attribution (measured separately, kernel-level H=12):
`trtllm_kda_chunk` 325/615/1171 us vs `b10_kda_chunk_prefill`
189/376/~740 at 4k/8k/16k — the KERNEL SWAP is the story. The D2H
sync removal alone is ~1-3% of the layer at these sizes (kept in the
opt, but below the "write a kernel for it" bar on its own; its real
cost in an engine is the launch-pipeline bubble, which a layer bench
understates).

### Correctness note (why baseline-vs-opt cosine reads 0.90)

Optional `local_debug/kda_prefill_gate_debug.py` evidence, S=512,
against the exact fp32
recurrence oracle (`kda_recurrent_reference`):

- b10 kernel (canonical gate): cosine **0.999993**
- b10 kernel (safe gate + prenormed q/k, the layer's exact glue):
  cosine **0.999989**
- TRT Triton chunk pipeline: cosine **0.8235** in linear_attn's own
  harness (rated PASS there — accumulated bf16 chunk rounding).

So the 0.90 layer-level agreement is the BASELINE's accumulated
rounding; the opt path is the numerically cleaner one.

## 3. What's next (gated by estimated e2e win ≥1%)

| lever | est. layer | est. e2e | verdict |
|---|---|---|---|
| decode attn_res kernel | measured | — | NEGATIVE in-layer: CuTeDSLGen kernel (`kernels/attn_res_cutedsl.py`) is correct and 5.0 us standalone (vs ~7 TRT) but +1.5-2 us SLOWER inside the layer and breaks graph capture at B=1 (PDL-chain interaction). Vendored, default off (B10_ATTNRES_KERNEL). |
| decode f_b_proj fusion into b10 core | ~8% | ~4% | DO |
| prefill chunk kernel with fused l2norm/gate prologue | ~10-12% @16k | ~5% | DO — closes 16k to ~+23-25% |
| decode in_proj GEMM swap | +2.7 us at B=1 only | <1% | SKIP (cuBLAS near roofline; disproven) |
| D2H syncs alone | ~2% | ~1% | done as part of opt; not standalone |
| gated-norm / conv tweaks | <1% | <0.5% | skip per rule |

Reproduce:

```bash
# decode and prefill (single GPU; automatic boundary at 128 tokens)
CUDA_VISIBLE_DEVICES=4 python3 kimi_k3_layer/bench_b10_kimi_k3_kda_layer.py \
    --shards tp8 --token-sizes 1 2 4 8 16 128 4096 8192 16384
# prefill kernel menu (linear_attn)
cd linear_attn/benchmarks && python3 bench_kda_prefill.py --heads 12 \
    --batch-sizes 1 --seq-lens 4096 8192 16384
```
