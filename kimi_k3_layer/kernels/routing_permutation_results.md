# Kimi-K3 routing+permutation report

Date: 2026-08-19
Qualification: QUALIFIED
Decision: route = unchanged `sgl_radix` (adopted; it is at SOL).
Permutation = `b10_permute_v1_pdl` for B<=256, unchanged `trt_moe_sort`
from B=512 (crossover measured between 256 and 512). Fusing route+permute
into one kernel was implemented twice and measured slower — keep two
kernels. Shipped in the layer as a measured per-batch dispatch (radix
everywhere except 32<B<=96); full-range layer A/B in the Integration
section.

Prefix key: `sgl_` = SGLang, `trt_` = TensorRT-LLM, `vllm_` = vLLM,
`b10_` = written in this repo.

## Operation contract

FP32/BF16 logits `[B, 896]` + FP32 bias `[896]` ->
(a) route: top-16 expert IDs + weights (exact FP32 sigmoid, select by
score+bias, ties to lower ID, weights = score/sum);
(b) permutation: expert-major maps and grouped-GEMM tile descriptors for the
112 local experts `[224, 336)`, tile 128, `-1` for non-local.
One B200, CUDA-graph medians, preallocated buffers.

## Ecosystem implementation survey — what each kernel actually does

Route (all produce IDs+weights only):

- **SGLang `RouteRadixKernel`**: 224 threads per token, 4 experts per
  thread in registers; finds the top-16 boundary by byte-wise radix
  *counting* (a few histogram passes, no sorting, no serial pick loop).
  Fastest because selection is parallel counting, not 16 dependent picks.
- **vLLM `grouped_topk Tier<896,16>`**: 256 threads per token; 4 warps each
  select a local top-16 of their 224-expert slice, then one warp merges the
  64 candidates. ~30% slower than radix: the merge is a second serial stage.
- **TRT `noaux_tc_op`**: generic top-k, 16 dependent block-wide
  max-reductions per token. 4-8x slower; supplemental only.

Route+permutation in one launch:

- **`trt_routing_custom`** (the production fused front): one 1024-thread
  block (B<=4) or an 8-block cluster (B<=256) does sigmoid+top-k+histogram+
  scan+maps in one kernel — but it routes each token with ONE warp (a long
  serial instruction stream per token) and pays cluster synchronization, so
  it loses to every two-kernel composition at every size we measured.

Permutation from IDs (all emit the same map/descriptor ABI):

- **`trt_moe_sort`** (what SGLang's trtllm-gen path runs): the routingCustom
  pipeline minus scoring; an 8-SM cluster kernel up to 8K tokens (grows with
  tokens, mis-tuned threshold for top-16), a 140-SM cooperative kernel above.
- **candidate `b10_permute_v1_pdl`**: one 512-thread CTA — shared
  histogram, one warp-level scan, parallel scatter — launched with PDL so
  its prologue hides under the route kernel. Wins B<=256 because the whole
  job is tiny; the multi-SM machinery above only pays off at prefill sizes.

## SOL model (reachable by construction)

SOL(stage, B) = measured time of a probe kernel that performs only the work
every implementation must do: route = read all B x 896 logits + exact
sigmoid+bias per expert + a reduction; permutation = read B x 16 IDs +
112-bin histogram + scan for B<=512; above that (multi-CTA regime) a real copy
kernel moving the permutation's mandatory bytes (ids read + map writes).
Region = route + permutation - one launch (PDL).
These probes run on this GPU in the same graph mode as the results, so the
bound is attainable; naive launch-floor/bytes bounds are excluded as
unreachable. Decision before candidate work: PROCEED for the permutation
(clear headroom vs moe_sort), STOP for the route (radix already at bound).
Harness: warm setup 1.1 s, full 15-shape matrix (B=1..16384) ~2 s measured time.

## Results (us; "x SOL" = latency / reachable bound)

Route:

| B | sgl_radix | vllm_grouped_topk | trt_noaux | SOL | sgl_radix x SOL |
|---:|---:|---:|---:|---:|---:|
| 1 | **2.08** | 2.95 | 9.24 | 2.25 | 0.9x |
| 2 | **2.45** | 2.98 | 9.63 | 2.28 | 1.1x |
| 4 | **2.43** | 2.97 | 9.62 | 2.23 | 1.1x |
| 8 | **2.43** | 2.94 | 9.59 | 2.25 | 1.1x |
| 16 | **2.46** | 2.98 | 9.64 | 2.24 | 1.1x |
| 32 | **2.45** | 3.20 | 9.72 | 2.25 | 1.1x |
| 64 | **2.44** | 3.29 | 9.72 | 2.29 | 1.1x |
| 128 | **2.48** | 3.30 | 10.19 | 2.31 | 1.1x |
| 256 | **2.92** | 3.99 | 10.23 | 2.33 | 1.3x |
| 512 | **3.09** | 5.47 | 13.05 | 2.39 | 1.3x |
| 1024 | **4.67** | 8.09 | 18.41 | 2.51 | 1.9x |
| 2048 | **7.36** | 14.00 | 30.91 | 4.25 | 1.7x |
| 4096 | **12.74** | 25.75 | 53.82 | 6.86 | 1.9x |
| 8192 | **23.67** | 49.18 | 99.57 | 11.83 | 2.0x |
| 16384 | **45.12** | 95.49 | 193.55 | 21.66 | 2.1x |

Permutation:

| B | trt_moe_sort | b10_permute_v1_pdl | SOL | best x SOL |
|---:|---:|---:|---:|---:|
| 1 | 3.06 | **2.01** | 1.34 | 1.5x |
| 2 | 3.18 | **2.00** | 1.28 | 1.6x |
| 4 | 3.29 | **2.01** | 1.48 | 1.4x |
| 8 | 4.60 | **2.02** | 1.47 | 1.4x |
| 16 | 5.13 | **2.09** | 1.50 | 1.4x |
| 32 | 5.48 | **2.17** | 1.49 | 1.5x |
| 64 | 6.09 | **2.55** | 1.71 | 1.5x |
| 128 | 4.51 | **2.77** | 1.70 | 1.6x |
| 256 | 5.84 | **3.85** | 1.90 | 2.0x |
| 512 | 5.90 | **5.86** | 2.38 | 2.5x |
| 1024 | **6.28** | 9.91 | 1.07 | 5.9x |
| 2048 | **7.22** | 20.15 | 1.18 | 6.1x |
| 4096 | **8.81** | 41.86 | 1.19 | 7.4x |
| 8192 | **12.32** | 81.07 | 1.30 | 9.5x |
| 16384 | **7.33** | 159.84 | 1.96 | 3.7x |

Whole region:

| B | sgl_radix+b10_permute(PDL) | sgl_radix+trt_moe_sort | trt_routing_custom (fused) | SOL | best x SOL |
|---:|---:|---:|---:|---:|---:|
| 1 | **3.99** | 5.39 | 11.10 | 3.01 | 1.3x |
| 2 | **4.35** | 5.87 | 11.23 | 2.97 | 1.5x |
| 4 | **4.35** | 5.98 | 11.50 | 3.13 | 1.4x |
| 8 | **4.35** | 7.58 | 11.66 | 3.14 | 1.4x |
| 16 | **4.62** | 7.62 | 12.25 | 3.16 | 1.5x |
| 32 | **4.52** | 8.23 | 12.74 | 3.16 | 1.4x |
| 64 | **5.29** | 8.85 | 13.31 | 3.41 | 1.6x |
| 128 | **5.36** | 7.46 | 11.94 | 3.42 | 1.6x |
| 256 | **7.07** | 9.07 | 13.08 | 3.65 | 1.9x |
| 512 | 9.51 | **9.22** | 17.41 | 4.19 | 2.2x |
| 1024 | 15.27 | **11.56** | 22.55 | 2.99 | 3.9x |
| 2048 | 29.31 | **15.15** | 23.95 | 4.85 | 3.1x |
| 4096 | 59.71 | **22.33** | 38.78 | 7.47 | 3.0x |
| 8192 | 114.78 | **37.75** | 61.01 | 12.55 | 3.0x |
| 16384 | 225.89 | **52.90** | 113.05 | 23.03 | 2.3x |

Why the winner wins per regime:
- **Route, all B: sgl_radix** — parallel byte-radix counting has no serial
  pick loop, so at decode it sits at the mandatory-sigmoid bound; at prefill
  it degrades to ~2x the bound only because its 224-thread-per-token layout
  is latency-shaped, not streaming-shaped (recorded, low value to fix).
- **Permutation, B<=256: b10_permute_v1_pdl** — the whole job fits one CTA;
  a single small kernel with a PDL-hidden prologue beats trt_moe_sort's
  8-SM cluster machinery, which only amortizes at prefill sizes.
- **Permutation, B>=512: trt_moe_sort** — multi-SM histogram/scatter beats
  any single-CTA layout once the expanded index space is tens of thousands.
- **Region: the two-kernel compositions beat trt_routing_custom at every
  size** because routingCustom routes each token with one warp (serial
  instruction stream) and pays cluster synchronization.

Correctness: every method matched the FP32 oracle exactly on IDs and within
dtype rounding on weights (<=3e-8 FP32) across random/tied/concentrated/
no-local cases; all maps bijective and mutually consistent.

## Why the remaining gap is not recoverable

- **Route: there is no gap.** Radix equals the probe (0.9-1.1x at decode);
  the probe IS the mandatory sigmoid instruction stream. (At prefill radix
  is 2.1x the bound — its per-token 224-thread layout does not stream at
  full bandwidth — but 23 us of headroom under millisecond expert GEMMs is
  not worth a kernel.)
- **Permutation: the 1.3-1.6x is the outputs.** The probe omits the
  mandatory map writes (-1 padding fill, scatter with atomics, tile
  arrays); those writes are the difference, measured by ablation. A tighter
  kernel was attempted (int4 fill, id caching) — both within noise.
- **Fusion cannot close it.** Two fused candidates were built: v1 (one CTA,
  28 keys/lane) lost to per-warp instruction serialization (21-119 us);
  v2 (route CTAs + last-CTA epilogue) lost because the all-CTA wait + fence
  chain costs more than the 0.58 us launch it removes (6.9-9.7 us). TRT's
  own fused kernel loses for the same reason. The two-kernel + PDL design
  is the measured optimum for this contract.

## Alternatives and decision

Rejected: b10_fused_v1, b10_fused_v2 (above), TRT fused front (slowest
region everywhere), vLLM route (dominated by radix), Triton branch router
(2.6-3x slower than radix), local CuTe routers (prior study). Chosen:
radix + b10_permute_v1_pdl (B<=256), sgl_radix + trt_moe_sort (B>=512); crossover
measured between 256 and 512.

## Integration (TP8 MoE layer, full range, opt latency us)

Layer routing strategies: `trtllm` = raw logits to the in-kernel front,
`b10 triton` = the branch Triton router, `sgl_radix + dispatch` = the new
default (radix everywhere except the measured 32<B<=96 window where the
in-kernel front is used). Expert selection is bit-identical across configs
at every size (identical rel_err vs the fp32-gate baseline).

| B | trtllm baseline (raw logits) | old default (b10 triton router) | new default (sgl_radix + dispatch) | delta new vs old |
|---:|---:|---:|---:|---:|
| 1 | 76.7 | 51.7 | **48.8** | -2.9 |
| 2 | 81.4 | 52.3 | **47.4** | -4.9 |
| 4 | 83.7 | 52.9 | **50.1** | -2.8 |
| 8 | 88.1 | 63.9 | **61.7** | -2.2 |
| 16 | 93.9 | **78.7** | 100.6 | +21.9 |
| 32 | 106.7 | 105.1 | **89.9** | -15.2 |
| 64 | 116.5 | **108.8** | 109.1 | +0.2 |
| 96 | 124.4 | 120.5 | **119.9** | -0.5 |
| 128 | 131.7 | 142.0 | **133.4** | -8.6 |
| 256 | 1593.8 | 1690.9 | **557.7** | -1133.2 |
| 512 | 730.1 | 763.1 | **561.8** | -201.3 |
| 1024 | 734.3 | 790.1 | **553.9** | -236.2 |
| 2048 | 735.8 | 793.7 | **552.0** | -241.8 |
| 4096 | 952.8 | 814.2 | **715.2** | -99.0 |
| 8192 | 1838.6 | 1168.4 | **1164.3** | -4.1 |
| 16384 | 3604.2 | 2183.2 | **2066.3** | -116.9 |

Why the regime switches happen as B grows:
- **B<=32**: routing is a visible fraction of a ~50-90 us layer; sgl_radix's
  2.1-2.5 us route (vs triton's ~6, in-kernel's ~11+) wins directly.
- **32<B<=96**: PROFILED (gzipped traces
  `local_result/moe_kimi_k3_tp8_b64_{base,opt}_graph.trace.json.gz`): the
  in-kernel routingCustom actually costs ~17-32 us here — 10x radix — but it
  executes inside the expert runner where it hides under the overlapped
  shared-expert + allreduce_fusion chain on the aux stream, while radix's
  2.4 us sits exposed on the critical path before run_moe. Net ~1-2 us for
  the raw-logits front — the one window where the handoff does not pay.
- **B=128**: radix wins again (-8.6): the in-kernel cluster router grows
  with tokens past what the overlap window hides, while radix stays ~2.5 us.
- **B>=256**: the sequence-parallel prefill path dominates everything
  (copy-engine RS+RS+AG tail); routing choice is secondary there, and the
  new default rides the prefill path correctly (557-2066 us vs the old
  config's 1690-2183). B=256 sits at the prefill-path floor (~555 us flat
  through 2048 — the tail is latency-bound, not token-bound, until 4096).
- **Variance note**: B=16 (+21.9) and B=32 spreads across runs are larger
  than the deltas (EP-skew with n_inputs=4); the strategy search re-checks
  these points — treat single-run mid-size deltas under ~10 us as noise.

Raw data: `kimi_k3_layer/local_result/routing_permutation_bench*.json`,
`bench_moe_kimi_k3_tp8_*.csv`. History and full method detail: git log of
this file.

## Reproduce

```bash
# inside the TRT-LLM container, one visible GPU
CUDA_VISIBLE_DEVICES=0 python3 kimi_k3_layer/kernels/bench_routing_permutation.py \
    --include-fork --include-candidates   # main matrix + JSON (~12 s)
CUDA_VISIBLE_DEVICES=0 python3 kimi_k3_layer/kernels/bench_routing_permutation.py \
    --impls trt_noaux --json .../noaux.json  # supplemental, own process (~30 s)
# integration subset (~49 s): mpirun -n 8 kimi_k3_layer/bench_moe_kimi_k3.py \
#   --sizes 32,64,128,256 --iters 60 --n-inputs 2 --csv-only
```
