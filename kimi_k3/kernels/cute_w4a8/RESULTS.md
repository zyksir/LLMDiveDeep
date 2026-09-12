# CuteDSL candidates for the tp_te expert GEMMs (2026-08-30)

Target: the two routed-expert GEMMs of tp_te decode (W4A8_MXFP4_MXFP8,
per-rank slices, contiguous-grouped over ~B+15 distinct experts).
Baselines: trtllm-gen `bmm_*` kernels measured standalone AND in-layer
(matching within 15%): GEMM1 19.3us / 3.6TB/s, GEMM2 22.25us / 1.55TB/s
at B=32 (47 groups x ~11 tokens).

## SOL recap (B=32)
- GEMM1 (w1w3, 68.7MB): at the achievable floor. Headroom ~0-10%.
- GEMM2 (w2, 34.4MB): 1.55TB/s = 2.3x below GEMM1's achieved BW. Mechanism:
  trtllm-gen's t128x8 tile (N=8!) against N=3584/K=384; bf16 cuBLAS probe
  reaches ~6TB/s at the same orientation -> kernel tile choice, not hardware.
- routing pair 12.9us (SOL ~3), quantize+fill 4.3us: fusion targets.

## CuteDSL grouped GEMM (fork's blockscaled_contiguous_grouped_gemm),
## GEMM2 shape 11x3584x384x47, correctness PASS, cold L2:

| config | time | vs trtllm-gen GEMM2 22.25us |
|---|---|---|
| tiler 128x128, cluster 1x1 | **16.59us (2.07TB/s)** | **1.34x** |
| tiler 128x256 | 17.16us | 1.30x |
| tiler 256x128, 2cta | 26.74us | 0.83x |
| (warm-L2 reference: 13.66us — flattered, do not quote) | | |

GEMM1 shape 11x768x3584x47: 17.93us (3.83TB/s) vs 19.3 -> 1.08x — confirms
GEMM1 is already at floor; CuteDSL matches it.

## Full-range validation (B=1..128) + dispatch policy

Cold-L2, per-B grouped shapes from the perfect-router distinct counts,
correctness PASS at every row (both kernels):

| B | distinct | m/grp | trtllm-gen GEMM2 | CuteDSL GEMM2 | winner |
|---|---|---|---|---|---|
| 1 | 16 | 1 | **5.92** | 13.97 | trtllm-gen 2.4x |
| 2 | 17 | 2 | **6.42** | 13.03 | trtllm-gen 2.0x |
| 4 | 19 | 3 | **6.71** | 13.68 | trtllm-gen 2.0x |
| 8 | 23 | 6 | **7.88** | 13.77 | trtllm-gen 1.7x |
| 16 | 31 | 8 | **12.62** | 13.08 | ~tie |
| 32 | 47 | 11 | 22.25 | **16.69** | CuteDSL 1.33x |
| 64 | 79 | 13 | 35.45 | **24.74** | CuteDSL 1.43x |
| 128 | 143 | 14 | 50.67 | **41.47** | CuteDSL 1.22x |

Mechanism on both sides: at tiny B, trtllm-gen's small N=8 tile is
latency-optimal (2.8-3.2 TB/s achieved at B<=8!) while the CuteDSL kernel
pays its 128-rows-per-group M padding (at B=1 that is 128x wasted MMA per
group) and sits latency-floored ~13-14us flat until B=32. The trtllm-gen
tile pathology only emerges as the distinct-expert count grows (>~40
groups x serial N loop). GEMM1: trtllm-gen tracks the byte floor at EVERY
B (6.9 -> 46.4us) — no reason to replace it anywhere.

**Dispatch rule: num_tokens <= 16 -> trtllm-gen (unchanged); num_tokens
>= 32 -> CuteDSL grouped kernel.** Decode CUDA graphs are per-batch-size
buckets, so this is a compile-time bucket choice with zero runtime
dispatch cost; the boundary sits safely inside a ~tie region (B=16).

## The W4A8 mixed-dtype finding

The DSL exposes the mixed atom (`MmaMXF8F6F4Op(a=e4m3, b=e2m1)` constructs),
and `make_blockscaled_trivial_tiled_mma` accepts separate a/b dtypes — but
the F8F6F4 MMA family requires fp4 operands in 8-BIT CONTAINERS in smem,
while the kernel (and packed model weights) use MXF4's 2-per-byte packing.
Candidate `grouped_gemm_w4a8.py` (4 call sites + check removed) compiles
and runs but mis-reads B (huge mismatch) for exactly this reason. A true
packed-weights W4A8 CuteDSL kernel needs an unpack pipeline stage
(TMA packed -> smem, CTA nibble-expand -> byte-container smem -> MMA):
multi-day work, est. landing 15-17us.

The sweep numbers above are the pure-MXF4 recipe (activations quantized to
fp4 instead of fp8) — a SEMANTIC CHANGE for GEMM2's input that needs
approval + accuracy evaluation before it can be the production path.

## Menu (per-B=32-layer impact; expert op is 58us of the 118us layer)
1. Tiling alone via CuteDSL, no semantic change (needs the unpack stage
   for W4A8): GEMM2 -5.7us, GEMM1 -1.4us -> ~7us (12% of expert op).
2. + routing/quantize fusion into GEMM1 prologue (TileRT pattern): ~10us.
3. Activation-fp4 GEMM2 (approval needed): reaches today's 16.6us with
   zero new kernel work.
4. SM103 `SM103MmaMXF4Op` (Ultra FP4, K=96 GB300-native): untested lever
   for the fp4 variant.

Artifacts: grouped_gemm_w4a8.py (mixed candidate), run_w4a8_grouped_gemm.py
(harness, --b_dtype), logs /node-storage/var/w4a8_gate{1,2}.log,
sol_experts2.log, gemm2_geometry_probe.py.
