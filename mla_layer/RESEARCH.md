# K3 attention (MLA + KDA) layer workstream

Directive (Yikai, 2026-08-23): after the MoE side is closed, open a
separate dir, build a standalone attention block aligned kernel-for-kernel
with production, then optimize with better kernels. Central idea to
evaluate: fuse attention with AllReduce (prefill and decode).

## Status: alignment prep (inventory extracted, standalone not built yet)

## What production runs per layer (serving trace m1-isl512-c8, bs8 decode)

K3's 93 layers are HETEROGENEOUS: ~75% KDA (linear attention) layers,
~25% MLA layers (per-layer kernel counts n=0.75 / 0.26 in the trace).
Attention-side inventory at bs8 (per average layer; medians carry
in-graph skew attribution):

| us/lyr | n/lyr | med | kernel |
|---|---|---|---|
| 17.3 | 1.00 | 17.3 | attention output AR (ar_fusion, grid[16]) — FIRST sync after attention, absorbs rank skew |
| 14.8 | 0.75 | 19.8 | KDA-side GEMM (nvjet splitK[700]) |
| 12.7 | 2.03 | 6.4 | kimi_k3::sm100::fwd_prod_v2 (MLA decode core) |
| 9.5 | 0.75 | 12.4 | _fused_kda_decode_kernel |
| ~11 | 2.02 | 6.0 | qkv/attn projections (nvjet [112]) — shared name w/ MoE, needs per-layer-type split |
| 3.2 | 0.52 | 7.5 | flashinfer rmsnorm (q/kv norm) |
| 2.3-2.8 | 0.26 | 9-11 | MLA split-kv + dsv3MinLatency + rope (MLA layers only) |

Caveats: dummy weights skew per-rank compute; the AR medians inflate with
absorbed skew (the 17.3 here vs 7.2 in the C=4 trace is batch- and
skew-dependent, not payload). Attention AR+residual+norm is ALREADY fused
(AttnRes decoder layers thread output_norm_weight/eps into the fused AR
op) — the naive fusion lever is taken.

## Transferable lesson from the MoE tail arc (MOE_FINALIZE_AR.md v3)

Do NOT replace trt's tuned one-shot ARs with custom lamport rounds at
small payloads (ours cost 10-12 us/round vs trt's ~7; monolithic fusion
lost in-layer at every bs). Fusion candidates must fuse cheap NEIGHBORS
into existing fast kernels, or attack kernels trt doesn't fuse.

## Idea list (each gets implement -> measure -> explain)

1. **o_proj epilogue -> AR staging write**: o_proj GEMM writes its
   partial straight into the AR's symm staging (removes one HBM
   round-trip + lets the AR kernel skip its stage-in). Zero-copy AR
   staging already exists in the fork for the MoE side
   (`_split_ar_staging`) — check if attention output uses it; if not,
   port it. LOW RISK, uses existing trt AR.
2. **KDA-side GEMM tuning**: the splitK[700] 19.8 us GEMM (0.75/lyr) is
   the single largest attention kernel — check its shape and whether a
   better nvjet variant/tiling exists (same class as the bs4 fc2
   variant bug).
3. **fwd_prod_v2 (MLA decode)**: 2 launches x 6.4 us; check fusibility
   of the pair (same kernel twice = q-absorbed and out-absorbed passes?)
   and split-kv reduction fusion.
4. **q/kv norm fusion**: flashinfer rmsnorm 7.5 us at 0.52/lyr — can it
   fold into the preceding/following GEMM epilogue?
5. **Prefill attention AR**: payload-bound regime -> multimem variant
   (1x egress) where the eager rank-skew findings apply
   (k3-prefill-eager-vs-graph).

## Alignment plan (the standing methodology)

1. `mla_layer/b10_kimi_k3_attn_layer.py`: standalone block = input norm
   -> qkv proj -> (MLA fwd_prod | KDA path) -> o_proj -> fused AR+norm,
   constructed from the production classes (same flags as serving),
   TP8, CUDA-graphed bench like the MoE layer bench.
2. Trace-align against serving (kernel names+grids+medians) BEFORE any
   optimization; the L2-warmth artifact applies to every memory-bound
   kernel — validate wins against serving traces, and remember serving
   prefill is EAGER.
3. Optimize per the idea list; every verdict (win or refutation) lands
   here with the mechanism.

## Idea #2 verdict (2026-08-23): KDA in_proj GEMM — at floor, REFUTED as a kernel item

Shape identified: in_proj_qkvgfab = [8, 7168] x [7168 -> 6288 local]
(3*qkv 4608 + g 1536 + f_a 128 + beta 12, padded 16) = 90 MB bf16
weights -> 13.9 us DRAM floor. L2-cold duel (92 rotating weight
replicas, the honest-serving condition): torch.matmul 18.8 us vs
serving's splitK[700] 19.8 us — the serving pick is within 1 us of the
generic heuristic, and L2-warm (13.7 us) sits ON the floor. There is no
better bf16 kernel; the GEMM is weight-bandwidth-bound.

The actual lever is PRECISION: KDA projections ship bf16 while the MoE
side is FP4. FP8 in_proj/out_proj would halve the dominant bytes
(~7 us on 75% of layers ≈ 5 us/layer avg; more with fp4). This is a
checkpoint/accuracy decision — flagged for Yikai, not actionable as a
pure runtime change.

## Idea #1 verdict (2026-08-23): o_proj zero-copy AR staging — decode-irrelevant

The attention AR's decode stage-in is a [8, 7168] copy = 114 KB ≈ 0.2 us;
the AR kernel's 7-17 us is skew-absorption, not staging. Zero-copy helps
at PREFILL payload sizes only (where the MoE side already uses it).
Deferred to the prefill workstream.

## Decode headroom budget @bs8 (corrected numbers, serving-honest)

b10 TPOT 12.2 ms / 93 layers = 131 us/layer wall. Components (serving
medians, uniform-corrected where dummy-skewed) vs their floors:

| component | now | floor | available | lever |
|---|---|---|---|---|
| experts (2 bmms) | 44.9 | 38-40 | ~5 | none practical (roofline; 85-89% eff) |
| KDA in_proj/out GEMMs | ~25 (0.75/lyr avg) | ~19 bf16 | ~6 | PRECISION ONLY (fp8/fp4 checkpoint) |
| MoE tail (2 ARs+finalize+addmm) | 30.6 wall | ~28 | 2-3 | exhausted (4 verdicts in MOE_FINALIZE_AR.md) |
| attention AR + norms | ~20 | skew-bound | ~3-5 | protocol-level only (persistent AR, spec slots) |
| front (dual_out+AG+routing) | ~30 | ~25 | ~5 | small fusions |
| launch gaps | ~20 | ~10 | ~10 | more graphs/PDL chains, diminishing |

CONCLUSION for Yikai: after this pass, the runtime-only decode headroom
at bs4-8 is ~15-20 us/layer of which most is hard (gaps, skew) — the
+14 us/layer needed for 10% E2E at C=4-8 is NOT reachable from kernel
work alone on the current architecture. The three levers that ARE big
enough: (1) KDA/attention weight quantization (fp8: ~5-6 us/layer,
checkpoint decision), (2) rank-skew/protocol work (persistent cross-layer
AR, speculative slots — high risk, ~5-8 us/layer), (3) architectural
(Mega-MoE-style token ownership for decode — measured a2a floor 144 us
says no at bs<=128). Decision needed on direction.

## Standalone KDA layer: BUILT + ALIGNED (2026-08-24 early)

`mla_layer/bench_kda_layer.py`: production KimiDeltaAttention with
duck-typed metadata/cache stubs (the whole cache-manager surface the
forward touches is 9 attributes), dummy weights, TP8 CUDA-graphed.
Decode timings: 47.7/51.5/56.2 us/layer @B1/4/8.

Alignment vs the serving trace (kernel, grid, median):
| kernel | standalone | serving | verdict |
|---|---|---|---|
| in_proj splitK[700] | 21.2 | 19.8 | aligned |
| _fused_kda_decode [8,12] | 12.9 | 12.4 | aligned |
| out GEMM TNT[112] | 5.9 | 6.0 | aligned |
| splitKreduce[197] | 3.6 | 4.1 | aligned |
| f_b/g GEMM TNN[96] | 3.5 | 3.8 | aligned |
| AllReduce | nccl symk 15.2 | trt ar_fusion (fused +res+norm) | KNOWN GAP: bench AUTO strategy; wire trt fused AR next |

Caveats carried from the MoE arc: single-layer L2 warmth flatters the
115 MB of weights (in_proj 90 + out 22 + f_b 3) — rotate weight sets
before trusting absolute wins; serving prefill is eager.

Optimization state: the GEMMs are at their bf16 floors (idea #2
verdict) — the layer's runtime levers are (a) trt fused AR wiring in
the bench for parity, then (b) the precision decision (fp8 KDA weights,
~5-6 us/layer avg) which this standalone can evaluate for accuracy
cheaply once Yikai approves trying it.

**Parity completed (2026-08-24):** with BENCH_ALLREDUCE_STRATEGY=ONESHOT
the standalone picks serving's trt ar_fusion AR (7.6 us, grid[32]) and
measures 45.2 us/layer @B8 vs serving's ~47 — full kernel-for-kernel
alignment. The KDA testbed is ready for the precision experiment
(fp8 in_proj/out_proj) the moment it is approved.

**L2-honesty check (2026-08-24):** --weight-sets rotation added (default
8). Rotating vs single-set: 45.68 vs 45.17 us — the KDA bench is already
L2-honest because 112 MB of weights exceed what the 126 MB L2 can retain
alongside activations; the MoE front's dual_out (41-56 MB) was the
flattered case. Rule of thumb recorded: single-layer benches flatter
kernels whose weights fit L2 with room to spare.

## Next-item EV assessment (for direction review)

The MLA-layer (full-attention) standalone needs the real KV-cache
machinery (largest harness build left) but targets only 26% of layers
whose kernel budget (~2x6.4 us fwd_prod + split-kv) offers ~3-5 us/layer
x 0.26 = ~1 us/layer expected value — the LOWEST-value remaining item.
Held until direction lands; the high-EV items (fp8 KDA weights, prefill/
Mega-MoE) both need Yikai's call.

## fp8 + custom-GEMV exploration (2026-08-24): measured, verdicts mixed

**fp8 what-if (stock torch._scaled_mm, L2-cold, M=8):** rowwise scales
are SLOWER than bf16 (23.9 vs 18.7 in_proj — cublasLt lacks small-M fp8
kernels); per-tensor scales: in_proj 14.3 us (-4.4 vs bf16, 24% win),
out_proj 14.1 (SLOWER than bf16's 11.1-12.4). Net off-the-shelf fp8-KDA
decode win: only ~2 us/layer avg — the earlier ~5-6 estimate needs
custom small-M fp8 kernels to materialize.

**Efficiency-gap correction:** my "at bf16 floor" verdict was wrong for
out_proj: torch 11.1 / serving ~9.6 vs a 3.4 us byte floor (30%
efficient); in_proj is at 74% (18.7 vs 13.9). Real kernel headroom
exists: ~5 us (in) + ~6 us (out) per KDA layer IF floor-rate kernels
existed.

**Custom small-M GEMV attempts (kernels/small_m_gemv.py), 3 iterations:**
smem-staged (bank-conflict pathology, 502 us) -> register-only with L2
x-reloads (126) -> row-tiled R=4 + occupancy sweep (42.8 in / 14.2 out,
saturated at 296 blocks, 2.1 TB/s). VERDICT: a naive SIMT GEMV cannot
beat cublasLt here — cublasLt's splitK runs 4.8 TB/s via TMA/cp.async
bulk transfers; matching or beating it (and capturing out_proj's 3x
inefficiency) requires a TMA-based kernel (CuTeDSL class work). Logged
as a real, quantified opportunity (~8-11 us/KDA-layer combined with
fp8) with a known engineering cost — not a quick win.

## kda_decode grid.z parallelism (2026-08-24): built, measured, REFUTED

Premise (from the floor audit): 12.4 us for ~12.6 MB of state traffic at
grid (8,12)=96 blocks looked occupancy-starved (~1 TB/s, 6x off floor).
Built the grid.z variant (kernels/fused_kda_decode_z.py: value tiles as
program_id(2), atomic last-tile-finalizes RMS, cached self-cleaning
scratch, bench monkeypatch via K3_KDA_Z=1). Correct (finite, layer runs)
but SLOWER: kernel 12.9 -> 19.7 us at grid [8,12,4]; layer 45.9 -> 53.2.

WHY (the honest model): the kernel is NOT bandwidth-bound. Its cost is
the per-(token,head) PREAMBLE — q/k 4-tap convs + SiLU, two 128-wide
L2-norm reductions, gate/beta/decay loads and exp/sigmoid — which every
tile block must redo (4x redundancy at tile=32), plus the cross-CTA
atomic finalize pass. The 12.6 MB byte-floor argument mispriced a
compute-bound kernel. Remaining honest headroom here: preamble
vectorization / transcendental fusion, ~2-3 us at best — deprioritized.

Bottom-line target list update: #2 (kda_decode) closes as refuted;
remaining approved targets: residue-fused tail AR port (+9.1pp, awaits
freeze lift) and the bf16 TMA small-M GEMM (+5-6pp, highest kernel
risk).

## TMA/Triton small-M GEMM (2026-08-24): REFUTED by the bandwidth control

Triton tl.dot sweep (36 configs, deep pipelining): best 24.8 us in_proj /
15.2 out_proj — LOSES to cublasLt (18.6 / 10.2). Decisive control: a pure
streaming read (torch.sum) of the same weights achieves only 3.68 TB/s
at 90 MB and 1.54 TB/s at 22 MB, L2-cold — DRAM bandwidth SCALES WITH
TRANSFER SIZE at these working sets (queue-depth/ramp bound), and
cublasLt's splitK (4.8 TB/s) already exceeds generic streaming. The
earlier "30-74% of floor" claims divided by an 6.5-7 TB/s peak that is
unachievable at single-kernel decode-size granularity. cublasLt is at
the practical limit; no custom kernel (SIMT, Triton, or TMA) has
headroom here. The only true levers left for these GEMMs are
architectural (batching streams across layers) — out of scope.

## Attention-side kernel exploration: CLOSED (all three targets resolved)
1. residue-fused tail AR: +9.1pp WIN (chain-bench-proven, exact math) —
   serving port awaits the trt-llm freeze lift.
2. kda_decode grid.z: refuted (compute/preamble-bound, not occupancy).
3. small-M GEMM floor capture: refuted (the floor was a measurement-
   model error; cublasLt is at the achievable bandwidth).
Rule extracted for the memory: BYTE-FLOOR ARGUMENTS MUST USE THE
SIZE-DEPENDENT ACHIEVABLE BANDWIDTH (measure a pure-read control at the
same working-set size), not the datasheet peak.
