# The fused MoE finalize+AR kernel: why it wins, and how to beat it

Topic owner: Yikai. This doc collects everything measured on GB300/TP8
(2026-08-22) about production's decode-tail kernel so a better one can be
written. Bench: `kernel_benchmarks/bench_moe_finalize_ar.py`. Traces:
`/node-storage/var/traces/` (see MANIFEST.md).

## What production runs (the kernel to beat)

`tensorrt_llm::ar_fusion::moe::moefinalize_allreduce_fusion_kernel_oneshot_lamport`
(fork commit f17ea3ab32, exposed as `MoEAllReduce.finalize_allreduce_rmsnorm_concat`).
ONE kernel that does, per token:

1. **finalize** — gather this rank's top-k expert outputs from the permuted
   fc2 buffer, weighted-sum with the routing scales (in registers);
2. **one-shot lamport AR** — write the local partial of
   `concat([routed_latent 3584 | shared 7168])` to every peer's symm buffer,
   spin on lamport flags, reduce locally;
3. **rmsnorm** on the latent half, epilogue-fused;
4. emit two outputs: normed routed latent + reduced shared.

Inputs: unfinalized `fc2_output [n_permuted, 3584]` (from `run_moe`
with `do_finalize=False`), `expanded_idx [B, top_k]` INT32, `scales [B, top_k]`
(same dtype as input), shared `[B, 7168]`. Cap: 128 tokens
(`MoEAllReduce.max_token`), TP ∈ {2,4,8,16}.

## Why it beats our previous tails (mechanism)

Our packed tail (finalize kernel → cat copy → oneshot AR [B, 10752] → split →
rmsnorm) loses on four counts:

1. **HBM round-trips**: we materialize the finalized `[B, 3584]`, then the
   `[B, 10752]` cat, then the AR output — three extra full passes. The fused
   kernel keeps the finalize accumulator in registers and writes it straight
   into the AR's symm staging.
2. **Launches & sync points**: 4 kernels + their ordering edges vs 1.
3. **Latency hiding**: the lamport spin-wait (inter-rank arrival skew)
   overlaps the finalize math instead of following it. Rank-skew evidence:
   on dummy-weight serving traces, heavy-bmm ranks show 12.8 µs finalize+AR,
   light ranks 19.5 µs — bmm+tail sums to ~27.5 µs on EVERY rank.
4. **The norm is free** — fused into the AR epilogue instead of a separate
   RMSNorm launch.

Measured (B=8, TP8, uniform routing): fused 18.1 µs standalone; our packed
tail ≈ 47 µs kernel-level from the layer trace (finalize 6.0 + AR 35.4 +
norm ~3 + cat ~2), multimem-tail comm ≈ 67 µs standalone at these sizes.
Sweep (exact-size permuted input, `bench_moe_finalize_ar.py`):

| B | fused | notes |
|---|---|---|
| 1–16 | ~18 µs | flat — latency-bound |
| 32 | 20.3 | |
| 64 | 28.2 | |
| 128 | 49.9 | payload-bound growth begins |

## Weak-spot hypotheses: one refuted, what remains

**REFUTED (2026-08-22): the padded-gather hypothesis.** The bench's
`fused_padded` variant hands the kernel a tile-padded ~896-row buffer with
fully scattered indices — penalty measured **1.00×** at every batch (18.0 vs
18.3 µs at B=8). The in-layer readings of 38.7 µs (and the co-inflated
fc2_latent_proj at 43.9 µs) are a **graph-replay profiler attribution
artifact**, not real cost: two same-stream kernels cannot overlap, yet the
trace shows them overlapping. Treat in-graph per-kernel durations with
suspicion; standalone duels are the ground truth.

**True cost profile** (standalone, TP8): flat **~18 µs for B ≤ 16**
(latency-floor regime — the one-shot lamport round trip dominates, payload
irrelevant), 21.8 @32, 27.8 @64, 49.4 @128 (payload-bound). For reference
our packed tail (torch-reference finalize + Collectives AR + norm) is
95–136 µs and the multimem tail comm ~67 µs at these sizes.

## Candidate directions (to be designed/owned by Yikai)

The kernel is near the one-shot latency floor at B ≤ 32 — beating it there
means beating the FLOOR, not the implementation:

1. **Multimem variant (B ≥ 32 first)**: one-shot lamport writes P−1 remote
   copies (7× egress at TP8); NVLink5 `multimem.st.reduce` is 1× egress with
   switch-side reduction. The fused kernel's payload-bound growth
   (28 → 49 µs at B 64 → 128) is where this wins first; fusing finalize+norm
   into our `b10_multimem` AR is the missing piece.
2. **Latent-only fused finalize+AR (the primary target)**: the stock kernel
   ARs concat([latent|hidden]) = 10752 columns. What the sharded-fc2 tail
   needs is finalize fused into an AR of the LATENT ONLY (3584); its second
   AR already carries shared+fc2 partials together. A composable hybrid
   (stock fused op + sharded fc2 + extra output AR,
   `DecodeTail.FUSED_FINALIZE_AR_SHARDED_FC2`) was measured 2026-08-22 and
   REFUTED: +6.0% @bs8 → −0.6% @bs64 vs deploy's +8.4/+7.0 — it double-pays
   AR payload. Only the latent-only kernel captures the ~5–7 µs/layer
   finalize gap; nothing composable today reaches it.
3. **bmm-epilogue fusion**: fc2 bmm epilogue scatters weighted partials
   straight into AR symm staging — removes the finalize read entirely;
   the AR kernel only spins+reduces+norms. Helps most where payload is
   already minimal (combines with 2).
4. **Latency floor itself**: at B ≤ 16 the 18 µs is round-trip-bound; only a
   protocol change (speculative lamport slots, persistent AR kernel across
   layers) attacks it — highest risk, highest ceiling.

## Placement rule

All communication kernels and their benches live under `communication/`
(`kernels/`, `kernel_benchmarks/`); layer code only *selects* them. The
fused-tail integration points: `kimi_k3_layer/b10_kimi_k3_moe_layer.py`
(`DecodeTail.FUSED_FINALIZE_AR`, `stock_forward`) — both call through
`MoEAllReduce` today and should switch to the winning kernel here when it
lands.

## v2 status (2026-08-23): data-encoded lamport — gap closed to ~2 µs

v2 adopts the stock kernel's own protocol (all in
`kernels/finalize_ar_norm.py`): the payload IS the arrival signal
(buffers pre-filled with bf16 −0.0 = 0x8000; writers canonicalize
−0.0→+0.0; readers spin per-16B vector until no lane is the sentinel),
3 rotating slots with the consumed slot cleared in-kernel, spin
distributed across all threads, round counter bumped by the last block.
Deleted: flag words, both `__threadfence_system`, the tid-0 serialized
spin, and the trailing bump kernel — the tail is ONE launch,
CUDA-graph-safe.

Standalone duel (`fan_duel_v2.log`, vs v1 in parens):

| B | fused (stock) | ours v2 | ours/fused |
|---|---|---|---|
| 1 | 19.3 | 18.6 (23.1) | **0.96×** |
| 4 | 23.4 | 21.6 | **0.92×** |
| 8 | 18.7 | 20.2 (29.3) | 1.08× |
| 16 | 18.5 | 21.2 | 1.15× |
| 32 | 20.5 | 21.7 (31) | 1.05× |
| 64 | 27.4 | 23.8 | **0.87×** |
| 128 | 49.2 | 28.5 (37.7) | **0.58×** |

The B=4–32 gap fell from ~11 µs to 1.5–2.7 µs; we now win at B=1, 4,
64, 128. In-layer (stockplus `K3_FRONT_STAGES=fused_gate,radix,
fan_tail`, deploy timing): +1.6/+3.0/+2.4/+2.0% @bs8/16/32/64 vs pure
stock, i.e. ~+0.3–0.6 pp over the front-only ladder — up from v1's −3 pp
but only break-even-plus against the stock tail, because the stock tail
hides its full fc2 prefetch under the AR spin via PDL while our
follower (torch-launched addmm + AR) cannot accept the PDL baton
(confirmed in the 50-iter trace: nothing overlaps the fan kernel's spin
window). Remaining lever: a PDL-launched follower GEMM (trtllm-gen
sharded-fc2 with usePdl, or fusing the fc2-shard epilogue into the fan
kernel itself). Traces: `debug-decode-bs8-{stock,fantail}-50iter` in
/node-storage/var/traces (opt = 3 true lanes; in-graph spin-kernel
durations carry the usual PDL attribution inflation).

## PDL follower GEMM: built, measured, REFUTED (2026-08-23)

Hypothesis: the fan tail loses stock's spin-overlap because the
follower (sharded-fc2 addmm) is torch-launched and can't take the PDL
baton; a custom PDL GEMV that prefetches its weight tile into smem
BEFORE `griddepcontrol.wait` would reclaim it. Built as
`fc2_shard_pdl` in `kernels/finalize_ar_norm.py` (correct to bf16
noise, token-chunked smem, PDL launch).

Measured (tail-vs-tail duel arms in `bench_moe_finalize_ar.py`;
stock_tail = fused op + full fc2 matmul; ours = fan + follower + one
hidden AR):

| B | stock_tail | ours (pdl gemv) | ours (torch addmm) |
|---|---|---|---|
| 1 | 34.6 | 34.6 | 55.7 |
| 8 | 35.0 | 44.4 | 50.8 |
| 32 | ~38 | 120.8 | 57.4 |
| 128 | 79.7 | 313.9 | 61.7 |

REFUTED for two structural reasons: (1) the naive GEMV cannot compete
with nvjet as B grows (compute-bound, no tensor cores); (2) the prize
is tiny by construction — OUR sharded fc2 reads 6.4 MB (~1 µs of
hideable bandwidth, 3.9 µs kernel) whereas stock's overlap hides a
51 MB full-fc2 prefetch (~15 µs) under its spin. Our tail already
removed the thing PDL-overlap is good at hiding. The eager duel also
overstates our composite cost (3 host launches, no producer GEMM to
absorb them); in-graph in-layer the fan tail nets +1.6–3.0%.

What remains on the table for the tail: fuse the sharded-fc2 partial
straight into a second lamport AR round inside one kernel (saves one
launch + part of the ~13.5 µs hidden-AR floor), or attack the B≤16
one-shot latency floor itself. Bench-harness war story: the duel's
`norm_w` was unseeded (rank-divergent) — harmless for every
rank-local arm, but any SHARDED-fc2 arm mixes rank-normed latents
across ranks and "mismatches" at rel≈0.2; norm weights are model
weights and must be rank-identical in benches.

## v1 implementation status (2026-08-22, `kernels/finalize_ar_norm.py`)

The latent-only fused finalize+AR+rmsnorm kernel EXISTS and is correct
(err ≈ bf16 noise at every size incl. padded-row guards), PDL-launched
(`cudaLaunchKernelEx` + griddepcontrol wait / early launch_dependents).
Standalone duel (`bench_moe_finalize_ar.py`, `ours` column): 23.1 µs @B=1
(beats stock 0.96×), 29.3 @B=8, 31 @B=32, 37.7 @B=128 (0.77× — wins in the
payload-bound region). Beats our packed tail ~3× and multimem ~2×
everywhere. In-layer (`DecodeTail.FUSED_FAN_SHARDED_FC2`): −2% vs stock,
i.e. NOT yet competitive with the deployed sharded tail (+16%).

Remaining gaps, measured: (1) ~11 µs at B=4–32 vs stock's kernel — its
persistent tuned CTA shape and data-encoded lamport (no flag words, no
bump kernel) vs our per-token flags + trailing bump launch; (2) the PDL
baton dies at the follower — torch-launched addmm can't accept the early
trigger, so the spin-overlap that makes stock's tail great is not
realized. Both are kernel-engineering items squarely in this doc's scope.
Debug war stories worth keeping: half-warp `__shfl_down_sync` with a full
mask deadlocks; `griddepcontrol` in non-PDL launches must be predicated;
run_moe's expanded_idx carries invalid padded rows that must be bounds-
checked; torch load_inline under mpirun leaves stale build batons when
killed.

## v3 "mega-tail" (2026-08-23 late): built, measured in-layer, REFUTED

One kernel = finalize + latent lamport AR + rmsnorm + fc2-shard GEMV
(smem W tile, token-parallel) + hidden lamport AR (column-tiled round 2,
grid 112 = all-resident, data-encoded protocol, one launch + a tiny
scratch-clear). Correct at every size (`kernels/mega_tail.py`,
probe_mega). Standalone 40.8 µs @B8; in-layer (stockplus
fused_gate,radix,mega_tail): **−6.1% @bs4, +0.1% @bs8, −4.9% @bs16 vs
pure stock** — worse than fan_tail (+1.6..3.0%) and the deploy tail.

WHY it loses (the explainable version): the serving tail's cost is
NOT launch overhead — it is two AR latencies, and trt's tuned one-shot
lamport ARs cost ~7 µs each at [8,3584]/[8,7168] while our data-encoded
rounds cost ~10-12 µs each under the same rank skew. Fusing everything
into one kernel saves ~8 µs of launches/gaps but pays ~8-10 µs of
slower rounds AND serializes the whole chain inside one grid (no
inter-kernel pipelining/PDL prefetch for the follower). Monolithic
fusion is only a win when the fused rounds match the split kernels'
round latency — which our protocol does not yet at small payloads.

Standing conclusion for the MoE decode tail after v1/v2/PDL-follower/v3:
trt's split one-shot ARs + CUDA graphs are near-optimal at bs4-16; the
tail's remaining headroom is ~2-3 µs of gaps, not 8-10. The transferable
lesson for the MLA workstream: do NOT replace trt ARs with custom
rounds; fuse the cheap neighbors INTO the existing trt fused variants
(e.g. allreduce_norm) instead.

## Cross-node alignment with TRT's AllReduce (2026-08-24)

Why TRT's oneshot-lamport survives GB300 cross-node while our custom
kernels did not: TRT's workspace uses LEGACY cudaIpc handles
(_ipc_utils.IpcMemory) which the NVL72 IMEX daemon extends across the
NVLink domain, gated by tp_size <= gpus_per_node (production launchers
set gpus_per_node to the DOMAIN size); its true multi-node path is
MnnvlMemory (cuMemCreate FABRIC + IMEX channels), tried first under
AUTO with graceful downgrade to IPC-lamport then NCCL. Our kernels used
torch symm_mem (intra-node in most builds).

ALIGNED: communication/kernels/peer_alloc.py reuses TRT's IpcMemory
with TRT's gates (K3_PEER_ALLOC=ipc|symm|auto; K3_GPUS_PER_NODE = the
NVLink-domain size on fabrics). fan v2 switched and validated — duel
numbers identical on the ipc allocator, zero mismatches. mega_tail /
persistent_ar / col_quant can switch the same way when needed; the
kernels only consume the pointer table.
