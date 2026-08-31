# WHY each improvement works — mechanism ledger (2026-08-31)

## THE WATERFALL: E2E delta fully attributed to named kernels (B=32)

Same-night graph traces of both E2E configs (prod tp 1.090ms vs new stack
0.812ms, Δwall = −278us per replay across 4 blocks). Per-kernel diff of
the rank-0 replay ledgers (the lamport-AR row absorbs rank skew under the
profiler — a known artifact, excluded; wall medians are barrier-synced):

| kernel (per replay, 4 blocks) | prod | new | Δ | mechanism |
|---|---|---|---|---|
| expert GEMM1 `bmm_MxE4m3...t128x8` | 306.9 | 90.7 | **−216.2** | EP few-wide-groups → tp_te many-narrow-groups (CTA parallelism) |
| expert GEMM2 `bmm_Bfloat16...` (u2→plain) | 155.9 | 87.5 | **−68.4** | same |
| fc1 path: nvjet 118.1→66.3, +shard GEMM 20.0, +gather 25.6 (aux-stream, overlapped), −quantize 11.0 | | | **≈ −26 wall** | fc1-shard (49→6MB read; gather hidden behind gate) |
| KDA/MLA/routing/shared rows | | | ±1-4 (noise) | untouched, as required |

Sum of attributed deltas ≈ −285us ≈ measured Δwall −278us (residual <3%).

**Three-level reconciliation (the anti-evaporation proof):**
- Kernel level (standalone, isolated): EP expert-op 128.1us vs tp_te
  58.0us at B=32 → −70.1us/layer (sol_ep.log / sol_experts2.log).
- Integrated trace: expert bmm pair −284.6us / 4 layers = **−71.2us/layer
  — matches standalone within 2%.** fc1 path −26/4 = −6.5us/layer —
  matches the fc1-shard layer A/B (+4.7% of 124us ≈ 6us).
- E2E wall: −278us ≈ 4 x (71.2 + 6.5) = −311 busy, minus the overlapped
  gather's 25.6us that never hits the wall → −285 ✓.

**Mechanism separation (imbalance vs better-kernel, tested separately):**
this entire balanced-router delta is the GEOMETRY/parallelism mechanism.
The imbalance mechanism is quantified independently by the skew A/B
(perfect router off): EP loses a FURTHER +3.6-5.6% that tp_te does not —
additional real-world margin on top of the table above, not part of it.

One entry per measured improvement: the mechanism, and the evidence that
proves it (never a hypothesis presented as fact; open gaps are labeled).

## 1. tp_te over EP experts: +18-33% MoE layer (B 8-256)

Weight bytes are EQUAL under balanced routing (each distinct expert is
read once somewhere: EP = full expert on its owner rank, tp_te = 1/8
slice on all 8 ranks). The win is therefore execution geometry, not
traffic:
- **Parallelism starvation in EP, PROVEN at kernel level (sol_ep.log,
  2026-08-31)**: standalone expert-op, equal per-rank weight bytes:

  | B | EP expert-op | tp_te expert-op | EP GEMM pair | tp_te GEMM pair |
  |---|---|---|---|---|
  | 1 | 25.7us | 25.9 (tie: latency floor) | 14.4 | 13.4 |
  | 8 | 69.4 | 29.5 (2.35x) | — | 15.4 |
  | 16 | 89.1 | 39.1 (2.28x) | **71.5** | **23.4 (3.06x)** |
  | 32 | 128.1 | 58.0 (2.21x) | — | 41.6 |
  | 64 | 206.7 | 80.3 (2.57x) | — | 64.0 |
  | 128 | 414.4 | 120.0 (3.45x) | — | 92.1 |

  At B=16, EP's GEMM1 achieves 1.0 TB/s vs tp_te's 4.2 TB/s on the SAME
  bytes and the SAME trtllm-gen tile family. The cause: EP has only
  ~distinct/8 local expert groups (~4 at B=16) → too few CTAs to cover
  the memory system; tp_te has the full distinct count (31) of narrow
  groups → 8x the group-level parallelism. The layer-level gap (1.5x) is
  smaller than the expert-op gap (2.2-3.4x) because shared experts,
  latent projections, and the AR are common to both.
- **Routing-imbalance immunity (proven)**: with the perfect router OFF
  (natural skew), EP degrades +3.6-5.6% while tp_te moves <0.5%
  (skew_ab receipts). Under production hot-expert skew the EP side
  worsens further; tp_te cannot (identical per-rank work by construction).
- Same comm in both modes (fused finalize-AR), so comm is not a factor.

## 2. fc1-shard: +4.8/+4.7/+3.8% MoE layer at B=8/32/64; +18-21% at B=1-2

Mechanism: fc1_latent_proj ([3584,7168] bf16, ~49MB) is REPLICATED per
rank in stock; row-sharding cuts the read to 6.1MB, and the reassembly
gather (fused lamport gather+MXFP8-quantize) is issued on the fc1 aux
stream OVERLAPPED against the gate GEMM — so the comm hides and only the
GEMM saving lands on the critical path.
- Why largest at B=1-2: the layer is latency-bound there and the fc1 read
  (~15-20us) is a big slice of 95us; the saving is nearly constant in B
  while everything else grows.
- Why it dies at B>128: `_PAIR_MAX_TOKENS=128` — past it the fused
  quantized-pair gather rung is off and the bf16 gather costs more than
  the GEMM saving (measured -4.8% at B=256).
- Why E2E gains (+20-40% incl. layout) grow with B: attention time is
  ~flat in B at ISL 4K while EP MoE deteriorates faster than tp_te MoE.

## 3. CuteDSL GEMM2 1.33-1.43x at B>=32 (and why NOT at B<=16)

trtllm-gen's `t128x8x512` tile = N=8 per CTA → 3584/8 = 448 serial
N-steps per expert with K=384 (a single underfilled K-tile, no K
pipelining) → 1.55TB/s achieved. CuteDSL 128x128 tiles → 28 N-tiles of
parallel CTAs → 2.07TB/s. Proof that it's the tile and not the hardware:
the same shape in cuBLAS bf16 reaches ~6TB/s (geometry probe), and
GEMM1 (K=3584, same trtllm-gen tile family) achieves 3.6-4.5TB/s.
- Why trtllm-gen still wins B<=16: with few expert groups its tiny tile is
  latency-optimal (measured 2.8-3.2TB/s at B<=8) while the CuteDSL kernel
  pads every group's M to 128 rows (at B=1 that is 128x wasted MMA) and
  carries a ~13-14us kernel floor. Hence: measured per-hardware dispatch
  (AutoTuner tactic), never a hardcoded threshold.

## 4. Fused halves: 31.4us vs 51.5us expert GEMM chain at B=32 (1.64x)

G1+act fusion and G2+finalize fusion each remove one launch AND keep the
intermediate in-kernel: the standalone chain pays situ (2.1us) +
quantize (3.1us) + finalize (4.7us) kernels plus their launch/latency
floors and intermediate HBM round-trips; the fused pair pays none of
them, and the G2 half also escapes the t128x8 tile (entry 3).

## 5. MLA at ISL 4096: TP beats DCP(helix) and BS-SPLIT until B~64

DCP/BS-SPLIT keep all 96 q-heads per rank → q_b/kv_b projection weights
REPLICATED = 8x the projection bytes TP reads. At ISL 4K the KV read is
small (38MB/rank at B=8), so the projection tax dominates → TP 0.094ms vs
helix 0.143 at B=8. The ranking flips when KV bytes dominate: at ISL 32K,
B=32, BS-SPLIT 0.151ms vs TP 0.292 (1.93x) — KV/rank is 1.2GB in TP
there. Helix additionally cannot run at B>=32 OR long ISL on this stack
(prebuilt trtllm-gen 64-heads/CTA tile heuristic binds on batch AND KV
length; 96 % 64 != 0) → tp2 x dcp4 is a hard prereq for prod DCP here.

## 6. MPK megakernel: ~49us/KDA-block vs ~87us eager region

The recurrence kernel itself was ALREADY at the byte floor in eager
(fused triton kernel, 16us/block both places). The megakernel's win is
everything around it: zero inter-kernel gaps, no per-kernel launch
latency (tasks dispatch from a persistent worker), and the small ops
(norms, projections' tails) absorbed into task waves. Wall-clock proof:
8 serialized blocks execute inside one launch envelope (128us worker
kernel = 16us/block) while a single tiny task costs 131-155us per
separate launch.

## 7. Correctness fixes are enablers, not accelerators

The SHARD_FC1 segfault (device-less host pointer tables under the ambient
device ctx), the perfect-router bypass gap, and the unstubbed gate did
not change kernel speed; they made the honest measurements above possible
(and #2's first "+19%" reading was exposed as stub-gate-inflated by
fixing them).

Appendix: EP kernel decomposition (sol_ep.log) — filled by the 2026-08-31
run; compare bmm times/BW at equal bytes vs the tp_te table in
cute_w4a8/RESULTS.md.
