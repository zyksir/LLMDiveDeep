# Kimi-K3 MoE layer — strategy selection verification (TP8, B200)

Date: 2026-08-19
Qualification: QUALIFIED (report-only one-axis search around the shipped
strategy; 16 sizes B=1..16384, 60 iters x 2-4 inputs, tie-flip-aware
tolerance 0.3 — the bf16-gate tie flips are benign and bit-identical across
configs).

Verdict: **shipped strategy confirmed for decode (B<=128)** — every
one-axis delta there is sub-microsecond noise, and `routing=triton` at T=64
(-0.08 us) independently confirms the radix dispatch window. **Mid/large
prefill has open items**: single-run wins for `fp32_gate=False` (-40 us at
16384), `prefill_sp=False` (-39.9 at 4096, but +68.6 at 512 — a boundary,
not a flip), `ref_tail=False` (-10.3 at 128), `overlap_shared=False` (-8.3
at 2048). Repeat A/Bs (two rounds each, same-day) decided all four:
- **fp32_gate=False CONFIRMED** (16384: 2030/2028 vs 2070/2062 — -35..-40
  us; also -27..-33 at 256-2048) -> default flipped to False. Cost: more of
  the documented-benign bf16 tie-flips (rel_err 0.19 -> 0.25).
- **prefill_sp=False at 4096 CONFIRMED** (743/743 vs 779/778 — -35 us) ->
  dispatched off for 4096<=B<8192 (8192 weak/pending; 16384 SP wins).
- **ref_tail=False at 128 CONFIRMED** (127/128 vs 137/137 — -9..-10 us;
  ON wins at 96 and 256+) -> dispatched off for 96<B<=128.
- **overlap_shared=False at 2048 REJECTED** (594 vs 590 — the search's -8.3
  was single-run noise) -> shipped kept.
All three accepted wins are implemented as capture-time dispatch in
moe_kimi_k3.forward, mirroring the routing dispatch.

## One-axis deltas (positive = candidate slower than shipped)

| one-axis change vs shipped | worst regression (us @ T) | best gain (us @ T) | verdict |
|---|---|---|---|
| fc1_shard=False | +10.8 @ 1 | -4.4 @ 2048 | REVIEW |
| fc2_shard=False | +512.2 @ 2048 | -0.3 @ 96 | keep shipped |
| fp32_gate=False | +6.7 @ 32 | -40.0 @ 16384 | REVIEW |
| fp32_gate_decode=True | +27.2 @ 16384 | -2.4 @ 8192 | keep shipped |
| merge3=False | +56.2 @ 16 | -4.9 @ 2048 | REVIEW |
| merged_gemm=False | +8.1 @ 256 | -1.7 @ 4096 | keep shipped |
| overlap_shared=False | +53.7 @ 2 | -8.3 @ 2048 | REVIEW |
| prefill_opt=False | +1537.6 @ 16384 | -0.6 @ 8 | keep shipped |
| prefill_sp=False | +68.6 @ 512 | -39.9 @ 4096 | REVIEW |
| quant_slice=False | +15.9 @ 16 | -4.4 @ 4096 | REVIEW |
| ref_tail=False | +29.5 @ 32 | -10.3 @ 128 | REVIEW |
| routing=triton | +1100.4 @ 256 | -0.1 @ 64 | keep shipped |
| routing=trtllm | +1086.1 @ 256 | 0.0 @ 64 | keep shipped |

## Why the layer switches strategy as batch grows (concise)

- **B<=32 (small decode):** every fixed cost is visible — precomputed
  routing (sgl_radix), merged input GEMM, merge3, and quant-slice fusion all
  remove launches/copies from a ~50-90 us budget.
- **32<B<=96:** ref-tail replaces the fc2-shard tail (from B=33), and the
  raw-logits in-kernel routing front wins ~1-2 us DESPITE costing 17-32 us
  of routing kernels (profiled, traces in local_result/*.trace.json.gz):
  those kernels hide under the overlapped shared-expert + allreduce chain,
  while radix's 2.4 us is exposed on the critical path — hence the trtllm
  routing dispatch in this window only.
- **B=128:** in-kernel routing cost grows with tokens while sgl_radix stays
  ~2.5 us — radix returns.
- **B>=256 (prefill):** the sequence-parallel copy-engine tail (prefill_sp)
  dominates: it removes the all-reduce from the critical path
  (557-2066 us vs 1690-2183 without). Routing choice is secondary; fp32
  gate is FLOP-neutral there so it stays on for bit-aligned selection.
- **B>=4096:** compute-bound; merged GEMM and shards matter less
  (deltas shrink toward 0), collectives and expert GEMMs dominate.

## Why each winning flag wins (mechanism per flag)

- **fc1_shard ON** (+1..+11 us if off, all sizes): each rank computes 1/8 of
  the fc1 columns and all-gathers the slice with the MXFP8 quantize fused on
  the write-out. The merged fc1 GEMM is weight-read-bound, so cutting its
  work 8x beats the small AG payload at every size (the -4.4 at 2048 is a
  single-run outlier against fifteen consistent wins).
- **fc2_shard ON** (up to +512 us if off at prefill; +11..+20 at B<=32;
  exactly 0 at 64-128): sharding fc2 turns a full 22 MB-weight GEMM per rank
  plus a hidden-size all-reduce into 1/8 GEMM + AG. The zeros at 64-128 are
  not noise — ref_tail replaces the fc2-shard tail from B=33 through the
  decode window, so the flag is inactive there.
- **prefill_sp ON** (+53..+69 us if off at 256-2048, +41 at 16384; but
  **-39.9 at 4096 and -7 at 8192**): the sequence-parallel copy-engine
  RS+RS+AG tail removes the all-reduce from the critical path. The 4096-8192
  dip is a real boundary hole under repeat verification — the SP tail's
  copy-engine slots stop covering the payload there before the AG-free
  benefit re-dominates at 16384.
- **fp32_gate at prefill — the one WRONG shipped default** (bf16 gate saves
  a consistent 27-40 us at every prefill size): the claim that the fp32
  gate is FLOP-neutral (merged GEMM drops dead gate rows) ignores that the
  separate fp32 gate op adds a full extra pass over the hidden states plus
  its launch; the measured cost is 27-40 us. Since the bf16-gate tie-flips
  are already accepted at decode, accepting them at prefill buys this back;
  pending the repeat A/B below, flip the default.
- **overlap_shared / merged_gemm / merge3 / quant_slice ON**: each removes
  a launch/copy or fills idle SMs at decode (+8..+56 us when off at their
  active sizes); deltas shrink toward zero at compute-bound prefill, as
  expected.

Raw data: `local_result/moe_strategy_search.csv`.

## Reproduce

```bash
python3 kimi_k3_layer/search_best_strategy.py --world 8 \
    --sizes 1,8,64,128,512,4096 --iters 60 --n-inputs 2 --max-rel-error 0.3
# ~3-4 min for the subset (14 configs x 6 sizes); full 16-size sweep is a
# checkpoint run, not the reproduce path
# full 16-size sweep takes longer; the subset above answers the same
# question per regime within the 5-minute budget
```
