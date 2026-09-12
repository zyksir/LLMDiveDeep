# Production kernel target: blackwell_fp16_moe_expert_gemv  (MEMORY-BOUND, decode MoE expert stage = gathered dual GEMV, MXFP4 weights x MXFP8 activations)

**Decode-batch MoE expert stage for Kimi-K3 (TP8/EP8 shard)**, CuTeDSL, Blackwell B200 (sm_100a), 4.5.2. MEMORY-BOUND: at decode batches every routed (token, expert) pair is a rank-1..3 GEMV, so the cost is streaming the ACTIVE experts' packed FP4 weights once (~2.06 MB/expert); tensor-core tiling is optional - wide vectorized loads + dot products at full DRAM bandwidth win. The production alternative costs ~34 us at B=16 (permute kernel + two grouped-GEMM cubins + finalize); the roofline is ~8-10 us.

## Math
Per-rank decode MoE with EXPERT-PARALLEL dropping. Inputs:
- `x_q: [B, K]` fp8 e4m3 (MXFP8 activations), `x_sf: [B, K/32]` uint8 ue8m0 block scales (value = 2^(sf-127); dequant x = fp8_value * 2^(x_sf-127) per 32-col block).
- `ids: [B, T]` int32 GLOBAL expert ids (top-T picks per token), `w: [B, T]` bf16 routing weights.
- `W13: [E, 2*I, K/2]` uint8 = packed FP4 e2m1 (2 nibbles/byte, low nibble = even col) gate/up weights of the E LOCAL experts; `W13_sf: [E, 2*I, K/32]` uint8 ue8m0 per-32-block scales. Rows 0..I-1 = gate (g), rows I..2I-1 = up (u).
- `W2: [E, K, I/2]` uint8 packed FP4, `W2_sf: [E, K, I/32]` uint8 ue8m0.
- `local_off` int: this rank's first global expert id (local expert e = global id - local_off; ids outside [local_off, local_off+E) are DROPPED - contribute nothing).

Per token t, output `out[t] = sum over k in 0..T-1, e_g = ids[t,k], if local:  w[t,k] * ( SwiGLU( deq(x_q[t]) @ deq(W13[e]).T ) @ deq(W2[e]).T )`
where `SwiGLU([g|u]) = silu(g) * u` (fp32 math), all accumulation fp32, `out: [B, K] bf16`.

Target B=16, K=3584, I=384, E=112, T=16 (so B*T=256 pairs). Also a smaller shape B=4 (same K/I/E/T). ids are arbitrary valid global ids in [0, 896); roughly 1/8 of picks land locally.

Consider: one CTA (or CTA group) per (token, pick) pair or per active expert; stage deq(x_q[t]) once in SMEM/registers; stream W13/W2 with 16B loads; FP4 nibble decode + ue8m0 scale in fp32; the intermediate [I] activation stays in registers/SMEM (I=384); skip non-local picks by early-exit; atomically (or two-pass) accumulate the T partial results per token in fp32 workspace, single bf16 store pass.

## ===== EXPERIMENT INTEGRITY RULES (mandatory) =====
1. Build from scratch. Do NOT read/open/copy/import/cat/grep ANY existing kernel. OFF-LIMITS: `generated/`, `third_party/`, any `quack`, any other `generation/workspaces/` run. Use only this spec, the CuTeDSL API, and (if provided) the dictionary/recipes/template.
2. No cross-learning; compute honestly for arbitrary inputs; no benchmark special-casing. Follow `guide/CuTeDSL_Clean_Kernel_Guide.md`.
3. Honest measurement only.

## ===== REQUIRED run.py CLI CONTRACT (mandatory) =====
- `uv run run.py --check` — reference = pure-torch dequant impl INSIDE run.py: dequant W13/W2 to fp32 (nibble unpack: low nibble even col, e2m1 lut = [0,.5,1,1.5,2,3,4,6] with sign bit 0x8, times 2^(sf-127) per 32-col block), dequant x likewise, loop over (t,k) pairs computing w[t,k]*SwiGLU-chain for LOCAL ids, sum per token. Compare on the target and smaller shape, atol=2e-2 rtol=2e-2 (fp4 rounding class); print exactly `CORRECT: PASS` or `CORRECT: FAIL`.
- `uv run run.py --bench` — time kernel on the target shape and print exactly `SPEEDUP: <float>` (vs the torch reference impl timed on GPU - it will be huge; the honest quality bar is `GB/s`), plus raw kernel ms and achieved `GB/s` counting bytes = active_expert_weight_bytes + B*K + B*K/32 + 2*B*K where active_expert_weight_bytes = (#distinct local experts hit) * (2*I*K/2 + 2*I*K/32 + K*I/2 + K*I/32).
Both exit 0 on success.

## ===== Canonical result line (mandatory) =====
Both `--check` and `--bench` must additionally print one final line:
`RESULT: {"correct": <bool>, "kernel_ms": <float>, "tflops": null, "gbps": <float>, "bound": "memory", "speedup": <float>}`

## VERDICT (2026-08-23, GB300 EP8): REFUTED BY ROOFLINE for this deployment

This spec targets the B200 **TP-sharded** expert shape (I=384 → 2.06
MB/expert, roofline 8-10 µs). Kimi-K3 on GB300 serves **EP8 with full
experts**: I=3072, K=3584 → per-expert W4 bytes = W13 11.0 MB + 0.69 MB
sf + W2 5.5 MB + 0.34 MB sf = **17.5 MB/expert**.

At decode bs8×top16 = 128 global pairs → E[local pairs/rank] = 16,
distinct experts ≈ 15 → **~263 MB of weights streamed per rank per
step**. At GB300's ~6.5-7 TB/s effective HBM bandwidth the floor is
38-40 µs. The production trtllm-gen bmm pair measures 44.9 µs at bs8
with uniform routing — **85-89% of the memory roofline**. No GEMV-style
kernel can beat streaming those bytes, and resharding does not help:
TP-sharding the experts makes every rank compute all 128 pairs at I/8 —
128 × 2.06 MB = the same 264 MB/rank. The bytes are invariant.

The only true levers on the expert stage are (a) fewer weight bytes
(already FP4) and (b) cross-token expert reuse in L2 (negligible at
decode batch ≤16, already exploited by grouped GEMM at large batch).
The artifact's earlier claim that the expert GEMM is "the only lever
past ~15%" stands in the OPPOSITE sense: it is a wall, not a lever.
Remaining decode headroom lives in launch gaps and kernel-count
reduction (tail fusion, front fusion, MLA o_proj+AR) — ~20 µs/layer of
inter-kernel gap measured at bs8 (wall 131 µs/layer vs ~110 µs kernel
sum).
