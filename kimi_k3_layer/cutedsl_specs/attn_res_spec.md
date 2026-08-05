# Production kernel target: blackwell_fp16_attn_res  (MEMORY-BOUND, decode depth-mixer: residual add + block write + softmax depth-mix + RMSNorm)

**Kimi-K3 AttnRes decode step** (attention-residual depth mixer, one call per decoder layer), CuTeDSL, Blackwell B200 (sm_100a), 4.5.2. MEMORY/LATENCY-BOUND: at decode batches the whole working set is B*(N+2) rows of 7168 bf16 (~1.5 MB at B=8) — the production kernel takes ~7 us where the bytes cost <1 us; the win is one fused low-latency pass (the softmax over N+1 depth-candidates is per-row, tiny).

## Math
All row vectors have D=7168. Inputs (bf16 unless noted):
- `prefix: [B, D]` (UPDATED IN PLACE), `delta: [B, D]`, `blocks: [B, N, D]` (block bank, N=8; UPDATED IN PLACE when writing), `norm_w: [D]`, `proj_w: [D]` (norm_weight * proj_weight, may be given as two vectors), `out_norm_w: [D]`.
- scalars: `num_blocks` = n <= N (runtime int), `block_write_idx` = w (runtime int, -1 = no write), `eps`, `out_eps` (floats).

```
prefix += delta                       (in place, bf16)
if w >= 0: blocks[:, w] = prefix      (in place)
V = concat(blocks[:, :n], prefix[:, None])    # [B, n+1, D], fp32 math below
Vn = V * rsqrt(mean(V^2, -1) + eps)           # per-row RMS normalize
score[b, j] = sum_d Vn[b, j, d] * norm_w[d] * proj_w[d]
p = softmax(score, dim=-1)                    # over the n+1 candidates
o = sum_j p[b, j] * V[b, j]                   # fp32
out = (o * rsqrt(mean(o^2, -1) + out_eps)) * out_norm_w   -> bf16 [B, D]
```
- Target B=8, D=7168, N=8, n=5, w=-1. Also correct for B in 1..16, n in 0..8 (n=0 -> out = RMSNorm(prefix)), and w >= 0 (write path). fp32 accumulate everywhere.
- Consider: one CTA per token row-set; stage the n+1 rows once in SMEM/registers; the two reductions (per-candidate sumsq+score, then output sumsq) are row-local; a single kernel, no host sync, PDL-friendly.

## ===== EXPERIMENT INTEGRITY RULES (mandatory) =====
1. Build from scratch. Do NOT read/open/copy/import/cat/grep ANY existing kernel. OFF-LIMITS: `generated/`, `third_party/`, any `quack`, any other `generation/workspaces/` run. Use only this spec, the CuTeDSL API, and (if provided) the dictionary/recipes/template.
2. No cross-learning; compute honestly for arbitrary inputs; no benchmark special-casing. Follow `guide/CuTeDSL_Clean_Kernel_Guide.md`.
3. Honest measurement only.

## ===== REQUIRED run.py CLI CONTRACT (mandatory) =====
- `uv run run.py --check` — reference = a pure-torch fp32 implementation of the math above INSIDE run.py (clone prefix/blocks before, verify the in-place updates AND the output), on the target shape and (B=2, n=8, w=3); atol=2e-2 rtol=1e-2; print exactly `CORRECT: PASS` or `CORRECT: FAIL`.
- `uv run run.py --bench` — time kernel vs the torch reference on the target shape; print exactly `SPEEDUP: <float>` plus raw kernel ms and achieved `GB/s` (bytes = 2*B*D*(n+3) read + 2*B*D*(2 + (w>=0)) written, bf16).
Both exit 0 on success.

## ===== Canonical result line (mandatory) =====
Both `--check` and `--bench` must additionally print one final line:
`RESULT: {"correct": <bool>, "kernel_ms": <float>, "tflops": null, "gbps": <float>, "bound": "memory", "speedup": <float>}`
