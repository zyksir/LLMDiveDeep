# Production kernel target: blackwell_fp16_kda_chunk_fused  (MEMORY-BOUND, KDA chunked prefill with FUSED l2norm + safe-gate prologue)

**Kimi-K3 KDA chunk prefill** (delta-rule linear attention, TP8 shard), CuTeDSL, Blackwell B200 (sm_100a), 4.5.2. MEMORY-BOUND at H=12: the existing champion kernel (430 us at B=1 S=16k... kernel-only) leaves ~4 fp32 elementwise passes of PROLOGUE on the torch side (q/k l2norm, safe-gate activation, beta sigmoid, ~250-400 us at S=16k) - fusing them into the kernel's load path is the win. The chunk algorithm itself is the known INT21/FlashKDA schedule.

## Math
Inputs bf16 unless noted: `q,k,v: [B,T,H,128]`, `raw_g: [B,T,H,128]` (gate logits), `A_log: [H] fp32`, `dt_bias: [H*128] fp32` (per (h,d)), `beta_logit: [B,T,H]`, `S0: [B,H,128,128] fp32`.
Prologue (fp32 math, INSIDE the kernel on tiles as they load):
```
qn = q / max(||q||_2 per row-of-128, eps)   * (128 ** -0.5)   # l2norm along d + attention scale
kn = k / max(||k||_2 per row-of-128, eps)
g  = -5.0 * sigmoid(exp(A_log[h]) * (raw_g + dt_bias[h,d]))   # safe gate, per (h, d)
beta = sigmoid(beta_logit)
```
Then the standard KDA chunked delta-rule forward over chunks of C=64 (any correct chunking):
```
per token t (within the recurrence semantics):
  S <- diag(exp(g_t)) S            # decay per key channel
  S <- S + beta_t * kn_t (v_t - kn_t^T S)^T
  o_t = qn_t^T S                   # [128] -> output row
```
Outputs: `o: [B,T,H,128] bf16`, `S_T: [B,H,128,128] fp32`.
- Target B=1, T=16384, H=12. Also correct at B=1 T=4096 and B=2 T=8192. T % 64 == 0 may be assumed.

## ===== EXPERIMENT INTEGRITY RULES (mandatory) =====
1. Build from scratch. Do NOT read/open/copy/import/cat/grep ANY existing kernel. OFF-LIMITS: `generated/`, `third_party/`, any `quack`, any other `generation/workspaces/` run. Use only this spec, the CuTeDSL API, and (if provided) the dictionary/recipes/template.
2. No cross-learning; compute honestly for arbitrary inputs; no benchmark special-casing. Follow `guide/CuTeDSL_Clean_Kernel_Guide.md`.
3. Honest measurement only.

## ===== REQUIRED run.py CLI CONTRACT (mandatory) =====
- `uv run run.py --check` — reference = pure-torch fp32 sequential recurrence of the math above INSIDE run.py, on (B=1,T=512) and (B=2,T=256); cosine >= 0.999 AND atol=5e-2 rtol=5e-2 on o; print exactly `CORRECT: PASS` or `CORRECT: FAIL`.
- `uv run run.py --bench` — time the kernel on the target shape vs a torch composition (separate l2norm/gate passes + the same chunk math in torch, fp32); print exactly `SPEEDUP: <float>` plus raw kernel ms and achieved `GB/s` (bytes = 2*B*T*H*128*5 read + 2*B*T*H*128 written + 4*B*H*128*128*2).
Both exit 0 on success.

## ===== Canonical result line (mandatory) =====
Both `--check` and `--bench` must additionally print one final line:
`RESULT: {"correct": <bool>, "kernel_ms": <float>, "tflops": null, "gbps": <float>, "bound": "memory", "speedup": <float>}`
