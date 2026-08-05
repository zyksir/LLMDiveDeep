# Production kernel target: blackwell_fp16_moe_tail_gemm  (MEMORY-BOUND, tall-skinny bf16 GEMM + bias-add)

**Decode-batch MoE tail projection `D = C + A @ W^T`** (Kimi-K3 fc2 latent->hidden, TP8 decode), CuTeDSL, Blackwell B200 (sm_100a), 4.5.2. Memory-bound: M is tiny (a decode batch) so the 51 MB weight read dominates; cuBLAS only reaches ~4.6 TB/s on this shape (11.2 us) vs the ~8 TB/s HBM roofline (~6.5 us) - the kernel is a weight-streaming exercise, not an MMA exercise.

## Math
`A: [M, K]` bf16 (row-major), `W: [N, K]` bf16 (row-major; used transposed), `C: [M, N]` bf16 (row-major), fp32 accumulate.
```
D = C + A @ W.T            D: [M, N] bf16
```
- Target M=16, K=3584, N=7168. Also a smaller shape M=4, K=3584, N=7168 (both must be correct; M in 1..64 may be assumed small).
- Consider: with M<=16 the whole A fits in registers/SMEM of every CTA; tile over N (and split K only if it helps); the win is streaming W at full DRAM bandwidth with wide loads while C/D traffic is negligible.

## ===== EXPERIMENT INTEGRITY RULES (mandatory) =====
1. Build from scratch. Do NOT read/open/copy/import/cat/grep ANY existing kernel. OFF-LIMITS: `generated/`, `third_party/`, any `quack`, any other `generation/workspaces/` run. Use only this spec, the CuTeDSL API, and (if provided) the dictionary/recipes/template.
2. No cross-learning; compute honestly for arbitrary inputs; no benchmark special-casing. Follow `guide/CuTeDSL_Clean_Kernel_Guide.md`.
3. Honest measurement only.

## ===== REQUIRED run.py CLI CONTRACT (mandatory) =====
- `uv run run.py --check` — vs `torch.addmm(C, A, W.t())` on the target and the smaller shape, atol=2e-3 rtol=1e-2; print exactly `CORRECT: PASS` or `CORRECT: FAIL`.
- `uv run run.py --bench` — time kernel + torch baseline (`torch.addmm(C, A, W.t())`) on the target shape; print exactly `SPEEDUP: <float>` plus raw kernel ms and achieved `GB/s` (count bytes = 2*(M*K + N*K + 2*M*N), the N*K weight term dominating).
Both exit 0 on success.

## ===== Canonical result line (mandatory) =====
Both `--check` and `--bench` must additionally print one final line:
`RESULT: {"correct": <bool>, "kernel_ms": <float>, "tflops": null, "gbps": <float>, "bound": "memory", "speedup": <float>}`
