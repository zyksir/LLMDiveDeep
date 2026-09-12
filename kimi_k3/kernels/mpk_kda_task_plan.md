# MPK mega-KDA task plan (sm_103)

Date: 2026-08-30. Status: substrate VALIDATED on this node; design note
before any CUDA (incremental-dev rule). Companion: mirage_megakernel_results.md
(B200 study), k3-production-moe-bench memory (why attention is the target).

## Substrate status on this node (executed today)

- mirage rebuilt from fresh clone at `/node-storage/mirage`, venv
  `/node-storage/var/mirage-venv` (persistent paths — the old /workspace
  container is gone). z3 4.13 built from submodule with
  `-DZ3_BUILD_PYTHON_BINDINGS=ON`, python package + libz3 copied into venv
  site-packages, `LD_LIBRARY_PATH=/node-storage/mirage/deps/z3/build`.
- **sm_103 patches to our clone** (upstream gates are `target_cc == 100`):
  `python/mirage/mpk/persistent_kernel.py`, `python/mirage/utils.py`,
  `python/mirage/kernel.py` — accept `(100, 103)` and parameterize
  `-gencode=arch=compute_{cc}a,code=sm_{cc}a`; test setup.py files need the
  same 100a→103a swap per directory.
- **PASS**: persistent worker+scheduler kernel (205 KB smem) +
  `glm_moe_router` testmode correctness on sm_103.

## Why KDA attention is the megakernel target (from today's analysis)

MoE is solved for B<=256 decode (tp_te at ~1.0-1.2x byte floor from B>=16).
The 4-block attention region at B=32 spends ~0.35 ms/replay across ~85
kernels whose 5-50MB reads run at 0.5-3.5 TB/s (size-dependent achievable —
measured control). Fused per-block granularity (110MB @ 4.4 TB/s) bounds
attention at ~0.10-0.15 ms → **~0.2-0.25 ms/replay prize at B=32**, larger
relatively at B<=8. KDA is 3 of 4 blocks.

## Ingredient map (mirage HEAD, blackwell tasks)

| KDA stage | task status |
|---|---|
| in_proj / out_proj GEMMs | EXISTS (linear task family; bf16 path) |
| depthwise short conv k=4, fp32 state in-place | **EXISTS**: `inkling_sconv_sm100.cuh` (semantics match: x[SEQ,HIDDEN] bf16, taps [HIDDEN,K] fp32, state [K-1,HIDDEN] fp32 in-place) |
| gated delta-rule recurrence (fp32 state [H,d,d], decay via A_log/dt_bias, beta gate) | **MISSING** — the ~250-line task to write |
| gated RMSNorm epilogue (norm_before_gate, sigmoid) | MISSING (small; can fold into recurrence task epilogue) |
| router/MoE | out of scope (tp_te already at byte floor) |

## KDA recurrence task spec (draft)

Reference math: fork `tensorrt_llm/_torch/modules/mamba/fused_kda_decode.py`
(triton `_fused_kda_decode_kernel`, measured 48us/3 blocks/replay at B=32 in
the block4 trace; b10_kda layer 36us@B=1 vs 24-25us bound from the study).

- Signature (decode, one token/seq): inputs q,k,v,g [B, H_t, D] bf16
  (post-conv, post in_proj split), beta [B,H_t], A_log/dt_bias [H_t] fp32;
  state [B, H_t, D, D] fp32 in HBM updated in place; out [B, H_t, D] bf16.
- Tile plan: one task per (batch, head-group); D=128 → state tile 128x128
  fp32 = 64KB — fits smem; load state → rank-1 delta update → q·state → gated
  rmsnorm epilogue → store state + out. Memory-bound by state (64KB/head);
  B=32 x 12 heads/rank = 24MB state traffic → ~12us at size-dependent BW.
- Registration touchpoints (3, per study): task enum + task_header.cuh
  include + python-side task registration in mpk persistent_kernel.py.
- Test plan: standalone task test dir modeled on sm100_moe tests
  (build_ext 103a); correctness vs the triton fused_kda_decode on random
  states/inputs (fp32 tolerance); then a 3-task chain test
  (sconv -> recurrence -> linear) vs the fork's KDA decode path timing.

## Execution log (2026-08-30, autonomous stretch)

**Task BUILT, REGISTERED, and VALIDATED end-to-end.**

- Kernel: `include/mirage/persistent_kernel/tasks/blackwell/kda_recurrence_sm100.cuh`
  (warp-per-row, float4-vectorized, fp32 state in place, gated-RMSNorm
  epilogue). Standalone wrapper test (`tests/runtime_python/blackwell/
  sm100_kda_recurrence/`): correctness PASS (out 2e-2, state 1e-4 tol) at
  tokens 1/8/32.
- Perf (200-rep, 12 heads, D=128): tokens=1 12.3us (latency floor),
  tokens=8 12.3us (1.0 TB/s), tokens=32 **14.4us / 3.51 TB/s** state
  traffic — AT the measured achievable-BW floor for 50MB (control: ~3.5
  TB/s). vs v1 thread-per-row: 41us/115us → 3.3x/8x. Optimization frozen
  (>=2 TB/s target met).
- Version history: v1 thread-per-row correct but uncoalesced; v2
  warp-per-row 84% wrong — root cause the **NUM_THREADS trap**:
  `worker_config.h` pins `constexpr NUM_THREADS = 128` while Blackwell
  workers launch `WORKER_NUM_THREADS = 256`, so warps 4-7 re-processed
  rows (signature: out uniformly ~1/sqrt(2) of ref, worst state rows
  starting at 4). Fix: stride by `blockDim.x` (the `inkling_sconv` idiom);
  all 256 worker threads do enter `_execute_task`.
- MPK registration complete (7 files + 1): TASK_KDA_RECURRENCE_SM100=501
  (new Kimi group 500-549) in `runtime_header.h`; include in blackwell
  `task_header.cuh`; decl in `task_register.h`; impl in `task_register.cc`;
  dispatch in `graph.cc`; **`task_type_to_name` in `runtime.cc`** (missed
  at first — generated dispatch emits the enum NAME; an absent entry
  yields `task_type ==  &&` and a codegen-side nvcc error); generic
  `kda_recurrence_layer` in `persistent_kernel.py`.
- API shape (7-input/3-output hard limit forced packing): inputs q/k/
  raw_gate [T,H*Dk] bf16, v/out_gate [T,H*Dv] bf16, raw_beta [T,H] bf16,
  head_params [H,1+Dk+Dv] fp32 (A_log ++ dt_bias ++ rmsnorm weight, folded
  per head at load); outputs out [T,H*Dv] bf16 + state [T,H*Dv*Dk] fp32
  in-place. grid (T,H,1), imap slices dim0 by tokens / dim1 by heads,
  head_params (-1,0,-1). Scalars (lower_bound, scale, eps) via params as
  float bits.
- Pipeline test `test_kda_recurrence_testmode.py` PASS (out diff 0.0,
  state diff 4.8e-7). Build trap: `setup.py build_ext` does NOT track
  `libmirage_runtime.a` — after a C++ rebuild, `touch
  python/mirage/_cython/core.cpp` to force the core.so relink.

Chain milestone (same day): `tests/runtime_python/test_mode/
test_kda_chain_testmode.py` — 3x inkling_sconv (q/k/v channel groups) ->
kda_recurrence -> linear o_proj [1536->7168], tokens=32, one task graph.
**PASS** (sconv states exact, recurrence out 9.8e-4, o_proj 3.9e-3).
Timing (three measurements that only make sense together):
- 1-block chain wall: 131.3 us/iter; single-tiny-task control: 154.9;
  8x SERIALIZED blocks (each block's conv input = previous block's o_proj
  out): 133.4 — wall is per-invocation host-launch bound, and 7 extra
  blocks of real work cost ~2 us of wall.
- torch.profiler GPU decomposition of the 8-block graph:
  worker_kernel 128.1 us/iter (+ scheduler 124.7 concurrent) →
  **~16 us per KDA block executed inside the megakernel** at tokens=32
  (consistent with 384 recurrence tasks / 148 workers ≈ 2.6 waves ≈ 14 us
  + sconv/linear waves).
- vs eager (block4 prod TP trace, B=32, re-measured): the fork's
  `_fused_kda_decode_kernel` is 47.9 us/replay / 3 blocks = **16.0
  us/block** — and it already fuses conv + recurrence + gated norm. So
  the recurrence itself was ALREADY at the byte floor in eager; the
  megakernel's per-block ~16 us additionally absorbs the o_proj linear
  and all inter-kernel gaps. The real remaining prize in the attention
  region is the projection GEMM soup + gaps + MLA glue (the ~0.35 ms
  region vs ~0.10-0.15 ms fused-granularity bound from the byte-floor
  study), not the KDA recurrence kernel.
Faithful-gate caveat: per-launch comparisons are meaningless (launch floor
~130-155 us); the win exists only when one persistent launch spans many
layers/steps, which is production MPK's shape (one launch per decode step).

Faithful-shape 3-block assembly (bench at /node-storage/var/mpk_floor_out/
bench_kda_block_replicated.py): per block = lin_q/k/v [7168->1536] +
lin_g + lin_fa [7168->128] + lin_fb [128->1536] + 3x sconv + recurrence +
o_proj [1536->7168] (~112MB weights + 50MB state), blocks serialized.
worker_kernel 147.8 us GPU / 3 = **~49 us per full KDA block** inside the
megakernel at tokens=32 (~3.3 TB/s effective on ~162MB/block), vs the
eager block4 KDA-block region ~87 us/block incl gaps → **~1.7-1.8x**.
Caveats: inkling-sconv activation semantics (residual, no silu) and the
0.17MB beta GEMM skipped — byte/shape-faithful, not bit-faithful; no
o_proj AR (single GPU). Headroom left: linear-task efficiency (~75% of
achievable BW) and fusing the tiny f_a/f_b hops.

Exact-semantics conv DONE (same day): `kda_sconv_sm100.cuh`
(TASK_KDA_SCONV_SM100=502, `kda_sconv_layer`; causal_conv1d_update
semantics — silu, no bias, per-sequence state [T, C*(K-1)] channel-major).
Wrapper + testmode + exact chain PASS; 3-block assembly unchanged
(~49 us/block — bytes identical).

MLA-in-MPK recon (2026-08-30): mirage ships a full MLA decode task family
from the dsv3 work — `mla_mtp_decode_tp8_sm100.cuh` has a 16-head tile
(128/8); K3-TP8 needs 12 heads/rank → ride the 16-head kernel with 4
zero-padded heads (MLA decode is KV-bandwidth-bound; KV reads are shared
across heads, so padding is ~free). Plus mla_kv_cache_gather + reduce
tasks exist. VERDICT: the 4-block megakernel needs assembly + a
padding-aware builder, not new kernels. Remaining after that: MoE region
(MXFP4 expert tasks absent — the known gap) and the multi-GPU o_proj AR.

## Open risks

- MXFP4 tasks still absent (MoE-side; not needed for attention target).
- Transpiler path still blocked (z3 4.16 wheel) — irrelevant to MPK.
- sm_103a intrinsics: router + sconv compile on 103a; recurrence task uses
  plain smem+FMA (no tcgen05) — low risk.
