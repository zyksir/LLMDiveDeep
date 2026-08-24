# Mega MoE research: what it is, kernel-vs-sequential plan, Kimi-K3 fit

Status: survey complete, awaiting Yikai's review before implementation.
Date: 2026-08-22. Sources: deepseek-ai/DeepGEMM (via sglang's
`sgl-deep-gemm==0.1.5.post3` pip package), sgl-project/sglang @ master
(cloned to `/node-storage/var/sglang-src`), our trt-llm fork
`optimized/k3 @ 165cdc3dcf`.

## 1. What Mega MoE is

ONE persistent kernel that owns the entire distributed MoE forward:

    token dispatch (EP all-to-all over an NVLink symmetric buffer)
    → FC1 grouped GEMM → activation → FC2 grouped GEMM
    → combine (all-to-all back) [+ top-k weighted reduce]

Communication is overlapped with compute at TILE granularity inside the
kernel: while one tile's tokens are still arriving over NVLink, already-
arrived tiles run their GEMMs; combine of finished tiles overlaps the
remaining GEMMs. This is the DeepSeek-V4-era replacement for the
sequential pipeline (dispatch kernel → grouped GEMM → activation kernel
→ grouped GEMM → combine kernel), whose stages each expose their full
comm latency.

Entry points in DeepGEMM:
- `deep_gemm.fp8_fp4_mega_moe(out, l1_weights, l2_weights, symm_buf,
  recipe=(1,1,32), activation=..., activation_clamp=..., fast_math=...)`
  — FP8 (UE8M0-scaled) activations × FP4 weights. **This is K3's exact
  W4A8 MXFP4/MXFP8 recipe.**
- `deep_gemm.bf16_mega_moe(...)` — BF16 everything (what
  `mok_vs_deepgemm/` benchmarked on 2026-08-18 at DSV3 shapes).
- Setup: `get_symm_buffer_for_mega_moe(...)` (rendezvous'd NVLink symm
  buffer, capacity = max tokens/rank), `transform_weights_for_mega_moe`
  (+ gate/up interleave `[g0..7,u0..7,g8..15,...]`),
  `mega_moe_pre_dispatch(topk_ids, topk_weights, ...)` (routing metadata
  staged into the symm buffer before the fused kernel).

## 2. Where it ships

| stack | integration |
|---|---|
| DeepGEMM | the kernels themselves (`fp8_fp4_mega_moe`, `bf16_mega_moe`); pip: `sgl-deep-gemm==0.1.5.post3` |
| sglang | `srt/layers/moe/mega_moe.py` (generic, DSv4) + **`srt/models/kimi_k3.py` — a complete K3 latent-MoE integration** (a2a backend `is_megamoe()`); capacity env `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` (default 8192/rank); K3 has NO non-mega fallback when enabled |
| our trt-llm fork | TWO backends under `modules/fused_moe/mega_moe/`: `MegaMoEDeepGemm` (wraps `fp8_fp4_mega_moe`; gates: SM100/103, `quant==W4A8_MXFP4_MXFP8`, hidden%512==0, inter%512==0) and `MegaMoECuteDsl` (in-tree CuTe DSL port, NVFP4-only — wrong quant for K3) |

## 3. Task 1 — kernel vs sequential A/B (plan)

**Claim to verify**: above some per-rank token threshold, the fused
kernel beats the sequential version because comm is overlapped.

**Comparators** (same math, same EP8, same K3 shapes,
hidden=MOE_LATENT=3584, inter=3072, experts=896, top-k=16):
1. `mega` — `fp8_fp4_mega_moe` (one kernel).
2. `seq` — the same a2a algorithm unfused: `mega_moe_pre_dispatch` +
   dispatch, grouped FC1, activation, grouped FC2, combine as separate
   launches (DeepGEMM exposes the grouped GEMMs; dispatch/combine via
   DeepEP or the symm-buffer copies serialized). This isolates ONLY the
   overlap benefit.
3. `ours` — the replicated-EP path we ship (col-AG latent → local
   experts on every rank → fused finalize+AR). Different algorithm,
   practically the decision-relevant baseline: dispatch-EP moves each
   token once (bs×topk rows total across ranks) while replicated-EP
   gives every rank all tokens; crossover expected where GEMM+dispatch
   cost < replicate+AR cost.

**Sweep**: tokens/rank ∈ {8, 32, 128, 512, 2048, 8192} × EP8, CUDA
events, max-over-ranks, uniform routing (production-realistic — see
`k3-serving-decode-composition` note on dummy-weight collapse), warm
symm buffer, clocks locked. Report µs + the threshold where each pair
crosses. Kernel-only timing (no model, no serving), per Yikai's scope.

**Harness**: extend `mok_vs_deepgemm/` (its `deepgemm_backend.py`
already does symm-buffer setup + `bf16_mega_moe`; switch to
`fp8_fp4_mega_moe` + add the `seq` and `ours` arms). Requires
`pip install sgl-deep-gemm==0.1.5.post3` in trt-k3 (not currently
installed in either container).

## 4. Task 2 — Kimi-K3 latent-MoE fit (answer: already done upstream)

**sglang's K3 integration proves the adaptation** — and DeepGEMM's
kernel already carries the K3 specifics:

- `fp8_fp4_mega_moe(..., activation="situ")` — the **SiTU kernel
  exists in DeepGEMM** and bakes K3's constants (asserted upstream:
  `beta=4.0, linear_beta=25.0`).
- Latent semantics: the kernel's "hidden" is just the token width — K3
  passes MOE_LATENT=3584 (7×512 ✓ TMA gate), inter 3072 (6×512 ✓).
- Quant: exact match — our checkpoints are W4A8_MXFP4_MXFP8, the
  kernel's native recipe `(1,1,32)` (UE8M0 per-32 scales).
- Routing: pre-dispatch consumes raw `topk_ids/topk_weights`
  (`noaux_tc` gate output) — our radix router produces exactly these.

**What integrating it into OUR stack changes (the real work):**
1. **EP comm pattern flips** from replicate+AR to dispatch+combine.
   Our fc1-shard/col-AG and the fused finalize+AR tail all live in the
   replicate+AR world; under Mega MoE they are replaced wholesale by
   the in-kernel a2a. The B10 wins that SURVIVE: radix routing (feeds
   pre_dispatch), fused front for gate+fc1 (fc1 stays outside the mega
   kernel — it projects hidden→latent BEFORE dispatch), shared-expert
   overlap. The wins that DON'T: col-AG+quant (kernel quantizes/moves
   tokens itself), fused finalize+AR (combine is in-kernel).
2. Weights need the mega layout: `transform_weights_for_mega_moe` +
   gate/up interleave (one-time, load-time).
3. Symm buffer capacity must cover the prefill chunk
   (tokens/rank ≤ cap; sglang default 8192) — sizing interacts with
   `max_num_tokens`.
4. Shared experts run replicated (tp1) under a2a — sglang's convention;
   our TP-sharded shared experts would need the same switch.
5. trt-llm side: `MegaMoEDeepGemm` backend exists but its ctor
   currently types the activation as Swiglu; the `activation` string it
   forwards must carry "situ" (small patch, mirrors trtllm-gen's
   Swiglu+hidden_act=="situ" convention). `MegaMoECuteDsl` is NVFP4 and
   NOT usable for K3 without a quant port.

## 5. Expected economics (hypothesis for task 1 to confirm)

Per-rank work under dispatch-EP scales with tokens×topk/EP (each token
computed once globally); under replicate-EP every rank computes every
token's local experts (same total FLOPs at uniform routing, but
replicate pays col-AG of the FULL latent + full-batch AR, while
dispatch pays 2× a2a of topk-expanded rows). Small bs: a2a latency
(two hops, in-kernel) vs one AR + one AG — replicate likely wins
(that's why decode ≤128 stays on the current path). Large prefill
chunks: overlap hides the a2a and the mega kernel should win —
sglang routes ALL K3 batches through it, suggesting the crossover is
low on their hardware. THRESHOLD = task 1's deliverable.

## Open questions for Yikai's review

1. Which `seq` decomposition do you want as the primary comparator —
   unfused-same-algorithm (isolates overlap only) or our replicated-EP
   (decision-relevant)? Plan measures both.
   measure both
2. sgl-deep-gemm 0.1.5.post3 vs deepseek-ai/DeepGEMM master — pin which?
   (sglang's package is what carries the SiTU kernel; verify the
   deepseek repo has it too.)
   check both version, choose which ever is better
3. Decode: keep the current path ≤128 tokens and use Mega MoE for
   prefill only (sglang's shape), or sweep decode sizes too?
   sweep decode as well

## Environment verification (2026-08-22, trt-k3)

`pip install sgl-deep-gemm==0.1.5.post3` — installed and verified:
`deep_gemm.fp8_fp4_mega_moe` present (signature: `y, l1_weights,
l2_weights, sym_buffer, ..., activation: str = 'swiglu',
activation_clamp`), `mega_moe_pre_dispatch` present, and the SiTU
activation is compiled in (`include/deep_gemm/impls/
sm100_fp8_fp4_mega_moe.cuh` and `_C.so` both carry it; pass
`activation="situ"`). The task-1 harness can start immediately on
approval.

## Task-1 execution log (2026-08-22, first actual runs)

`bench_megamoe_k3.py` written (EP8, K3 shapes, tokens/rank 8..8192). Env
obstacles found and handled:

1. deep_gemm's mega runtime needs SYS_NICE (FIFO thread priority) — the
   standard containers abort in glibc; dedicated `trt-mega` container
   created with `--cap-add SYS_NICE --ulimit rtprio=99`.
2. Killed mpirun runs leave stale OpenMPI vader segments in host /dev/shm
   (`--ipc=host`) whose robust mutexes assert on the next run — wipe
   `/dev/shm/vader_segment.*` after any aborted run.
3. **pip `sgl-deep-gemm==0.1.5.post3` (latest on PyPI) is too old for K3**:
   its SymmBuffer asserts swiglu-only (no SiTU — sglang master's K3 path
   requires a newer build), and its `fp8xfp4` symm-buffer path throws
   (`Unknown device: -96`), bf16 path throws `Owner died` from inside its
   allocator while plain torch symm_mem works in the same container
   (verified by probe). → installing deepseek-ai/DeepGEMM master from git;
   sglang's vendored `sglang.kernels` is the fallback source.
4. DeepGEMM master (2.6.1+559d79f) works on GB300 but the API differs from
   sglang's fork: there is NO `mega_moe_pre_dispatch` (that kernel is
   sglang's own — it fuses quant+staging); staging is direct copies into
   the SymmBuffer views (`buf.x/x_sf/topk_idx/topk_weights`), mok-style.
   Weight contract for `fp8_fp4_mega_moe`: payload dtype **int8**
   (`kPackedFP4 == torch::kInt8`, 2 fp4/byte), scales **int32-packed
   UE8M0** (4/word, K/128 words) and **MN-major** (`stride(-2)==1`) for
   the TMA check; `transform_weights_for_mega_moe` then interleaves
   gate/up and transposes SFs for UTCCP. `activation="situ"` is REJECTED
   by master (`apis/mega.hpp:180: activation == "swiglu"`) — SiTU lives
   only in sglang's fork; it is an epilogue-local template swap, so
   swiglu timings are representative.

## Task-1 RESULTS (2026-08-23, GB300 TP8/EP8, DeepGEMM master)

`fp8_fp4_mega_moe` (K3's exact fp8-act × fp4-weight recipe (1,1,32)) and
`bf16_mega_moe` both RUN at K3 shapes (hidden=3584, inter=3072, 896
experts, top-16, 112 local experts/rank). Timing = staging copies + one
fused kernel, 50 iters, max-over-ranks; uniform random routing; x
pre-quantized outside the loop (sglang fuses that quant into
pre_dispatch, so add ~one quant kernel ~5 µs for a serving-honest read).
Logs: `/node-storage/var/megamoe_{fp4,bf16}.log`.

| tokens/rank | global tokens | mega fp8xfp4 | mega bf16 | our deployed layer (opt / stock) |
|---|---|---|---|---|
| 8    | 64    | **143.9 µs** | 280.2 µs  | ~230 / 255 µs (bs64 decode) |
| 64   | 512   | **286.9**    | 868.4     | 413 / 479 µs |
| 512  | 4096  | **422.7**    | 1197.0    | 1150 / 1480 µs |
| 2048 | 16384 | **958.8**    | 2038.8    | 2882 / 4908 µs |
| 8192 | 65536 | **3668.7**   | 7558.8    | (beyond layer sweep) |

Comparison caveats (why this is indicative, not apples-to-apples):
- Different algorithm AND coverage: mega = dispatch-EP (each token owned
  by one rank, moved twice over NVLink) covering dispatch+FC1+act+FC2+
  combine, finalized output, NO trailing AR needed. Our layer = 
  replicate-EP covering additionally routing gate, shared experts,
  fused finalize+AR+rmsnorm. Matched at equal global tokens the per-rank
  routed FLOPs are identical (16 T token-expert pairs/rank at uniform).
- Our layer numbers are the honest multi-iter graph numbers from
  `bench_moe_layer_tp8_b300_situ` sweeps (deploy new/baseline columns).

**Threshold read (task-1 deliverable):**
- Prefill regime (global ≥ 4096, i.e. ≥512 tokens/rank): mega wins big —
  **2.7× vs our optimized layer at 4096, 3.0× at 16384**, and scaling is
  sub-linear in tokens (422→959 µs for 4×) because the in-kernel a2a
  overlap keeps SMs busy while our sequential AG → grouped GEMM → AR
  exposes every comm phase. Even discounting ~15% for the layer's extra
  stages (gate, shared, norm), the gap is decisive.
- Mid sizes (global 512): mega 287 vs our 413 µs — still ahead (~1.4×),
  same discount caveat applies; crossover with a fully-loaded comparator
  sits somewhere below 512 global tokens.
- Decode (global 64): mega 144 µs vs our full layer 230 µs, BUT the mega
  number excludes gate/shared/norm and its dispatch-EP semantics don't
  match TP8 replicated-token decode (tokens would first need
  reduce-scatter-style ownership). Decode ≤128 stays on the current
  replicate+AR path; the a2a latency floor (~144 µs at 8 tok/rank) is
  ~5× our fused finalize+AR tail budget.

**Prefill usage sketch (keep GPUs busy):** K3 prefill steps are ~2048
global tokens today (scheduler does not cross-request pack). At 2048
global (256/rank, interpolating) mega is ~350–400 µs vs our ~1000 µs
opt layer middle — the single biggest prefill lever we have measured,
larger than all B10 front-stage wins combined. Requirements to land it:
token-ownership split before the MoE block (rank owns its contiguous
token slice — natural in sequence-parallel prefill), SiTU port (sglang
fork carries it; epilogue swap in the JIT template), and the trt-llm
`MegaMoEDeepGemm` backend already gates for exactly our quant mode.
Next measurement: a `seq` arm (same a2a algorithm, unfused launches) to
isolate the overlap-only benefit, per the plan's comparator 2.

## Task-1 comparator 2 (2026-08-24): fused vs SAME-algorithm unfused a2a

`bench_megamoe_seq.py`: identical a2a-EP algorithm as separate ops
(torch all_to_all_single dispatch -> sort/pad -> DeepGEMM
m_grouped_bf16_gemm_nt_contiguous FC1 -> swiglu -> grouped FC2 -> a2a
combine + weighted reduce), bf16, K3 shapes, fixed routing plan per size.

| tokens/rank | seq (unfused) | bf16_mega_moe (fused) | mega advantage |
|---|---|---|---|
| 8    | 525.7 us  | 280.2 us  | 1.9x |
| 64   | 1504.0    | 868.4     | 1.7x |
| 512  | 2419.1    | 1197.0    | 2.0x |
| 2048 | 5585.7    | 2038.8    | 2.7x |

The overlap+fusion benefit GROWS with size (comm turns payload-bound and
the fused kernel hides it under tile compute; the unfused pipeline
exposes both a2a phases fully). Caveat: the seq arm's permutations are
plain torch ops, so it is an upper bound on sequential cost — but the
a2a bytes and GEMM work are identical, and even a perfectly-permuted
seq pipeline keeps the two exposed a2a phases that dominate the gap.

Task-1 answer set now complete:
- comparator 2 (same algorithm, unfused): mega wins 1.7-2.7x, growing.
- comparator 3 (our replicated-EP deployed layer): mega wins ~1.4x at
  512 global tokens to 3.0x at 16K global (prefill regime).
- decode <=128 tokens stays replicate+AR (a2a floor ~144 us/step).
