# Journal

## 2026-08-27 — round-3 finalization: single config, Triton gate PASS 18/18

- User acceptance update applied: one configuration for every shape, Triton
  gate hard, SOL gate best-effort with plateau evidence.
- Sweep 6 (FP32 ring) closed the campaign: at group-span 4 the FP32 ring is
  the new best on every long shape (dense B8 0.0286→0.0268/0.0278 ms; strided
  long 0.0408→0.0399/0.0401 ms); at span 8 it regresses via register
  pressure. `v4_tile16_gs4_fr` chosen as the single config — worst-case +3.7%
  vs per-shape best, all other shapes within ~2.3%.
- Selected config made the backend default (env unset): stream W4, vec 4,
  128 threads, tile 16, group loads span 4, FP32 ring, tanh SiLU, BF16 param
  fragments (bitwise-identical), FP32 accumulation. W2/W3 keep the direct
  kernel.
- Final gates: case correctness 17/17, full matrix 108/108 (states bitwise);
  five-round main benchmark PASS 18/18 vs unchanged git-HEAD Triton,
  1.774–3.949x, worst margin 77% (no <5% rerun triggered). Receipt:
  `results/r3_acceptance_summary.json`.
- Final NCU (selected + Triton strided): dense long 24.5–25.2 µs (47.6–48.1%
  compute, 31.0–31.8% DRAM SOL, 72 regs); strided long 33.9–34.4 µs
  (53.4–54.4% compute, 41.8–42.6% DRAM SOL, 64 regs). FP32 ring cut ALU pipe
  64–66%→37–39% and cycles/issue 14.2→10.3–11.5; residual is long-scoreboard
  (DRAM latency) at ~52% of cycles under register-bound occupancy. No
  counter-indicated mechanism remains inside the SIMT architecture;
  exploration stopped per the acceptance update.
- SOL vs calibrated roofs: 52% dense D3072 long, 60–61% dense D1536 long,
  ~65% strided production long, 31–40% launch-floor-bound short/medium.
- REPORT/README/ledger rewritten to the round-3 framing; `r3_final_inner.sh`
  and `build_r3_acceptance.py` added for reproduction.

## 2026-08-27 — round-3 mechanism campaign

- Calibrated attainable roofs with same-pattern streaming copies
  (`calibrate_roof.py`): contiguous T8192/D3072 attains 6957 GB/s (1 GiB roof
  was loose in the tight direction), strided production 5845 GB/s (partial
  128 B lines at row edges), short/medium payloads launch/ramp bound.
- Kimi-K3 production strided shape (D=4608, stride(0)=4752, no bias) added as
  first-class gate rows; harness generates true strided views; wrapper
  derives alignment/vector width from real pointers (rows are 32B- but not
  128B-aligned).
- GPU lease unblocked per user authorization: runs pinned to GPU 7 (0%
  utilization, co-resident idle memory), 1500 MHz lock kept, utilization
  verified before each run.
- Mechanisms confirmed: tanh SiLU (-16–25%), group-batched loads (-18–19%,
  the largest win; long-scoreboard share fell as predicted), FP32 ring at
  span 4 (-3–6%). Refuted: vec8/16 in the conv kernel, L2 prefetch hints,
  span 12, 256 threads, FP32 params (bitwise-identical, perf-neutral).
- Precision audit (`precision_audit.py`): FP32 accumulation everywhere; BF16
  param fragments bitwise-identical to FP32 fragments; config choices
  bitwise-invariant; tanh-vs-native worst max-abs 3.13e-2 within the frozen
  BF16 atol 1e-1 (expdiv 1.95e-3); states bitwise in both modes. tanh
  selected for speed with the delta documented; expdiv remains selectable.

## 2026-08-27 — second-round result

- The width-4 streaming ring reduced redundant input-row loads. Controlled
  vector/CTA/tile sweeps selected 128 threads and eventually 12-token chunks.
- Early long B1 NCU moved from 84 to 55 registers and 31.25% to 56.25%
  theoretical occupancy, but remained compute/scoreboard limited.
- One-vector software lookahead was required for D3072; removing it regressed
  T8192/B1 from about 0.0451 to 0.0501 ms.
- Input-precision parameter registers preserve exact values and FP32
  accumulation while reducing live parameter storage.
- Architecture-specific `prefetch.global.L2` was correctness-safe but slower
  (D3072/T8192/B8 0.06499→0.06683 ms) and was rejected.
- The largest mechanism win replaced the rectangular `max_blocks * B` grid
  with exact packed token-block prefixes embedded from existing CPU sequence
  metadata. This removed inactive uneven-B8 CTAs without an extra conversion or
  launch and reduced D3072/T8192/B8 to about 0.0461 ms.
- TensorSSA SiLU and tile12 produced the final W4 regime. W2/W3 retain the
  direct CuTe regime; no selected path invokes Triton.
- Full native correctness passes 72/72. Five-round primary timing passes the
  TRT gate 12/12 with `1.125–3.466x` speedups.
- Same-clock launch and copy floors were remeasured at 0.002963 ms and
  6335.91 GB/s. Primary analytical SOL efficiency is only 25.6–35.5%, so the
  independent SOL gate fails 12/12.
- Final long D3072 NCU reports 68 registers, 43.75% theoretical and about 40%
  achieved occupancy, 36% L1TEX scoreboard stalls, about 59% compute SOL, and
  only 20.3–20.5% Memory Throughput/SOL.
- Final status: **REJECTED/UNQUALIFIED**. The profile identifies current compute
  pressure but does not prove a necessary alternative roof, so acceptance was
  not relaxed.

## 2026-08-27 — corrected independent acceptance gates

- Replaced the prior soft language with two independent hard gates across all
  12 primary BF16/W4 shapes:
  1. repeatably beat unchanged exact git-`HEAD` TRT Triton;
  2. reach at least 80% analytical SOL efficiency, plus at least 80% NCU Memory
     Throughput/SOL for long memory-bound shapes. Short/crossover shapes instead
     use 80% of the same-clock measured launch/dispatch roof.
- A different necessary roof is admissible only with authoritative NCU evidence
  and a recomputed defensible bound; the 80% threshold is not relaxed.
- Equal output/state work, FP32 accumulation, one launch, padded/state behavior,
  and timing boundaries remain fixed. No candidate-only untimed conversion or
  Triton fallback is allowed.
- The first-round kernel fails both the long-shape SOL gate and four TRT gates,
  so its status is now explicitly `REJECTED/UNQUALIFIED`.
- Added profile-linked hypotheses R2-I01 through R2-I10 before code changes.
- Implemented the first second-round mechanism behind explicit environment
  selection: a W4 four-phase register ring, width 2/4/8 channel vectors,
  32/64/128/256-thread CTAs, and 4–128-token chunks. A runtime outer loop with
  static four-token phases permits larger chunks without tile-wide live values
  or fully unrolled code growth. This implementation is not a candidate result
  until native correctness, benchmark, and NCU gates run.

## 2026-08-27 — relocation

- Moved the complete package from CuTeDSLGen into
  `/node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill`.
- Updated runnable README commands and the NCU harness to use the new package
  path while retaining `/node-storage/CuTeDSLGen` as the CuTe environment and
  GPU-lease provider.

## 2026-08-26 — contract and environment

- Created the durable package at
  `generated/blackwell_kda_causal_conv1d_prefill`.
- Read the repository's CuTeDSL kernel-writing skill and production package
  conventions before candidate implementation.
- Confirmed hardware-reporting discrepancy: host `nvidia-smi` reports eight
  `NVIDIA L20D` devices at capability 8.9, while the CUDA runtime inside the
  workload container reports `NVIDIA L20D`, capability 10.3, 148 SMs and
  287,428,771,840 bytes. Runtime compilation therefore targets SM103. No B200
  product claim is made.
- Confirmed TensorRT-LLM working tree state:
  - native CUDA source is available and treated as the authoritative baseline;
  - the Triton source has local FP32-accumulation and padded-copy edits;
  - the git-`HEAD` Triton source hash is
    `68860408e57d076765d1e0be13cc87b9a3811ea9424de670d1c39821e5332432`;
  - the git-`HEAD` native CUDA source hash is
    `8c31ce1bc4e6137b530728f02d4052f0117c0fbba1d64ede8f9b9864aa266470`.
- Surveyed the unchanged native algorithm, git-`HEAD` Triton algorithm, local
  CuTeDSL vectorized-elementwise/depthwise-convolution/scan templates, and the
  dissimilar implicit-GEMM conv1d template. Details are in `IDEA_LEDGER.md`.
- Upstream example URLs listed by the local skill returned HTTP 404; recorded
  this and used the repository's promoted CuTeDSL 4.5.2 templates.

## 2026-08-26 — pre-candidate baselines and decision

- Added a backend-neutral runner, shared deterministic input/correctness/timing
  utilities, and isolated native/TRT-Triton backend modules. The runner shape
  loop contains no implementation-specific branches.
- Native baseline body remains the production sequence: transpose extraction,
  unchanged native CUDA op, and contiguous token-major materialization.
- Git-`HEAD` Triton is loaded from the exact source object after SHA256
  verification; the locally modified working-tree file is not imported.
- Baseline correctness passed normal, unscaled, adversarial, short, mixed
  initial-state, permuted-cache, no-bias/no-activation cases for FP16/BF16 and
  W2/W3/W4. Git-`HEAD` Triton padded output was explicitly skipped because its
  unchanged contract leaves that output uninitialized; native padded semantics
  passed.
- Pre-candidate artifacts:
  - `results/pre_candidate_correctness.json`
  - `results/pre_candidate_main_bench.json`
  - `results/pre_candidate_floors.json`
- Fixed-clock main BF16 W4 latency:
  - native pipeline: 0.02959–0.25206 ms;
  - git-`HEAD` one-launch Triton: 0.02924–0.05879 ms.
- Floors: launch 0.003013 ms; 1 GiB copy 6334.4 GB/s read+write; largest-shape
  mandatory-byte floor 0.01590–0.01593 ms.
- Decision: **PROCEED**. Direct row-major CuTe has a real comparison target and
  measurable headroom, but must beat or closely explain its gap to one-launch
  TRT Triton rather than relying on speedup over the native multi-kernel path.

Next: implement the minimal serious CuTe variants and gate every experiment on
native output/state correctness.

## 2026-08-26 — CuTe implementation and correctness

- Implemented a pure-SIMT CuTe DSL kernel that:
  - reads packed token-major input directly;
  - assigns CTAs to `(sequence, token tile, channel tile)`;
  - assigns each thread eight contiguous channels and uses 128-bit
    `cute.autovec_copy` loads/stores;
  - preloads FP32 weights/bias and accumulates in FP32;
  - fuses native-style fast SiLU;
  - preserves padded input and skips padded state;
  - updates state in the same launch.
- First implementation let the sequence-tail CTA update state. This failed
  output correctness while state itself was correct: the tail CTA could
  overwrite cached history before chunk zero read it.
- Corrected ordering by making chunk zero snapshot cached history into
  registers, update final state, and then compute all history-consuming output
  tokens. Later chunks never read cached state because token tile is at least
  four and `W-1 <= 3`. This exactly mirrors the safety argument in the TRT
  Triton implementation without adding a second launch.
- Focused correctness now passes 11 cases covering FP16/BF16, W2/W3/W4,
  normal/unscaled/adversarial data, short sequences, padded slots, mixed
  initial state, permuted slots, and no-bias/no-activation. State is bitwise
  equal to native. Artifact: `results/cute_t8_correctness_v2.json`.
- Full required matrix correctness passes 72/72 shapes directly against native.
  Artifact: `results/cute_auto_full_correctness.json`.

## 2026-08-26 — evidence-driven tiling

Quick BF16 W4 sweep, fixed 1500 MHz:

- tile 4: 0.008707 / 0.017643 / 0.158073 ms;
- tile 8: 0.011695 / 0.017789 / 0.103572 ms;
- tile 16: 0.018273 / 0.021488 / 0.091948 ms.

The three points are `(T128,D1536,B1)`, `(T1024,D1536,B8)`, and
`(T8192,D3072,B8)`. Larger tiles improve weight reuse and reduce CTA count on
long sequences but increase straight-line work/register pressure on short
sequences. Retained adaptive policy: tile 4 when maximum sequence length is at
most 1024, tile 16 otherwise.

A tile-32 trial produced no tracked exit status or JSON result and left the
terminal unavailable for roughly 23 minutes; no GPU process remained when the
terminal recovered. Because compile growth is already 1.46/2.60/4.98 s at
tiles 4/8/16 and tile 32 was operationally unstable, it is not accepted.

## 2026-08-26 — final matrix and standalone smoke

- Candidate full-matrix benchmark: 72 shapes, 153.60 s runner wall time,
  `results/cute_auto_full_matrix.json`.
- Baseline full matrix: native and git-`HEAD` Triton, 15.27 s runner wall time,
  `results/pre_candidate_full_matrix.json`.
- Candidate beats the native transpose+CUDA+materialize pipeline on 72/72
  shapes (2.15–4.90x, 3.43x geometric mean).
- Candidate beats git-`HEAD` one-launch Triton on 52/72 shapes (0.535–4.39x,
  1.88x geometric mean). The 20 losses are long-sequence cases.
- Main BF16 W4 candidate range: 0.00838–0.09283 ms. It beats Triton on 8/12
  main shapes and loses on all four `T=8192` shapes.
- Verified the standalone package through repository `uv run` with CuTeDSL
  4.5.2 / torch 2.9.1: `results/uv_standalone_smoke.json`.
