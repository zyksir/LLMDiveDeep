# Journal

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
