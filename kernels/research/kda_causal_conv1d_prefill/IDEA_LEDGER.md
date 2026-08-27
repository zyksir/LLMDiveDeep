# Idea ledger

## Survey

### TensorRT-LLM native CUDA causal-conv1d

Source: `cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu`.

The unchanged native algorithm assigns a CTA to `(sequence, channel)`, streams
the temporal axis in 128-bit per-thread chunks, exchanges one neighboring chunk
through warp shuffle/shared memory, accumulates in FP32, and updates one state
row per channel. Its channel-major input makes temporal loads contiguous. The
KDA caller pays a separate token-major-to-channel-major transpose before this
kernel and later consumes a transposed view.

Useful mechanisms:

- fixed `W` specialization and fully unrolled FP32 FMA;
- 128-bit vector transfer along the physically contiguous dimension;
- exact short-sequence state update;
- direct initial-state injection into the first temporal window.

Mismatch with the requested direct path:

- its `(sequence, channel)` geometry serializes all tokens of one channel;
- token-major KDA input requires a physical layout-copy kernel first;
- native output is channel-major, not the requested contiguous token-major
  package output.

### TensorRT-LLM Triton prefill

Source lineage: git `HEAD` of
`tensorrt_llm/_torch/modules/mamba/causal_conv1d_triton.py`.

The one-launch Triton kernel uses `(sequence, 8-token chunk, 256-channel tile)`.
It reads token-major physical storage through a transposed metadata view,
parallelizes channels and temporal chunks, uses FP32 accumulator preload, and
has the same rolling state logic. The working tree has two local changes:
explicit FP32 multiply conversion and optional padded-input copy. Those changes
are not treated as an unchanged baseline.

This is the strongest algorithmic comparison: direct row-major access and one
launch. It is also the nearest geometry to test rather than merely comparing
against the multi-launch native pipeline.

### CuTeDSL vectorized elementwise template

Source:
`optimizations/blackwell/dict/templates/elementwise_vectorized_fp16/run.py`.

Useful mechanisms:

- `cute.zipped_divide` over the contiguous last dimension;
- one thread owns an aligned vector and uses `cute.autovec_copy`;
- FP16 input is widened to FP32 register values;
- cached `cute.compile` callable with DLPack outside timed execution.

Mismatch:

- causal convolution gathers up to four token rows and has sequence boundaries;
- state update is an additional side effect;
- a bulk-plus-tail two-kernel template is forbidden by the one-launch contract.

### CuTeDSL depthwise-convolution template

Source:
`optimizations/blackwell/dict/templates/depthwise_conv_ref/run.py`.

Useful mechanisms:

- recognizes depthwise convolution as a low-arithmetic-intensity stencil rather
  than a tensor-core GEMM;
- vectorizes the contiguous dimension;
- compares no-SMEM register-window and SMEM-staging variants;
- preserves FP32 accumulation in accepted variants.

Mismatch:

- the template is fixed-shape NCHW 2-D spatial convolution;
- it has no packed-varlen boundaries, cached history, or in-place slot update;
- its per-plane/row geometry does not directly map to token-major channels.

### CuTeDSL cumsum scan template

Source:
`optimizations/blackwell/dict/templates/cumsum_scan_ref/run.py`.

Useful lesson: serial dependence along one axis should remain local to a thread
or warp while independent axes are parallelized. For this operation `W <= 4`,
so there is no long scan: every output token can independently gather its
bounded causal window. State update needs only the final CTA/chunk.

### Conv1d implicit-GEMM template

Source:
`optimizations/blackwell/dict/templates/conv1d_implicit_gemm_ref`.

Rejected as a starting algorithm. It is a dense cross-channel convolution with
im2col TMA and tcgen05 MMA, requires transformed channels-last storage and large
shape divisibility, and has no cached state semantics. This contract is
depthwise with only `W <= 4`; tensor-core setup would dominate and would not
remove packed-sequence control flow.

### Upstream example index

The skill's listed Ampere `elementwise_add.py` and `elementwise_apply.py` raw
URLs returned HTTP 404 on 2026-08-26. The local promoted vectorized elementwise
and depthwise templates above are newer CuTeDSL 4.5.2 references and were used
instead.

## Falsifiable ideas

| ID | Idea | Prediction | Test | Status |
|---|---|---|---|---|
| I00 | Direct token-major SIMT kernel, one CTA per `(sequence, token tile)`, vectorized over channels | Removes native layout copies and exposes channel/token parallelism; should beat the native pipeline at main BF16 W4 shapes | Compare full-pipeline native latency and one-launch CuTe latency | **confirmed:** beats native on 72/72 matrix shapes, 2.15–4.90x |
| I01 | One thread owns 8 contiguous FP16/BF16 channels, loading each causal row as a 128-bit vector | Coalesced row-major accesses and one 128-bit output store should reduce instruction count versus scalar channels | Compare against one-launch TRT Triton | **useful:** beats Triton on 52/72 shapes; scalar/vector-4 not pursued because the direct vector-8 result already won short/medium shapes and long shapes showed a token-tiling trend |
| I02 | Use 8-token CTA tiles like TRT Triton | Bounded register work with enough CTAs; should compare closely to Triton's one-launch path | Sweep token tile 4/8/16 with identical vector width | **refuted as universal:** tile 4 wins short/medium; tile 16 wins long; adaptive selection retained |
| I03 | Let only the CTA containing the sequence tail update state | Avoids atomics and duplicate state stores while preserving one launch | Correctness on short, mixed-state, permuted-slot cases | **refuted:** tail CTA races chunk-zero initial-state reads. Replaced by chunk-zero snapshot-then-update, matching native/Triton ordering |
| I04 | Preload channel weights/bias once per thread and reuse across token tile | Reduces repeated weight traffic; larger token tiles improve reuse until register pressure/serial work dominates | Tile sweep and latency by T/B | **confirmed tradeoff:** tile 4/8/16 large-shape latency 0.1581/0.1036/0.09195 ms; compile 1.46/2.60/4.98 s |
| I05 | Match native fast exponential for SiLU | Preserve accepted TRT tolerance while avoiding a slower high-precision path | Adversarial SiLU comparison | **confirmed:** `cute.math.exp(..., fastmath=True)` passes native tolerance; worst focused errors FP16 1.22e-4, BF16 1.95e-3, state bitwise |

## Decision gate

### Pre-candidate evidence

Machine floors at the fixed 1500 MHz clock:

- one-launch floor: 0.003013 ms;
- 1 GiB device copy: 0.339019 ms, 6334.4 GB/s counting read plus write;
- runtime-reported SM count/capability: 148 / 10.3;
- FP32 peak used for the arithmetic lower bound: 56.832 TFLOP/s
  (`148 * 128 lanes * 2 FLOP * 1.5 GHz`).

Main BF16 W4 baseline range:

- native three-kernel pipeline: 0.02959–0.25206 ms;
- git-`HEAD` one-launch Triton: 0.02924–0.05879 ms.

The largest shape `(T=8192,D=3072)` has:

- mandatory-byte lower bound: 100.73–100.92 MB;
- bandwidth floor: 0.01590–0.01593 ms;
- arithmetic floor: 0.00354 ms;
- serial four-FMA chain floor: 0.0000107 ms;
- defensible combined floor: 0.01590–0.01593 ms;
- measured Triton: 0.05097 ms (B1), 0.05879 ms (B8), or 31.2%/27.1%
  of the lower-bound SOL respectively.

Small/medium shapes are launch dominated by the conservative model, while their
measured ~0.029 ms latency remains roughly 10x the isolated launch floor. The
native path loses up to 4.3x to the one-launch Triton path at large B8 shapes,
confirming that layout copies and channel-serial geometry are material.

### Decision: PROCEED

The direct row-major CuTe idea is falsifiable and there is measurable headroom
against both the native pipeline and the byte/launch lower bounds. The
one-launch TRT Triton result, not the native pipeline alone, remains the
acceptance comparison. The CUDA runtime reports SM103, but the product name is
`NVIDIA L20D`; no B200 claim will be made.
