# KDA causal-conv1d prefill report

## Outcome

Implemented a durable one-launch CuTe DSL package for packed-varlen KDA prefill
causal conv1d. It directly reads token-major rows, vectorizes eight contiguous
channels per thread, accumulates in FP32, optionally applies bias and
native-style SiLU, writes contiguous token-major output, and updates cached
state without a second kernel.

Status: **correct and useful, but not universally faster than the strongest
one-launch TRT Triton path**.

- Full required matrix correctness: 72/72 against native output and state.
- Focused semantic suite: 11/11, including adversarial, short, padded, mixed
  initial-state and permuted-slot cases.
- Beats the native three-kernel pipeline on 72/72 shapes: 2.15–4.90x,
  3.43x geometric mean.
- Beats exact git-`HEAD` one-launch TRT Triton on 52/72 shapes: 0.535–4.39x,
  1.88x geometric mean.
- Main BF16 W4: wins 8/12 versus Triton; all four T=8192 shapes remain slower.

## Hardware and software

Identity reporting is inconsistent:

- host `nvidia-smi`: `NVIDIA L20D`, capability 8.9;
- CUDA runtime in the measured container: `NVIDIA L20D`, capability 10.3,
  148 SMs, 287,428,771,840 bytes.

The executable target is therefore SM103, but the device is not claimed to be
B200. GPU runs used the repository lease helper and a 1500 MHz SM clock lock.

- Candidate: nvidia-cutlass-dsl 4.5.2.
- Recorded matrix container: TensorRT-LLM 1.3.0rc23, torch
  2.12.0a0+5aff3928d8.nv26.05, CUDA 13.2.
- Standalone `uv run` smoke: torch 2.9.1+cu130, CUDA 13.0.
- git-`HEAD` Triton source SHA256:
  `68860408e57d076765d1e0be13cc87b9a3811ea9424de670d1c39821e5332432`.
- inspected native CUDA source SHA256:
  `8c31ce1bc4e6137b530728f02d4052f0117c0fbba1d64ede8f9b9864aa266470`.

## Pre-candidate gate and SOL lower bound

Decision: **PROCEED**.

At 1500 MHz:

- isolated one-kernel launch floor: 0.003013 ms;
- 1 GiB copy: 0.339019 ms, 6334.4 GB/s counting read and write;
- modeled FP32 peak: 56.832 TFLOP/s;
- main-shape native baseline: 0.02959–0.25206 ms;
- main-shape git-`HEAD` Triton baseline: 0.02924–0.05879 ms.

For BF16 W4 T8192 D3072, the conservative mandatory traffic is
100.73–100.92 MB. The resulting floors are:

- memory: 0.01590–0.01593 ms;
- arithmetic: 0.00354 ms;
- serial four-FMA chain: 0.0000107 ms;
- combined with launch: 0.01590–0.01593 ms.

The pre-candidate Triton result was 0.05097–0.05879 ms, leaving a falsifiable
gap while already removing the native layout-copy kernels.

## Design and variants

The accepted kernel uses a grid over `(channel tile, token tile, sequence)`.
One thread owns eight adjacent channels. For each token tile it:

1. preloads per-channel weights and optional bias to FP32 registers;
2. gathers up to four causal token/state rows with 128-bit row-major loads;
3. performs FP32 accumulation and optional fast exponential SiLU;
4. writes one 128-bit contiguous output vector.

Chunk zero snapshots initial cached state before updating the slot. This is
required: the rejected tail-CTA update raced chunk-zero history reads even
though final state itself looked correct.

Token-tile sweep on three BF16 W4 points:

| tile | T128 D1536 B1 | T1024 D1536 B8 | T8192 D3072 B8 | compile/setup |
|---:|---:|---:|---:|---:|
| 4 | 0.008707 ms | 0.017643 ms | 0.158073 ms | ~1.46–1.58 s |
| 8 | 0.011695 ms | 0.017789 ms | 0.103572 ms | ~2.60–2.73 s |
| 16 | 0.018273 ms | 0.021488 ms | 0.091948 ms | ~4.93–5.02 s |

Best variant is adaptive: tile 4 when maximum sequence length is at most 1024,
tile 16 otherwise. A tile-32 attempt produced no tracked exit status or JSON
and made the terminal unavailable for roughly 23 minutes; it was rejected.

## Full latency summary

All values below are one-launch candidate latency ranges over D={1536,3072},
T={128,1024,8192}, B={1,uneven 8}.

| dtype / W | CuTe range (ms) | native geometric speedup | Triton geometric speedup | Triton wins |
|---|---:|---:|---:|---:|
| FP16 / 2 | 0.008109–0.057200 | 4.03x | 2.18x | 9/12 |
| FP16 / 3 | 0.007825–0.069029 | 3.53x | 1.95x | 9/12 |
| FP16 / 4 | 0.008741–0.088605 | 2.91x | 1.63x | 8/12 |
| BF16 / 2 | 0.007828–0.058534 | 3.99x | 2.17x | 9/12 |
| BF16 / 3 | 0.007842–0.074090 | 3.43x | 1.88x | 9/12 |
| BF16 / 4 | 0.008380–0.092828 | 2.86x | 1.58x | 8/12 |

Main BF16 W4:

| D | T | B | CuTe ms | native ms | TRT Triton ms | vs native | vs Triton |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1536 | 128 | 1 | 0.008380 | 0.032523 | 0.030843 | 3.88x | 3.68x |
| 1536 | 128 | 8 | 0.008987 | 0.033088 | 0.030596 | 3.68x | 3.40x |
| 1536 | 1024 | 1 | 0.010762 | 0.033217 | 0.030594 | 3.09x | 2.84x |
| 1536 | 1024 | 8 | 0.017871 | 0.038438 | 0.030527 | 2.15x | 1.71x |
| 1536 | 8192 | 1 | 0.040698 | 0.100276 | 0.030661 | 2.46x | 0.75x |
| 1536 | 8192 | 8 | 0.058142 | 0.134691 | 0.031105 | 2.32x | 0.53x |
| 3072 | 128 | 1 | 0.008706 | 0.032841 | 0.030776 | 3.77x | 3.54x |
| 3072 | 128 | 8 | 0.012035 | 0.043873 | 0.030678 | 3.65x | 2.55x |
| 3072 | 1024 | 1 | 0.015031 | 0.033388 | 0.030406 | 2.22x | 2.02x |
| 3072 | 1024 | 8 | 0.026561 | 0.064377 | 0.030547 | 2.42x | 1.15x |
| 3072 | 8192 | 1 | 0.068906 | 0.189200 | 0.051229 | 2.75x | 0.74x |
| 3072 | 8192 | 8 | 0.092828 | 0.252461 | 0.059022 | 2.72x | 0.64x |

The complete 72-shape values are in the JSON receipts, not rounded tables.

## Correctness

- Full matrix: 72/72 output/state checks passed.
- Focused suite: 11/11 passed.
- State comparison is bitwise exact.
- Focused worst output error:
  - FP16 normal: 1.22e-4 maximum absolute;
  - BF16 unscaled: 1.95e-3 maximum absolute;
  - adversarial FP16 and padded/short cases: exact in the recorded cases.
- Tolerances: FP16 `rtol=1e-2, atol=1e-2`; BF16
  `rtol=1e-2, atol=1e-1`, matching TensorRT module practice.

## Compile and wall times

- Candidate per-shape cold setup/compile: 0.863–5.027 s.
- Sum of setup/compile over 72 distinct specializations: 152.35 s.
- Candidate full benchmark runner: 153.60 s.
- Candidate full correctness runner: 165.70 s.
- Baseline full benchmark runner: 15.27 s.
- Focused correctness runner: 30.71 s.
- Standalone cached `uv run` quick smoke: 10.69 s runner wall time.
- Cached tile-4 timed sections for 200 calls were 1.81 ms, 3.60 ms, and
  31.69 ms for the three quick points.

Compilation and output allocation are outside the timed kernel region.

## Commands run

From `/node-storage/CuTeDSLGen`:

```bash
# Baseline semantic suite
bash evaluation/decomposition_ab/gpu_run.sh -- bash -lc \
  'docker exec -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" trt-dev bash -lc "
   cd /node-storage/CuTeDSLGen/generated/blackwell_kda_causal_conv1d_prefill &&
   python3 run.py --mode correctness --backends trt_native,trt_triton_head"'

# Candidate focused and full correctness
bash evaluation/decomposition_ab/gpu_run.sh -- bash -lc \
  'docker exec -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" trt-dev bash -lc "
   cd /node-storage/CuTeDSLGen/generated/blackwell_kda_causal_conv1d_prefill &&
   KDA_CUTE_TOKEN_TILE=auto python3 run.py --mode matrix-correctness \
   --matrix full --backends cute"'

# Full candidate and baseline performance matrices
bash evaluation/decomposition_ab/gpu_run.sh -- bash -lc \
  'docker exec -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" trt-dev bash -lc "
   cd /node-storage/CuTeDSLGen/generated/blackwell_kda_causal_conv1d_prefill &&
   KDA_CUTE_TOKEN_TILE=auto python3 run.py --mode benchmark --matrix full \
   --backends cute --warmup 20 --iterations 100"'

bash evaluation/decomposition_ab/gpu_run.sh -- bash -lc \
  'docker exec -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" trt-dev bash -lc "
   cd /node-storage/CuTeDSLGen/generated/blackwell_kda_causal_conv1d_prefill &&
   python3 run.py --mode benchmark --matrix full \
   --backends trt_native,trt_triton_head --warmup 20 --iterations 100"'

# Repository-native standalone smoke
bash evaluation/decomposition_ab/gpu_run.sh -- \
  uv run python generated/blackwell_kda_causal_conv1d_prefill/run.py \
  --mode benchmark --matrix quick --backends cute --warmup 5 --iterations 20
```

## Limitations and profiling readiness

- Long T=8192 sequences are 1.34–1.87x slower than TRT Triton. The CuTe
  straight-line token-tile body likely trades fewer CTAs for register/code
  pressure; this is the first NCU target.
- The native op came from the immutable TensorRT-LLM 1.3.0rc23 container
  extension. Its observed semantics match the inspected checkout source, but
  the container binary was not rebuilt from that exact checkout hash.
- Exact git-`HEAD` Triton padded output is intentionally not a correctness
  baseline because it returns without initializing padded output. Native is
  authoritative there.
- Non-padded cache slots must be unique within a call.
- Dynamic sequence lengths require their CPU mirror at prepare time; omitting
  it causes a setup-only GPU-to-CPU metadata synchronization.
- Specializations key on dtype, W, D, T, B, maximum sequence length, bias,
  activation, and tile. Cold compile is not suitable for an uncached hot path.

**Ready for NCU profiling: yes.** Correctness is gated, launch count is one,
the strongest comparison is recorded, and the long-sequence regressions provide
specific profile targets: register count/occupancy, instruction count, achieved
DRAM/L2 bandwidth, and weight/input load reuse.

## Files added

```text
generated/blackwell_kda_causal_conv1d_prefill/
  README.md
  OPERATION_SPEC.md
  IDEA_LEDGER.md
  JOURNAL.md
  REPORT.md
  common.py
  kernel.py
  run.py
  backends/__init__.py
  backends/cute_direct.py
  backends/trt_native.py
  backends/trt_triton_head.py
  results/*.json
```

No TensorRT-LLM or KDA integration files were edited. No commit was created.
