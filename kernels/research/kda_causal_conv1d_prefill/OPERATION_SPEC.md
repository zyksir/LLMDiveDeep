# Packed-varlen KDA causal-conv1d prefill specification

## Contract

The package computes one depthwise causal convolution over packed variable-length
prefill sequences:

```text
history[b, t, c] =
  initial_state[cache_indices[b], c, t]              when t < 0 and has_initial_state[b]
  0                                                  when t < 0 otherwise
  projected[query_start_loc[b] + t, c]               when 0 <= t < sequence_length[b]

acc[b, t, c] = bias[c] + sum(k=0..W-1) weight[c, k]
                                      * history[b, t - (W - 1) + k, c]
output[b, t, c] = silu(acc[b, t, c]) or acc[b, t, c]
```

- `projected`: token-major row-major `[T_total, D]`, FP16 or BF16, channel
  stride one. Only `projected[:T]` is consumed.
- `weight`: contiguous `[D, W]`, same dtype as `projected`, `W in {2, 3, 4}`.
- `bias`: optional contiguous `[D]`, same dtype.
- `query_start_loc`: CUDA int32 `[B + 1]`, starts at zero, ends at `T`.
- `cache_indices`: CUDA int32 `[B]`; slots may be permuted.
- `has_initial_state`: CUDA bool `[B]`.
- `conv_states`: mutable `[num_slots, D, W - 1]`, same dtype. Channel storage
  need not be contiguous because the contracted layout has state-element stride
  one.
- Output: newly allocated or caller-provided contiguous `[T, D]`, same dtype.
- Accumulation and SiLU input are FP32. SiLU is `x / (1 + exp(-x))`; `None`,
  `"silu"`, and `"swish"` are accepted.
- Exactly one candidate GPU-kernel launch. Output allocation, compilation, and
  DLPack/FFI setup occur outside the timed kernel region.

## State and padding semantics

After a non-padded sequence, the selected slot holds its final `W - 1` input
tokens. If the sequence is shorter than `W - 1`, the old state is shifted left
and the sequence is appended when `has_initial_state[b]` is true; otherwise the
leading state elements become zero. This matches the unchanged TensorRT-LLM
native kernel.

`PAD_SLOT_ID = -1` means:

1. copy that sequence's input rows to output unchanged, even if bias or SiLU is
   requested;
2. do not read or modify any convolution-state slot.

The package requires non-overlapping live cache slots within one invocation.
Two non-padded sequences aliasing the same slot would introduce an unspecified
cross-CTA state race in both the native and proposed direct kernels.

## Required matrix

- dtype: FP16, BF16
- width: 2, 3, 4
- channels: 1536, 3072
- packed tokens: 128, 1024, 8192
- batch: one sequence and uneven eight-sequence packing
- state patterns: no initial state, mixed initial state, permuted slots, short
  sequences, padded slots
- primary performance slice: BF16, width 4

## Correctness tolerance

The unchanged native CUDA implementation is authoritative. State must be
bitwise equal because it only copies inputs/state. Output is checked with the
TensorRT module tolerance:

- FP16: `rtol=1e-2`, `atol=1e-2`
- BF16: `rtol=1e-2`, `atol=1e-1`

The harness additionally records maximum absolute and relative error.

## Obligations used for timing

The native pipeline includes its row-major-to-channel-major transpose kernel,
the unchanged in-place native causal-conv kernel, and the final materialization
to contiguous token-major output. The native body is isolated in its backend
module and is not rewritten.

The TRT Triton comparison uses the git-`HEAD` source, not the locally modified
working-tree file. Its original padded-slot behavior leaves padded output
uninitialized, so performance comparison excludes padded cases; padded
correctness remains gated against native semantics. Any adapter or conversion
needed by a backend is charged.

The CuTe backend accepts row-major input directly and writes row-major output.
Its precompiled callable and DLPack wrappers are prepared before timing; timing
contains one launch and device execution.

## Hardware observed for this run

The GPU identity surfaces disagree and both are retained:

- host `nvidia-smi`: eight devices named `NVIDIA L20D`, reported capability
  8.9;
- CUDA runtime inside the TensorRT/CuTe workload container: `NVIDIA L20D`,
  capability 10.3, 148 SMs, 287,428,771,840 bytes.

The executable CUDA runtime therefore confirms an SM103 target, but the product
name is not B200 and this package does not claim B200 measurements. Benchmarks
use the repository GPU lease helper with the SM clock locked to 1500 MHz.
