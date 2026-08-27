# Blackwell KDA causal-conv1d prefill

Standalone CuTe DSL 4.5.2 package for packed-variable-length KDA prefill
depthwise causal convolution. It consumes token-major `[T_total,D]` FP16/BF16
input directly, supports `W in {2,3,4}`, optional channel bias and SiLU/Swish,
writes contiguous `[T,D]`, and updates slot-indexed `[slot,D,W-1]` state in the
same GPU launch.

The CUDA runtime used for the recorded results reported a device named
`NVIDIA L20D` with capability 10.3 and 148 SMs. Host `nvidia-smi` reported the
same name but capability 8.9. This report does not label the device B200.

## Files

- `kernel.py`: standalone cached wrapper.
- `backends/cute_direct.py`: CuTe DSL kernel and backend adapter.
- `backends/trt_native.py`: unchanged native transpose+CUDA baseline body.
- `backends/trt_triton_head.py`: exact SHA-checked git-`HEAD` Triton baseline.
- `common.py`: shared inputs, matrix, correctness, timing, and JSON.
- `run.py`: backend-neutral CLI.
- `OPERATION_SPEC.md`: exact operation/state/padding contract.
- `IDEA_LEDGER.md`: template survey, hypotheses, and decision gate.
- `JOURNAL.md`: chronological trial record.
- `REPORT.md`: final correctness/performance report.
- `results/`: durable structured receipts.

## Standalone API

```python
from kernel import CausalConv1dPrefill

runner = CausalConv1dPrefill()
prepared = runner.prepare(
    projected,
    num_prefill_tokens,
    weight,
    conv_states=conv_states,
    query_start_loc=query_start_loc,
    cache_indices=cache_indices,
    has_initial_state=has_initial_state,
    bias=bias,
    activation="silu",
    sequence_lengths=sequence_lengths,  # CPU sequence lengths; optional
)

# One precompiled GPU launch. Output allocation and compilation already happened.
output = prepared.run()
```

If `sequence_lengths` is omitted, `prepare` derives them by copying
`query_start_loc` to CPU. That synchronization is setup work and must remain
outside timing.

Live non-padded cache slots must be unique within one call. `PAD_SLOT_ID=-1`
copies that sequence's input to output unchanged and does not modify state.

## Commands

Standalone CuTe smoke/benchmark using the CuTeDSLGen environment:

```bash
cd /node-storage/CuTeDSLGen
bash evaluation/decomposition_ab/gpu_run.sh -- \
  uv run python /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill/run.py \
  --mode benchmark --matrix quick --backends cute
```

Full candidate matrix:

```bash
cd /node-storage/CuTeDSLGen
bash evaluation/decomposition_ab/gpu_run.sh -- \
  uv run python /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill/run.py \
  --mode benchmark --matrix full --backends cute \
  --output /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill/results/cute_full.json
```

Direct native correctness requires a TensorRT-LLM environment with the
`trtllm` torch extension. The recorded command was:

```bash
docker exec trt-dev bash -lc \
  'cd /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill &&
   KDA_CUTE_TOKEN_TILE=auto python3 run.py --mode matrix-correctness \
   --matrix full --backends cute'
```

## Result

- Full matrix correctness: 72/72 against unchanged native output/state.
- Focused semantic cases: 11/11, including short and padded sequences.
- Candidate latency: 0.00782–0.09283 ms across the 72-shape matrix.
- Versus native pipeline: wins 72/72, 3.43x geometric-mean speedup.
- Versus one-launch git-`HEAD` Triton: wins 52/72, 1.88x geometric-mean
  speedup; all four main BF16 W4 `T=8192` shapes remain slower.

The package is ready for Nsight Compute profiling, especially on the long
sequence shapes where TRT Triton remains 1.34–1.87x faster.
