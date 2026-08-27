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
- `local_results/`: durable structured receipts.

## Standalone API (TRT integration-surface compatible)

`kernel.py` is call-compatible with the TRT-LLM stable integration surface
`tensorrt_llm/_torch/modules/mamba/causal_conv1d_prefill.py`: same argument
names/order, same validation errors, same grouped output mode. A caller can
swap the Triton `causal_conv1d_prefill` for the CuTe one without edits.

```python
from kernel import CausalConv1dPrefill

runner = CausalConv1dPrefill()
prepared = runner.prepare(
    projected,                       # [T, D]; row-strided views supported
    num_prefill_tokens,
    weight,
    conv_states=conv_states,
    query_start_loc=query_start_loc,
    seq_lens_cpu=seq_lens_cpu,       # CPU sequence lengths (required)
    cache_indices=cache_indices,     # optional; defaults to slot=sequence idx
    has_initial_state=has_initial_state,  # optional; defaults to all-False
    bias=bias,
    activation="silu",
    qkv_group_size=None,             # e.g. 1536 for grouped [D//G, T, G] output
    qkv_group_tokens=None,           # plane rows; >= num_prefill_tokens
)

# One precompiled GPU launch. Output allocation and compilation already happened.
output = prepared.run()
```

`sequence_lengths` remains accepted as a deprecated alias of `seq_lens_cpu`.

Grouped mode (`qkv_group_size=G`) returns `[D // G, group_tokens, G]`
contiguous — token-major within each channel-group plane, groups outermost —
with only the leading `num_prefill_tokens` rows of each plane written
(`qkv_group_tokens > num_prefill_tokens` leaves the tail rows uninitialized
for mixed-batch splicing). Deviation from the Triton surface: the CuTe
constraint is `G % 512 == 0` (CTA channel span = 128 threads x vector width
4), stricter than Triton's multiple-of-256; the K3 production group
`G = 1536 = 3 x 512` satisfies both. Grouped mode runs on the streaming W4
kernel only.

Live non-padded cache slots must be unique within one call. `PAD_SLOT_ID=-1`
copies that sequence's input to output unchanged and does not modify state —
a compatible superset of the TRT contract, which leaves padded rows
uninitialized and forbids reading them.

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
  --output /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill/local_results/cute_full.json
```

Direct native correctness requires a TensorRT-LLM environment with the
`trtllm` torch extension. The recorded command was:

```bash
docker exec trt-dev bash -lc \
  'cd /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill &&
   KDA_CUTE_TOKEN_TILE=auto python3 run.py --mode matrix-correctness \
   --matrix full --backends cute'
```

## Acceptance gate (round-3 framing)

This is an unintegrated research candidate. Per the round-3 user acceptance
update, **one single kernel configuration** (no per-regime dispatcher) must:

1. beat unchanged exact git-`HEAD` TRT Triton with a repeatable positive
   margin on every BF16/W4 combination of `D={1536,3072}`,
   `T={128,1024,8192}`, `B={1,uneven 8}` **and** on the Kimi-K3 production
   strided shapes (`D=4608`, row-strided view with `stride(0)=4752`, no bias,
   same T/B grid) — 18 gate shapes total (hard gate);
2. report achieved SOL fractions against the calibrated attainable roofs plus
   plateau evidence (best-effort gate; the previous hard 80% requirement was
   softened by the user).

Precision rules are binding: FP32 accumulation in every variant, documented
SiLU-form deltas, no silent downgrades. FP16/BF16 W2/W3/W4 full-matrix
correctness (dense 72 + strided production) with bitwise-exact states is
independently mandatory. Results within 5% require repeated rounds and
dispersion evidence.

## Round-3 result (final)

Selected single configuration — the backend default with no env vars set:
stream algorithm (W4), vector width 4, 128 threads/CTA, token tile 16,
group-batched loads with span 4, FP32 ring, tanh SiLU, BF16 parameter
fragments (bitwise-identical to FP32 fragments), FP32 accumulation. W2/W3
correctness shapes use the direct kernel (stream is W4-only; no gate shape is
W2/W3).

- **TRT Triton gate: PASS 18/18**, five-round repeatable speedups
  **1.774–3.949x** (worst margin 77%; no confidence rerun triggered).
- Correctness: case suite 17/17 and full matrix 108/108 rows against
  unchanged native; conv states bitwise exact.
- SOL vs calibrated roofs: 52.0–52.1% dense D3072 long, 59.7–60.7% dense
  D1536 long, 64.7–64.9% strided production long; short/medium 31–40% of the
  strict launch-floor bound. Plateau evidence (no saturated pipe; DRAM
  latency under register-bound occupancy) is documented in `REPORT.md`.
- Final NCU (selected config): dense long 24.5–25.2 µs at 47.6–48.1% compute
  / 31.0–31.8% DRAM SOL; strided long 33.9–34.4 µs at 53.4–54.4% compute /
  41.8–42.6% DRAM SOL. Unchanged Triton on strided long: 56.8–67.5 µs,
  80.7/73.9% compute-bound.
- Canonical artifacts: `local_results/r3_acceptance_summary.json`,
  `local_results/r3_selected_main_confidence.json`,
  `local_results/r3_selected_full_correctness.json`,
  `local_results/r3_silu_precision_audit.json`,
  `local_results/r3_roof_calibration.json`, `local_results/ncu/ncu_sol_report.json`.

Round-2 history (superseded): the per-regime dispatcher passed the dense TRT
gate 12/12 (1.125–3.466x) but was REJECTED on the then-hard 80% SOL gate at
25.6–35.5% analytical efficiency; artifacts remain under `local_results/r2_*`.

## TRT integration-surface alignment

The wrapper matches the redesigned TRT-LLM surface (see the Standalone API
section). Validation receipt `local_results/r3_trt_surface_alignment.json`:

- Signature parity verified programmatically against the TRT module (exact
  prefix match; only the deprecated `sequence_lengths` alias is appended).
- Grouped output bit-identical to the flat output regrouped via
  `view(T, D//G, G).permute(1, 0, 2)` on 6/6 cases (dense D3072 and strided
  production D4608, T in {128, 8192}, B in {1, 8}, mixed initial states,
  including a `qkv_group_tokens > T` tail-rows case); conv states bitwise
  identical between flat and grouped runs.
- Focused correctness suite re-passed after the store-path change
  (`local_results/r3_surface_case_correctness.json`).
- Grouped-store cost on the selected config: +0.5% dense T8192 B8
  (0.02803 -> 0.02819 ms, within noise) and +6.5% strided T8192 B8
  (0.03987 -> 0.04247 ms); the flat path is unchanged.
