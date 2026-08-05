# all_reduce

TP=8 allreduce tactic sweep on Kimi K3 decode shapes, following the exact
logic of TRT-LLM's allreduce autotuner: the same `AllReduceRunner` the
`tunable_allreduce` path profiles, the same tactic enumeration
(`get_valid_tactics`: NCCL_SYMMETRIC + NCCL always, ONESHOT while the message
fits the custom workspace, TWOSHOT once `num_tokens >= tp_size`), and the same
op call with the same workspace — but every tactic's latency and bus bandwidth
is reported instead of silently caching one winner.

Motivation: at TP=8 a K3 decoder layer issues two reduces per layer — the
attention `o_proj` (`[tokens, 7168]`) and the fused MoE latent+shared reduce
(`[tokens, 3584 + 7168]`). With the default `AUTO` strategy the runtime autotuner
picks the kernel per shape, and its cache-miss fallback is NCCL_SYMMETRIC —
which is how an NCCL kernel ends up in a decode trace where oneshot should win.
This bench measures what each kernel actually costs at those exact shapes.

## Setup (one-time)

The system Python has no `tensorrt_llm`; the bench uses a dedicated venv:

```bash
cd LLMDiveDeep
uv venv --python 3.12 .venv-trtllm
uv pip install --python .venv-trtllm/bin/python \
    tensorrt_llm==1.2.1 --extra-index-url https://pypi.nvidia.com
uv pip install --python .venv-trtllm/bin/python matplotlib
```

## Run

```bash
cd LLMDiveDeep
mpirun -n 8 --allow-run-as-root \
    .venv-trtllm/bin/python all_reduce/bench_allreduce.py
```

Results land in `all_reduce/results/bench_allreduce_tp8.{csv,png}`. Only rank 0
prints; latency is the max over ranks of the per-rank median.

`autotune_choices.py` (same launch line) runs the REAL autotuner path
(`AllReduce(strategy=AUTO)` inside `autotune()`) on both shapes and dumps the
tuner's cache: the selected tactic per (hidden, num-token bucket). On this box
the tuner picks ONESHOT up to bucket 64 for hidden 7168 but only up to bucket
32 for hidden 10752 — the MoE reduce crosses to NCCL_SYMMETRIC one bucket
earlier because its per-token message is 1.5x larger.

## Results (8x B200, TRT-LLM 1.2.1 wheel, NCCL 2.27.7)

Token sweep 1..2048 on both shapes; median us per call, max over ranks,
CUDA-graph replay timing. Winner by token count (both shapes agree):

| tokens | winner | latency (attn shape) | runner-up |
| --- | --- | --- | --- |
| 1 – 32 | `oneshot` | 4.8 – 8.7 us | nothing close (NCCL ~34 us) |
| 64 | `oneshot` (tie) | 14.3 us | `nccl_symmetric` 16.6 us |
| 128 – 2048 | `nccl_symmetric` | 20 – 83 us | `twoshot` from 256 up |

Takeaways:

- **Decode regime (<= 64 tokens): oneshot, 5–7x faster than NCCL.** These
  messages are handshake-bound; NCCL's ring never leaves its ~35 us floor.
- **Oneshot saturates at ~137 GB/s bus bandwidth** (its P2P-read algorithm
  moves the whole message to every rank), so it falls off past 128 tokens.
  NCCL_SYMMETRIC's NVLS multicast scales to ~620 GB/s and owns the large
  end — 2x over plain NCCL and 1.3–1.5x over twoshot everywhere.
- **TWOSHOT never wins a single point** in this sweep: at small sizes its two
  rounds each pay the handshake cost (128–280 us!), and at large sizes
  NCCL_SYMMETRIC beats it. Notably the static AUTO lookup table picks TWOSHOT
  for 512–4096 tokens at these hiddens, which measures ~1.5x worse than
  NCCL_SYMMETRIC here.
- **Plain NCCL wins nowhere.** An NCCL kernel on a decode-size reduce costs
  ~25–30 us extra per call (~50–60 us per K3 layer pair) versus oneshot,
  purely from tactic selection.

## bench_allreduce_norm.py — allreduce + RMSNorm fusion, TRT vs FlashInfer

K3 decode follows the o_proj allreduce with a residual add + RMSNorm. This
bench compares, per TP size (2/4/8) at hidden 7168 bf16, tokens 1/4/8/16/32:

- **`ar` group** (allreduce only): TRT one-shot vs FlashInfer one-shot
  (`trtllm_allreduce_fusion`, `kAllReduce`) vs NCCL vs two-shot.
- **`ar_norm` group** (allreduce + residual + RMSNorm): the unfused
  two-kernel pipeline (AR then `flashinfer.norm.fused_add_rmsnorm`) vs
  TRT's fused `AllReduceFusionOp.RESIDUAL_RMS_NORM` (ONESHOT and
  MIN_LATENCY) vs FlashInfer's fused `kARResidualRMSNorm`.

Note FlashInfer's comm kernels are a port of TRT-LLM's
`allReduceFusionKernels.cu`, so "TRT vs FlashInfer fused" compares two builds
of the same kernel family; no new kernel is needed to "fuse norm into the TRT
one-shot" — `RESIDUAL_RMS_NORM` already is that kernel.

```bash
cd LLMDiveDeep
export LD_LIBRARY_PATH=$PWD/.venv-trtllm/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
for n in 2 4 8; do
  mpirun -n $n --allow-run-as-root -x LD_LIBRARY_PATH \
      --mca pml ob1 --mca btl self,vader --mca coll ^ucc,hcoll \
      .venv-trtllm/bin/python all_reduce/bench_allreduce_norm.py
done
```

(The MCA flags force shared-memory transports: this pod's UCX/UCC stack
segfaults in MPI_Init trying to bring up a nonexistent IB device, and the
TRT wheel needs the venv's CUDA-13 `libnvrtc` on `LD_LIBRARY_PATH`.)

Results land in `results/bench_allreduce_norm_tp{2,4,8}.{csv,png}`.
Correctness is checked against an exact fp32 reference (gloo CPU sum, then
fp32 residual+RMSNorm); observed max-abs error is bf16 rounding (~0.03–0.12).

Findings (8x B200, TRT-LLM 1.2.1, FlashInfer 0.6.4, PDL on):

- **Fusing the norm always wins**: ~1.0–1.5 us saved vs the two-kernel
  pipeline at every TP and batch (one fewer launch + one fewer round trip
  through the output buffer). At TP8/B=1: 6.31 us fused vs 7.37 us unfused.
- **TP8: TRT wins everywhere** (both AR-only and fused), by 0.05–0.6 us.
- **TP4/TP2: mixed, within ~0.3 us** — FlashInfer's AR-only build is
  slightly faster at TP4, and its fused kernel wins a few mid-batch points
  at TP2; TRT fused wins the rest. Same kernel, different build/launch
  parameters — differences are noise-level.
- `MIN_LATENCY` fused == `ONESHOT` fused (identical dispatch at these
  token counts). Two-shot remains uncompetitive (100–300 us) at decode
  sizes, matching `bench_allreduce.py`.

## Notes

- The tactic list, runner, workspace, and op call are byte-for-byte what the
  autotuner profiles (`AllReduceRunner.get_valid_tactics` /
  `AllReduceRunner.forward`), so these numbers ARE the tuner's search space —
  just fully reported, with `busbw_gbps = 2(n-1)/n * bytes / t` alongside
  latency.
- Timing captures the `iters` collectives into one CUDA graph per tactic
  (cross-rank agreement guard; eager fallback in lockstep), which removes the
  per-launch CPU overhead that otherwise buries the ~5 us oneshot kernel.
- Correctness: all tactics reduce the same per-rank inputs; outputs are
  compared against the plain-NCCL result (bf16 sums; reduction order may
  differ by <= 1 ulp).
