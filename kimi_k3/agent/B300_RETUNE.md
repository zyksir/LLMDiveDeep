# B300/GB300 retune and serving-composition report — TP8, checkpoint width

Date: 2026-08-21. Node: 8x GB300 (sm_103a; nvidia-smi mislabels it "L20D"),
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23` (layer benches) and rc19 +
fork-built C++ @ `165cdc3dcf` (serving). Clocks locked 2032 MHz except where
noted. **All tables here are at the released checkpoint widths**
(`moe_intermediate_size=3072`, `num_shared_experts=2`, per
moonshotai/Kimi-K3 config.json) — every earlier table in
`moe_optimization.md` ran routed experts at the stale 384 width (the shared
product 16x384 == 2x3072 masked the error; shared-expert shapes were always
right).

## 1. MoE layer, TP8 (`bench_moe_layer_tp8_b300_i3072.csv`)

| B | baseline us | opt us | speedup | (at 384 width) |
|---:|---:|---:|---:|---:|
| 1 | 86.25 | 55.54 | +35.6% | +40.6% |
| 8 | 123.67 | 95.86 | +22.5% | +37.4% |
| 32 | 209.13 | 173.36 | +17.1% | +35.3% |
| 64 | 257.40 | 228.58 | +11.2% | +30.0% |
| 128 | 317.45 | 278.61 | +12.2% | +24.1% |
| 256 | 415.48 | 418.29 | **-0.7%** | +11.9% |
| 512 | 479.94 | 409.93 | +14.6% | +16.9% |
| 1024 | 748.48 | 524.36 | +29.9% | +22.0% |
| 2048 | 884.71 | 723.62 | +18.2% | +26.8% |
| 8192 | 2256.37 | 1686.13 | +25.3% | +35.6% |
| 16384 | 4553.71 | 2828.30 | +37.9% | +43.2% |

Decisions taken from this table:

* **`DECODE_MAX_TOKENS` 256 -> 128.** At width 3072 the decode plan loses at
  256 while the prefill plan wins (+2.9%, probe
  `bench_moe_layer_tp8_b300_i3072_pf256.csv`). The Aug-19 extension to 256
  was measured at the stale width.
* **Absolute saved-us hold or grow at prefill** (570 us at 8192, 1725 us at
  16384); decode percentages compress because the expert GEMMs scaled 8x
  while the optimized stages' savings are width-independent.
* Anomaly on record: 8192's tie-excluded error is 1.28e-02, ~2x other sizes
  (more MXFP4 accumulation at 3072?). Not investigated.

## 2. Kernel retunes and reports produced on this node

* **dual-out front tile** (`debug/probe_dualout_config.py`): at 256 rows
  `split_k=2` beats the B200 default `split_k=4` 14.43 vs 18.21 us,
  correctness-gated vs the FP32 oracle (fp32 err 0.92x of default, bitwise
  deterministic). Landed as the arch-keyed `_TUNED[(7168,1984,896,256,
  "sm103a")]`; `_bucket` cap raised 128->256 to match; **layer-level +2.5pp
  at B=256** (pre-boundary-change measurement). The remaining buckets were
  swept too (`dualout_buckets.log`, B=16/32/64/96/128): the B200 default
  wins every one on B300 — 256 rows is the only shape that retunes.
* **routing** (`kernels/bench_routing.py`, now benching the SHIPPED radix):
  radix 3.46-4.43 us, fastest at every size vs trtllm (6.7-8.1) and the
  research Triton kernel (5.6-6.6); oracle-gated.
* **routing region** (`kernels/bench_routing_permutation.py`): radix+
  moe_sort beats trtllm_routing_custom 1.47-2.01x everywhere. Permute is
  the gap: `b10_permute_v1_pdl` (candidate) cuts the region 25-47% at
  B<=256 and loses from ~1024 — a clean DECODE_MAX_TOKENS-shaped dispatch
  if promoted. Route stage is already at its practical bound at decode.
* **comm reports** (`communication/local_result/report_graph_*.csv`):
  all_reduce dim 7168 — trt wins <=32 tokens, b10_multimem 64+;
  allreduce_norm dim 7168 — **trt fused wins <=64 (6.8-16.2 us), plain
  AR+norm ("seq") wins 128+; multimem arnorm measured 39-52 us at every
  size — do not wire it on this arch.**
* **fused col-AG quant** (`bench_col_quant.py`): PASS, 16.3 us vs 73.7 us
  unfused reference.

## 2b. Baseline alignment finding (2026-08-21, late)

Aligning the bench baseline with the checkpoint (`hidden_act="situ"`,
beta 4.0/25.0 — previously the baseline ran SwiGlu experts and was not
measuring feat/k3's path) immediately reproduced the production crash
signature at layer level: `No kernel found ... mEltwiseActType: 2` — the
DECODE path's `_experts()` call pinned `ExpertBackend.NATIVE` as a
hardcoded default that `capabilities.filter()` never touched (the filter
only rewrites the prefill axis). Fixed in both repos: decode experts now
follow `capabilities.native_experts`, falling to FlashInfer on sm_103 +
SiTU exactly like prefill. Every situ-baseline number below this line is
measured post-fix.

## 3. Serving composition (fork `kimi_k3_optim`, 93-layer, 100k prompts)

Layer wins did NOT compose into serving until three integration gaps were
closed (torch traces via `TLLM_TORCH_PROFILE_TRACE`): decode hard-disabled
in the adapter; `_serving_auto_map` missing decode-dim AR rows (pick() fell
to torch_symm:2shot, +4.3 ms per 92-layer step); unfused col-AG and
allreduce_norm (the fused engines existed only in this repo). With decode
enabled + AR seeds: decode-isolated -28.9% -> -17.7%; col-AG + trt-arnorm
port in flight. Prefill composed once enabled: **-7% TTFT at 100k, conc
8/16, reproduced 3x**. Until the decode port fully lands, ship
`TRTLLM_KIMI_B10_MOE=1 KIMI_B10_DECODE_MAX_BATCH=0`.

## 4. Reproduction commands

```bash
mpirun -n 8 --allow-run-as-root python3 kimi_k3_layer/bench_b10_kimi_k3_moe_layer.py --sizes all --iters 100 --n-inputs 8
python3 kimi_k3_layer/kernels/bench_routing.py
python3 kimi_k3_layer/kernels/bench_routing_permutation.py --include-candidates
mpirun -n 8 --allow-run-as-root python3 communication/bench_comm_graph.py --ops allreduce,allreduce-norm --bs 1..16k --dims 7168
mpirun -n 8 --allow-run-as-root python3 communication/kernel_benchmarks/bench_col_quant.py
```
