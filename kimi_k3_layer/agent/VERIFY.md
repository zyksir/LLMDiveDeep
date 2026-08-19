# Verifying the Kimi-K3 MoE results (TP-8, 8×B200, trt-dev container)

All commands run on an **idle node** (check `nvidia-smi` first — collectives are
extremely sensitive to co-running jobs) from the host:

```bash
docker exec trt-dev bash -c "cd /workspace/diffusion_inference/LLMDiveDeep && <CMD>"
```

Every run prints a per-size correctness column (`err(...)`, max-abs vs the TRT
baseline forward on identical weights; bf16 noise is ≤ ~1.3e-2) next to the
latencies — a run is only valid if the err column stays at bf16 level.

## 1. Decode (graph-timed, production regime)

```bash
mpirun -n 8 --allow-run-as-root python3 kimi_k3_layer/bench_moe_kimi_k3.py \
    --sizes small
```

`--sizes small` = B ∈ {1,2,4,8,16,32,64,128,256}; the opt column is the
unified B10 class (decode `_forward_opt`), CUDA-graph replay timing, baseline =
stock TRT-LLM `KimiK3MoE.forward` captured the same way.

## 2. Prefill (eager = production prefill regime)

```bash
# Shipped SP tail vs baseline:
mpirun -n 8 --allow-run-as-root python3 kimi_k3_layer/bench_moe_kimi_k3.py \
    --sizes large

# Report-only one-axis comparison, including SP vs AR+AR:
python3 kimi_k3_layer/search_best_strategy.py --world 8 --sizes 512,1024,4096,8192
```

`--sizes large` = S ∈ {512, 1024, 2048, 4096, 8192} tokens. Prefill is timed
EAGER by default (host cost included — production prefill runs no graph).

## 3. Prefill, pure-GPU comparison (CUDA-graph, host removed)

```bash
B10_PREFILL_GRAPH=1 mpirun -x B10_PREFILL_GRAPH -n 8 \
    --allow-run-as-root python3 kimi_k3_layer/bench_moe_kimi_k3.py \
    --sizes large
```

The graphed err column doubles as the CE-collectives-under-replay correctness
check (they are graph-safe).

## Experiment controls

The layer's shipped defaults are ordinary `set_opt_flags` defaults, not
environment-driven runtime policy. `bench_moe_kimi_k3.py` accepts a complete
`B10_OPT_FLAGS` override for verification; `search_best_strategy.py` constructs
those complete candidates automatically. `B10_PREFILL_GRAPH=1` remains a
benchmark timing control, and `B10_CE_CPP=0` selects the communication
reference implementation.
| `B10_DIAG` | 0 | bench-only: per-token/channel error breakdown + baseline self-determinism probe |
