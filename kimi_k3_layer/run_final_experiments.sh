#!/bin/bash
# Final validation set for the Kimi-K3 MoE optimizations. Run INSIDE
# the trt-dev container (nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23)
# from the LLMDiveDeep repo root, on an idle 8-GPU B200 node:
#
#   bash kimi_k3_layer/run_final_experiments.sh 2>&1 | tee /tmp/final.log
#
# IMPORTANT: results are graph-replay max-over-ranks; ANY other job on
# ANY of the 8 GPUs inflates them. Check `nvidia-smi` first.
#
# What it produces (see README "Pending validation" for how to read):
#   1. fair tail-crossover probe (validates REF_TAIL_MIN_TOKENS=4)
#   2. tail sharding strategy probe (col vs row vs REF, standalone)
#   3. e2e crossover extremes: ref tail ALWAYS vs NEVER
#   4. base-vs-opt traces at B=1,2,4,8,16 -> results/*.trace.json
#   5. full ablation B=1..64 LAST -> regenerates
#      results/bench_moe_kimi_k3_tp8.{csv,md,png} from newest code
set -e
cd "$(dirname "$0")/.."

MPI="mpirun -x B10_REF_TAIL_MIN_TOKENS -n 8 --allow-run-as-root"

echo "=== 1. fair tail-crossover probe ==="
B10_REF_TAIL_MIN_TOKENS=4 $MPI python3 kimi_k3_layer/tmp_tail_ref_overlap.py

echo "=== 2. tail sharding strategy probe ==="
B10_REF_TAIL_MIN_TOKENS=4 $MPI python3 kimi_k3_layer/tmp_tail_shard_probe.py

echo "=== 3a. e2e ref tail ALWAYS (min_tokens=1) ==="
B10_REF_TAIL_MIN_TOKENS=1 $MPI python3 \
    kimi_k3_layer/bench_moe_kimi_k3.py --sizes 1,2,4,8,16,32,64

echo "=== 3b. e2e ref tail NEVER (min_tokens=10^9) ==="
B10_REF_TAIL_MIN_TOKENS=1000000000 $MPI python3 \
    kimi_k3_layer/bench_moe_kimi_k3.py --sizes 1,2,4,8,16,32,64

echo "=== 4. base-vs-opt traces B=1,2,4,8,16 ==="
B10_REF_TAIL_MIN_TOKENS=4 $MPI python3 \
    kimi_k3_layer/bench_moe_kimi_k3.py --sizes 1,2,4,8,16 \
    --profile 1,2,4,8,16

echo "=== 5. full ablation (regenerates results/bench_moe_kimi_k3_tp8.*) ==="
B10_REF_TAIL_MIN_TOKENS=4 $MPI python3 \
    kimi_k3_layer/bench_moe_kimi_k3.py --sizes 1,2,4,8,16,32,64 --ablate

echo "ALL DONE"
