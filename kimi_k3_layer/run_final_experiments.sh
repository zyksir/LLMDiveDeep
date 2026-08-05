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
# Size focus: B = 1,2,4,8,16,32,64,80 - especially 32 and 64. At those
# sizes messages leave the latency-bound regime, so runs 4a/4b
# re-decide the fc1-shard gate (currently 16) with the sender-side-
# quantized AG (fp8 wire, half bytes) now inside the AG autotune.
#
# What it produces (see README "Status" for how to read):
#   1. fair tail-crossover probe  (validates REF_TAIL_MIN_TOKENS=4)
#   2. tail sharding strategy probe (col vs row vs REF, standalone)
#   3a/3b. e2e crossover extremes: ref tail ALWAYS vs NEVER
#   4a/4b. fc1 shard forced ALWAYS vs NEVER at large B (qpush enabled)
#   5. base-vs-opt traces at B=1,8,16,32,64 -> results/*.trace.json
#   6. full ablation B=1..80 LAST -> regenerates
#      results/bench_moe_kimi_k3_tp8.{csv,md,png} from newest code
set -e
cd "$(dirname "$0")/.."

ENVV="-x B10_REF_TAIL_MIN_TOKENS -x B10_FC1_SHARD_MAX_TOKENS"
MPI="mpirun $ENVV -n 8 --allow-run-as-root"
SIZES="1,2,4,8,16,32,64,80"

echo "=== 1. fair tail-crossover probe ==="
B10_REF_TAIL_MIN_TOKENS=4 $MPI python3 kimi_k3_layer/tmp_tail_ref_overlap.py

echo "=== 2. tail sharding strategy probe ==="
B10_REF_TAIL_MIN_TOKENS=4 $MPI python3 kimi_k3_layer/tmp_tail_shard_probe.py

echo "=== 3a. e2e ref tail ALWAYS (min_tokens=1) ==="
B10_REF_TAIL_MIN_TOKENS=1 $MPI python3 \
    kimi_k3_layer/bench_moe_kimi_k3.py --sizes $SIZES

echo "=== 3b. e2e ref tail NEVER (min_tokens=10^9) ==="
B10_REF_TAIL_MIN_TOKENS=1000000000 $MPI python3 \
    kimi_k3_layer/bench_moe_kimi_k3.py --sizes $SIZES

echo "=== 4a. fc1 shard ALWAYS (gate=10^9; AG autotunes bf16 vs qpush) ==="
B10_REF_TAIL_MIN_TOKENS=4 B10_FC1_SHARD_MAX_TOKENS=1000000000 $MPI python3 \
    kimi_k3_layer/bench_moe_kimi_k3.py --sizes 16,32,64,80

echo "=== 4b. fc1 shard NEVER (gate=0) ==="
B10_REF_TAIL_MIN_TOKENS=4 B10_FC1_SHARD_MAX_TOKENS=0 $MPI python3 \
    kimi_k3_layer/bench_moe_kimi_k3.py --sizes 16,32,64,80

echo "=== 5. base-vs-opt traces B=1,8,16,32,64 ==="
B10_REF_TAIL_MIN_TOKENS=4 $MPI python3 \
    kimi_k3_layer/bench_moe_kimi_k3.py --sizes 1,8,16,32,64 \
    --profile 1,8,16,32,64

echo "=== 6. full ablation (regenerates results/bench_moe_kimi_k3_tp8.*) ==="
B10_REF_TAIL_MIN_TOKENS=4 $MPI python3 \
    kimi_k3_layer/bench_moe_kimi_k3.py --sizes $SIZES --ablate

echo "ALL DONE"
