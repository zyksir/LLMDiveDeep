#!/bin/bash
# Combo experiments for the B=32..80 dead zone, following the
# run_retune_b80.sh single-flag findings (Aug-5, this node):
#   fc1 shard ALWAYS beats the <=16 gate at 16..64 (90.6 vs 98.9 @32,
#   118 vs 132 @64); routing "ours" and the ref tail turn into
#   losses at B>=64. These runs flip 2+ flags at once (B10_OPT_FLAGS)
#   to find the best large-B config. Log: /tmp/combo_b80.log
set -u
CONTAINER=${CONTAINER:-trt-dev}
CPATH=${CPATH:-/workspace/diffusion_inference/LLMDiveDeep}
LOG=${LOG:-/tmp/combo_b80.log}

echo "[combo] started $(date)" > "$LOG"
maxclk=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
nvidia-smi -lgc "$maxclk,$maxclk" >> "$LOG" 2>&1

run() {
    echo "=== $1 ($(date)) ===" >> "$LOG"
    docker exec "$CONTAINER" bash -c \
        "cd $CPATH && BENCH_ALLREDUCE_STRATEGY=ONESHOT $2 timeout 3600 \
         mpirun -x BENCH_ALLREDUCE_STRATEGY -x B10_FC1_SHARD_MAX_TOKENS \
           -x B10_REF_TAIL_MIN_TOKENS -x B10_OPT_FLAGS \
           -n 8 --allow-run-as-root \
           python3 kimi_k3_layer/bench_moe_kimi_k3.py $3 2>&1" \
        | grep -E "^\s+[0-9]+\s|configs:|Error|error|Traceback" >> "$LOG" 2>&1
    echo "=== end: $1 ($(date)) ===" >> "$LOG"
}

FC1ON="B10_FC1_SHARD_MAX_TOKENS=1000000000"

run "E1 fc1 ALWAYS, rest default (small-B regression check)" \
    "$FC1ON" "--sizes 1,2,4,8,16,32,64,80"

run "E2 fc1 ALWAYS + ref tail NEVER" \
    "$FC1ON B10_REF_TAIL_MIN_TOKENS=1000000000" "--sizes 16,32,64,80"

run "E3 fc1 ALWAYS + routing noaux" \
    "$FC1ON B10_OPT_FLAGS=routing=noaux" "--sizes 32,64,80"

run "E4 fc1 ALWAYS + routing noaux + ref tail NEVER" \
    "$FC1ON B10_REF_TAIL_MIN_TOKENS=1000000000 B10_OPT_FLAGS=routing=noaux" \
    "--sizes 32,64,80"

run "E5 fc1 ALWAYS + merged off (split front at 32+)" \
    "$FC1ON B10_OPT_FLAGS=merged_gemm=False" "--sizes 32,64,80"

nvidia-smi -rgc >> "$LOG" 2>&1
echo "[combo] ALL DONE $(date)" >> "$LOG"
