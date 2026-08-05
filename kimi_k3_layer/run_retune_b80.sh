#!/bin/bash
# B=1..80 re-tune (RUNBOOK SS1-3 + fc1/qpush focus). Run ON THE HOST
# (uses docker exec + nvidia-smi). Waits for ALL GPUs idle 120s, locks
# clocks, then runs the decision set. Log: /tmp/retune_b80.log
#
#   nohup bash kimi_k3_layer/run_retune_b80.sh >/dev/null 2>&1 &
#
# Decisions this produces (read the log + results/):
#   step 1  qpush vs receiver-side AG+quant per B (+ bit-exactness)
#           -> "when to quantize-then-allgather"
#   step 2  GEMM-vs-batch sweep -> where each GEMM stops being flat
#   step 3  MAIN ablation B=1..80 -> every crossover
#           (B10_REF_TAIL/FC2_SHARD/FC1_SHARD/OVERLAP/ROUTED_SPLIT)
#   step 4  fc1 shard ALWAYS vs NEVER at 16..80 (qpush+grid64 in the
#           AG autotune) -> can fc1 stay sharded with no regression?
#   step 5  ref-tail extremes (ALWAYS/NEVER) -> min-tokens confirm
#   step 6  tail probes (standalone, B=1..80)
#   step 7  deploy classes: disagg + agg (+agg with fc2 weight AG)
#   step 8  traces base+opt at 1,8,16,32,64,80 (separate run - CUPTI
#           taxes kernels; NEVER take headline numbers from it)
#   step 9  final ablation -> regenerates
#           results/bench_moe_kimi_k3_tp8.{csv,md,png} (headline)
set -u
CONTAINER=${CONTAINER:-trt-dev}
CPATH=${CPATH:-/workspace/diffusion_inference/LLMDiveDeep}
LOG=${LOG:-/tmp/retune_b80.log}
SIZES="1,2,4,8,16,32,64,80"

echo "[retune] started $(date)" > "$LOG"

idle=0
while [ "$idle" -lt 24 ]; do  # 24 x 5s = 120s sustained idle
    busy=$(nvidia-smi --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits | sort -rn | head -1)
    mem=$(nvidia-smi --query-gpu=memory.used \
        --format=csv,noheader,nounits | sort -rn | head -1)
    if [ "$busy" -le 2 ] && [ "$mem" -le 2000 ]; then
        idle=$((idle + 1))
    else
        idle=0
    fi
    sleep 5
done
maxclk=$(nvidia-smi --query-gpu=clocks.max.sm \
    --format=csv,noheader,nounits | head -1)
nvidia-smi -pm 1 >> "$LOG" 2>&1
nvidia-smi -lgc "$maxclk,$maxclk" >> "$LOG" 2>&1
echo "[retune] GPUs idle, clocks locked at $maxclk; starting $(date)" >> "$LOG"

run() {  # $1 label, $2 env prefix, $3 command
    echo "=== $1 ($(date)) ===" >> "$LOG"
    docker exec "$CONTAINER" bash -c \
        "cd $CPATH && $2 timeout 5400 $3 2>&1" \
        | grep -vE "UCP worker|^\[TRT-LLM|NVLS|^\*|Detecting|intra-node|Warning|warn" \
        >> "$LOG" 2>&1
    echo "=== end: $1, exit=$? ($(date)) ===" >> "$LOG"
}

ENVX="-x BENCH_ALLREDUCE_STRATEGY -x B10_REF_TAIL_MIN_TOKENS -x B10_FC1_SHARD_MAX_TOKENS"
MPI="mpirun $ENVX -n 8 --allow-run-as-root"
ONESHOT="BENCH_ALLREDUCE_STRATEGY=ONESHOT"

run "1 AG duel: qpush vs recv-side (bit-exactness + timing)" "" \
    "$MPI python3 kimi_k3_layer/tmp_fc1_ag_duel.py"

run "2 GEMM-vs-batch sweep" "CUDA_VISIBLE_DEVICES=0" \
    "python3 debug/gemm_bs_sweep.py"

run "3 MAIN ablation B=1..80 (ONESHOT baseline)" "$ONESHOT" \
    "$MPI python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes $SIZES --ablate"

run "4a fc1 shard ALWAYS (gate=10^9)" \
    "$ONESHOT B10_FC1_SHARD_MAX_TOKENS=1000000000" \
    "$MPI python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes 16,32,64,80"
run "4b fc1 shard NEVER (gate=0)" \
    "$ONESHOT B10_FC1_SHARD_MAX_TOKENS=0" \
    "$MPI python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes 16,32,64,80"

run "5a ref tail ALWAYS (min_tokens=1)" \
    "$ONESHOT B10_REF_TAIL_MIN_TOKENS=1" \
    "$MPI python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes 4,8,16,32,64,80"
run "5b ref tail NEVER (min_tokens=10^9)" \
    "$ONESHOT B10_REF_TAIL_MIN_TOKENS=1000000000" \
    "$MPI python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes 4,8,16,32,64,80"

run "6a fair tail-crossover probe" "" \
    "$MPI python3 kimi_k3_layer/tmp_tail_ref_overlap.py"
run "6b tail shard strategy probe" "" \
    "$MPI python3 kimi_k3_layer/tmp_tail_shard_probe.py"

run "7a deploy: disagg decode" "$ONESHOT" \
    "$MPI python3 kimi_k3_layer/bench_moe_kimi_k3.py --moe-class disagg --sizes $SIZES"
run "7b deploy: agg decode+prefill" "$ONESHOT" \
    "$MPI python3 kimi_k3_layer/bench_moe_kimi_k3.py --moe-class agg --sizes 1,8,16,32,64,80,512,1024,4096,8192"
run "7c deploy: agg + one-time fc2 weight AG" \
    "$ONESHOT B10_AGG_FULL_FC2=1" \
    "$MPI -x B10_AGG_FULL_FC2 python3 kimi_k3_layer/bench_moe_kimi_k3.py --moe-class agg --sizes 16,32,64,80"

run "8 traces base+opt (CUPTI-taxed; numbers not headline)" "$ONESHOT" \
    "$MPI python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes 1,8,16,32,64,80 --profile 1,8,16,32,64,80"

run "9 FINAL ablation (regenerates results/bench_moe_kimi_k3_tp8.*)" \
    "$ONESHOT" \
    "$MPI python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes $SIZES --ablate"

nvidia-smi -rgc >> "$LOG" 2>&1  # unlock clocks
echo "[retune] ALL DONE $(date)" >> "$LOG"
