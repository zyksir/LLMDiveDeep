#!/bin/bash
# Baseline honesty sweep: AllReduce strategy x op backend, B=2,8,16.
# Only the base_us column matters; opt runs with default flags.
S=/tmp/claude-0/-workspace-model-performance-zyksir-llm-inference/ab452c80-d7d1-4957-a7a7-9339c411937b/scratchpad
cd /workspace/model-performance/zyksir/llm_inference/LLMDiveDeep
for strat in ONESHOT TWOSHOT MIN_LATENCY NCCL; do
  echo "=== AR strategy $strat ==="
  BENCH_ALLREDUCE_STRATEGY=$strat B10_REF_TAIL_MIN_TOKENS=1000000000 \
  mpirun -x BENCH_ALLREDUCE_STRATEGY -x B10_REF_TAIL_MIN_TOKENS -n 8 --allow-run-as-root \
    python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes 2,8,16 2>&1 \
    | grep -E "^ +[0-9]+ +[0-9]+\." | sed "s/^/[$strat] /"
done
echo "=== native op backend (BENCH_MOE_OP_BACKEND=trtllm) ==="
BENCH_MOE_OP_BACKEND=trtllm B10_REF_TAIL_MIN_TOKENS=1000000000 \
mpirun -x BENCH_MOE_OP_BACKEND -x B10_REF_TAIL_MIN_TOKENS -n 8 --allow-run-as-root \
  python3 kimi_k3_layer/bench_moe_kimi_k3.py --sizes 2,8,16 2>&1 \
  | grep -E "^ +[0-9]+ +[0-9]+\." | sed "s/^/[nativeop] /"
echo "SWEEP DONE"
