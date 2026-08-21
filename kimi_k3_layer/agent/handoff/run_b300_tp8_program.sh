#!/bin/bash
# The complete B300/TP8 measurement program. Single owner; self-gating.
# Waits until every GPU is genuinely free (neighbors included), locks clocks,
# then runs, in order:
#   1. TP8 contribution table (situ baseline, bs 1..64)   <- headline table
#   2. TP8 full layer table (--sizes all)
#   3. Strategy-crossover sweeps: decode_tail @16..128, decode_front @32,64
#   4. Profiled pairs at B=8 and B=64 (baseline+new traces) for the
#      unnecessary-copy / slow-kernel-vs-SOL review
#   5. TP4 ablation bs 1..64 (kernels must stay TP4-capable)
# Every stage logs under /node-storage/var and fails loudly; later stages
# still run so one failure does not empty the night.
set -u
VAR=/node-storage/var
LDD=/node-storage/LLMDiveDeep
run() {  # run <log> <cmd...>
  local log=$1; shift
  echo "=== $(date +%H:%M:%S) $log ==="
  docker exec -w $LDD trt-k3 bash -c "$*" > $VAR/$log 2>&1
  local rc=$?
  echo "EXIT=$rc $log"
  grep -aE "Wrote|PASS" $VAR/$log | tail -2
  [ $rc -ne 0 ] && grep -aE "Error|Traceback" $VAR/$log | grep -av torchao | tail -3
  return 0
}

echo "waiting for a free node (all GPUs < 10 GiB)..."
while true; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 10240' | wc -l)
  [ "$busy" -eq 0 ] && break
  sleep 120
done
echo "NODE FREE at $(date +%H:%M:%S); locking clocks"
nvidia-smi -lgc 2032,2032 >/dev/null 2>&1

run contrib_tp8.log "mpirun -n 8 --allow-run-as-root python3 -u kimi_k3_layer/ablate_moe_contributions.py --sizes 1,2,4,8,16,32,64 --iters 100 --n-inputs 8 --csv-suffix _b300_situ_tp8"
run layer_tp8_situ.log "mpirun -n 8 --allow-run-as-root python3 -u kimi_k3_layer/bench_b10_kimi_k3_moe_layer.py --sizes all --iters 100 --n-inputs 8 --csv-suffix _b300_situ"
run tailsweep_situ.log "mpirun -n 8 --allow-run-as-root python3 -u kimi_k3_layer/bench_b10_kimi_k3_moe_layer.py --mode exp --sizes 16,32,64,128 --iters 100 --n-inputs 8 --sweep decode_tail=sharded_fc2_output_reduce,full_fc2_multimem_shared_reduce,packed_latent_shared_reduce --csv-suffix _b300_situ_tail"
run frontsweep_situ.log "mpirun -n 8 --allow-run-as-root python3 -u kimi_k3_layer/bench_b10_kimi_k3_moe_layer.py --mode exp --sizes 32,64 --iters 100 --n-inputs 8 --sweep decode_front=fused_fc1_shared_gate_cute,separate_sharded_fc1,separate_full_fc1 --csv-suffix _b300_situ_front"
# profiled pairs: traces land in local_results/moe_tp8_b{8,64}_{baseline,opt}_graph.trace.json
run profile_b8_b64.log "mpirun -n 8 --allow-run-as-root python3 -u kimi_k3_layer/bench_b10_kimi_k3_moe_layer.py --sizes 8,64 --iters 100 --n-inputs 8 --profile 8,64 --csv-suffix _b300_situ_prof"
run contrib_tp4.log "mpirun -n 4 --allow-run-as-root python3 -u kimi_k3_layer/ablate_moe_contributions.py --sizes 1,2,4,8,16,32,64 --iters 100 --n-inputs 8 --csv-suffix _b300_situ_tp4v2"
echo "PROGRAM DONE at $(date +%H:%M:%S)"
