#!/usr/bin/env bash
# r3_final_inner.sh — round-3 final gate runs for the selected single config
# (backend defaults: stream W4, v4, 128 threads, tile16, group loads span 4,
# F32 ring, tanh SiLU). Run under the GPU lease.
set -uo pipefail

PKG=/node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill

run_pkg() {
    docker exec -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        -w "${PKG}" trt-dev "$@"
}

echo "=== [1/3] case correctness (selected defaults) ==="
run_pkg python3 run.py --mode correctness --backends cute \
    --output local_results/r3_selected_case_correctness.json \
    | grep -E '^(CORRECT|RESULT)'
echo "CASES_EXIT=$?"

echo "=== [2/3] full matrix correctness (dense 72 + strided production) ==="
run_pkg python3 run.py --mode matrix-correctness --matrix full --backends cute \
    --output local_results/r3_selected_full_correctness.json \
    | grep -E '^(CORRECT|RESULT)'
echo "FULL_EXIT=$?"

echo "=== [3/3] main benchmark matrix vs unchanged git-HEAD Triton (5 rounds) ==="
run_pkg python3 run.py --mode benchmark --matrix main \
    --backends trt_triton_head,cute --warmup 50 --iterations 500 --rounds 5 \
    --output local_results/r3_selected_main_confidence.json \
    | grep -E '^BENCH' \
    | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line[6:])
    q = r.get('latency_quantiles_ms') or {}
    print(f\"{r['backend'][:44]:<46} {r['shape']:<52} \"
          f\"{r['latency_ms']:.6f} ms  (min {q.get('min_ms', float('nan')):.6f} / \"
          f\"max {q.get('max_ms', float('nan')):.6f})\")
"
echo "BENCH_EXIT=$?"
echo "FINAL_DONE"
