#!/usr/bin/env bash
# r3_sweep_inner.sh — round-3 variant sweep, run under the GPU lease.
# Each variant runs the quick dense matrix and the strided production matrix.
set -uo pipefail

PKG=/node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill

run_variant() {
    local tag="$1" vec="$2" tile="$3" silu="$4" threads="${5:-128}" prefetch="${6:-1}" gl="${7:-0}" gs="${8:-4}" fp32p="${9:-0}" fr="${10:-0}"
    for matrix in quick quick-strided; do
        docker exec -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
            -w "${PKG}" trt-dev env \
            KDA_CUTE_ALGORITHM=auto \
            KDA_CUTE_TOKEN_TILE="${tile}" \
            KDA_CUTE_VECTOR_WIDTH="${vec}" \
            KDA_CUTE_THREADS="${threads}" \
            KDA_CUTE_SILU_MODE="${silu}" \
            KDA_CUTE_PREFETCH="${prefetch}" \
            KDA_CUTE_GROUP_LOADS="${gl}" \
            KDA_CUTE_GROUP_SPAN="${gs}" \
            KDA_CUTE_FP32_PARAMS="${fp32p}" \
            KDA_CUTE_F32_RING="${fr}" \
            python3 run.py --mode benchmark --matrix "${matrix}" \
            --backends cute --warmup 50 --iterations 500 \
            --output "local_results/r3_sweep_${tag}_${matrix}.json" \
            | grep -E '^BENCH' \
            | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line[6:])
    print(f\"${tag} {r['shape']:<55} {r['latency_ms']:.6f} ms\")
"
    done
}

# F32 ring is new: full correctness case suite before timing.
docker exec -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    -w "${PKG}" trt-dev env \
    KDA_CUTE_ALGORITHM=auto \
    KDA_CUTE_TOKEN_TILE=24 \
    KDA_CUTE_VECTOR_WIDTH=4 \
    KDA_CUTE_THREADS=128 \
    KDA_CUTE_SILU_MODE=tanh \
    KDA_CUTE_GROUP_LOADS=1 \
    KDA_CUTE_GROUP_SPAN=8 \
    KDA_CUTE_F32_RING=1 \
    python3 run.py --mode correctness --backends cute \
    --output local_results/r3_f32ring_correctness.json \
    | grep -E '^(CORRECT|RESULT)'
echo "FR_CORRECTNESS_EXIT=$?"

# run_variant tag vec tile silu threads prefetch group_loads group_span fp32params f32ring
run_variant v4_tile24_gs8_fr     4 24 tanh 128 1 1 8 0 1
run_variant v4_tile16_gs8_fr     4 16 tanh 128 1 1 8 0 1
run_variant v4_tile16_gs4_fr     4 16 tanh 128 1 1 4 0 1
run_variant v4_tile24_gs4_fr     4 24 tanh 128 1 1 4 0 1
echo "SWEEP_DONE"
