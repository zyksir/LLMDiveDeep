#!/usr/bin/env bash
# run_ncu_profiles.sh — NCU profiling harness for blackwell_kda_causal_conv1d_prefill.
# Uses the repository GPU lease + nsenter (host root) to bypass ERR_NVGPUCTRPERM.
#
# Usage:
#   bash /node-storage/LLMDiveDeep/kernels/research/kda_causal_conv1d_prefill/run_ncu_profiles.sh
#
# All GPU work is serialized; the same 1500 MHz SM clock lock as benchmark runs.
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${PKG_DIR}/local_results/ncu"
mkdir -p "${RESULTS_DIR}"

CUTEDSLGEN_ROOT="${CUTEDSLGEN_ROOT:-/node-storage/CuTeDSLGen}"
GPU_RUN="${CUTEDSLGEN_ROOT}/evaluation/decomposition_ab/gpu_run.sh"
CONTAINER_NAME="trt-dev"
CONTAINER_PID=$(docker inspect "${CONTAINER_NAME}" --format '{{.State.Pid}}')

echo "[ncu_harness] container=${CONTAINER_NAME} container_pid=${CONTAINER_PID}"
echo "[ncu_harness] results_dir=${RESULTS_DIR}"
echo "[ncu_harness] Claiming GPU via lease..."

# gpu_run.sh sets CUDA_VISIBLE_DEVICES + locks SM clock to 1500 MHz.
# ncu_inner_profiles.sh uses nsenter (host-root) to profile inside the container.
bash "${GPU_RUN}" -- \
    bash "${PKG_DIR}/ncu_inner_profiles.sh" "${CONTAINER_PID}" "${PKG_DIR}" "${RESULTS_DIR}"

echo "[ncu_harness] All profiles complete. Results: ${RESULTS_DIR}"
