#!/usr/bin/env bash
# ncu_inner_profiles.sh — called by gpu_run.sh with CUDA_VISIBLE_DEVICES set and
# SM clock locked to 1500 MHz. Runs all NCU profile sessions via nsenter (host-root)
# to bypass ERR_NVGPUCTRPERM inside the container.
#
# Arguments: <container_pid> <pkg_dir> <results_dir>
set -uo pipefail

CONTAINER_PID="$1"
PKG_DIR="$2"
RESULTS_DIR="$3"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
NCU_BIN="/usr/local/bin/ncu"   # path inside container filesystem (via nsenter)

# Container env needed for TRT-LLM/MPI initialization (Triton backend).
CONTAINER_LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
CONTAINER_PATH="/usr/local/lib/python3.12/dist-packages/torch_tensorrt/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin:/opt/tensorrt/bin"

echo "[ncu_inner] GPU=${GPU}"
echo "[ncu_inner] container_pid=${CONTAINER_PID}"
echo "[ncu_inner] results_dir=${RESULTS_DIR}"
echo "[ncu_inner] SM clock: $(nvidia-smi -i "${GPU}" --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null)"
echo "[ncu_inner] ncu: $(nsenter --target "${CONTAINER_PID}" --mount --pid --uts --net -- "${NCU_BIN}" --version 2>&1 | head -1)"

# ── Helper: run one NCU profile ──────────────────────────────────────────────
# Usage: run_one <tag> <backend> <T> <D> <B> [<tile> [<algorithm> [<vector_width> [<threads> [<extra driver args...>]]]]]
run_one() {
    local tag="$1" backend="$2" T="$3" D="$4" B="$5" tile="${6:-}"
    local algorithm="${7:-direct}" vector_width="${8:-4}" threads="${9:-128}"
    shift $(( $# > 9 ? 9 : $# ))
    local extra_args="$*"
    local report="${RESULTS_DIR}/${tag}"
    local log_file="${RESULTS_DIR}/${tag}_ncu.log"
    local csv_file="${RESULTS_DIR}/${tag}.csv"

    local tile_args=""
    [ -n "${tile}" ] && tile_args="--tile ${tile}"

    echo ""
    echo "============================================================"
    echo "[ncu_inner] BEGIN ${tag}"
    echo "  backend=${backend} T=${T} D=${D} B=${B} tile=${tile:-auto}"
    echo "  SM clock: $(nvidia-smi -i "${GPU}" --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null)"
    echo "============================================================"

    # nsenter enters the container's mount/pid/uts/net/ipc namespaces while keeping
    # host-root credentials, bypassing RmProfilingAdminOnly restrictions.
    # ncu flags:
    #   --range-filter 'yes:1:': profile only the first cudaProfilerStart/Stop range
    #     (the profiling driver marks exactly one range containing 1 kernel iteration).
    #   --set full: comprehensive metric sections (SpeedOfLight, LaunchStats,
    #     Occupancy, MemoryWorkloadAnalysis, WarpStateStatistics, InstructionStats).
    #   --launch-count 1: limit to one kernel launch within the matched range.
    #   -o / -f: save binary .ncu-rep, overwriting any existing file.
    # All ncu output (progress, warnings, CSV data) goes to log_file.
    # A clean CSV is then re-extracted from the saved report.
    nsenter --target "${CONTAINER_PID}" --mount --pid --uts --net --ipc -- \
        bash -c "
export CUDA_VISIBLE_DEVICES='${GPU}'
export LD_LIBRARY_PATH='${CONTAINER_LD_LIBRARY_PATH}'
export PATH='${CONTAINER_PATH}'
export OPAL_PREFIX=/usr/local/mpi
export OMPI_MCA_coll_hcoll_enable=0
export KDA_CUTE_TOKEN_TILE='${tile:-auto}'
export KDA_CUTE_ALGORITHM='${algorithm}'
export KDA_CUTE_VECTOR_WIDTH='${vector_width}'
export KDA_CUTE_THREADS='${threads}'
cd '${PKG_DIR}'
'${NCU_BIN}' \
    --range-filter 'yes:1:' \
    --set full \
    --launch-count 1 \
    -o '${report}' \
    -f \
    python3 '${PKG_DIR}/ncu_profile_driver.py' \
        --backend '${backend}' \
        --T '${T}' --D '${D}' --B '${B}' ${tile_args} \
        --algorithm '${algorithm}' \
        --vector-width '${vector_width}' --threads '${threads}' \
        ${extra_args} \
        --warmup 10 --iterations 1
" > "${log_file}" 2>&1
    local profile_rc=$?
    echo "[ncu_inner] profile exit=${profile_rc}"

    # Tail the log for visibility
    tail -6 "${log_file}" 2>/dev/null || true

    if [ ! -f "${report}.ncu-rep" ]; then
        echo "[ncu_inner] ERROR: no .ncu-rep produced for ${tag}"
    else
        echo "[ncu_inner] report: ${report}.ncu-rep ($(du -sh "${report}.ncu-rep" 2>/dev/null | cut -f1))"
        # Extract clean CSV from the saved binary report
        nsenter --target "${CONTAINER_PID}" --mount --pid --uts --net -- \
            bash -c "
'${NCU_BIN}' -i '${report}.ncu-rep' --csv --print-units base 2>/dev/null
" > "${csv_file}" 2>&1
        local csv_lines
        csv_lines=$(wc -l < "${csv_file}" 2>/dev/null || echo 0)
        echo "[ncu_inner] CSV: ${csv_file} (${csv_lines} lines)"
    fi

    echo "[ncu_inner] END ${tag}"
}

# ── Profile schedule (serialized, GPU clock locked by gpu_run.sh) ────────────

PROFILE_SET="${KDA_NCU_PROFILE_SET:-round2-early}"
if [ "${PROFILE_SET}" = "round2-early" ]; then
    run_one "r2_stream_v4_t128_tile16_T8192_D3072_B1" \
        cute 8192 3072 1 16 stream 4 128
elif [ "${PROFILE_SET}" = "round3-early" ]; then
    # Sweep winners: no-prefetch tanh for contiguous long, tile20 tanh for strided.
    run_one "r3_v4tile12_tanh_nopf_T8192_D3072_B1" \
        cute 8192 3072 1 12 stream 4 128 --silu-mode tanh --prefetch 0
    run_one "r3_v4tile20_tanh_T8192_D4608s_B1" \
        cute 8192 4608 1 20 stream 4 128 --silu-mode tanh --row-stride 4752 --no-bias
elif [ "${PROFILE_SET}" = "round3-gs8" ]; then
    # Group-loads span-8 tile-24 sweep-4 winner, dense and strided long.
    run_one "r3_v4tile24_gs8_T8192_D3072_B1" \
        cute 8192 3072 1 24 stream 4 128 --silu-mode tanh --group-loads --group-span 8
    run_one "r3_v4tile24_gs8_T8192_D4608s_B1" \
        cute 8192 4608 1 24 stream 4 128 --silu-mode tanh --group-loads --group-span 8 --row-stride 4752 --no-bias
elif [ "${PROFILE_SET}" = "round3-final" ]; then
    # Selected single config: stream v4 t128 tile16, group loads span 4,
    # F32 ring (backend default), tanh SiLU.
    run_one "r3_selected_T8192_D3072_B1" \
        cute 8192 3072 1 16 stream 4 128 --silu-mode tanh --group-loads --group-span 4
    run_one "r3_selected_T8192_D3072_B8" \
        cute 8192 3072 8 16 stream 4 128 --silu-mode tanh --group-loads --group-span 4
    run_one "r3_selected_T8192_D4608s_B1" \
        cute 8192 4608 1 16 stream 4 128 --silu-mode tanh --group-loads --group-span 4 --row-stride 4752 --no-bias
    run_one "r3_selected_T8192_D4608s_B8" \
        cute 8192 4608 8 16 stream 4 128 --silu-mode tanh --group-loads --group-span 4 --row-stride 4752 --no-bias
    run_one "r3_triton_T8192_D4608s_B1" trt_triton_head 8192 4608 1 "" direct 4 128 --row-stride 4752 --no-bias
    run_one "r3_triton_T8192_D4608s_B8" trt_triton_head 8192 4608 8 "" direct 4 128 --row-stride 4752 --no-bias
elif [ "${PROFILE_SET}" = "round2-final" ]; then
    run_one "r2_selected_tile12_T128_D3072_B1" \
        cute 128 3072 1 12 stream 4 128
    run_one "r2_selected_tile12_T8192_D3072_B1" \
        cute 8192 3072 1 12 stream 4 128
    run_one "r2_selected_tile12_T8192_D3072_B8" \
        cute 8192 3072 8 12 stream 4 128
    run_one "r2_triton_T128_D3072_B1" trt_triton_head 128 3072 1
    run_one "r2_triton_T8192_D3072_B1" trt_triton_head 8192 3072 1
    run_one "r2_triton_T8192_D3072_B8" trt_triton_head 8192 3072 8
else
    echo "[ncu_inner] ERROR: unknown KDA_NCU_PROFILE_SET=${PROFILE_SET}" >&2
    exit 2
fi

echo ""
echo "[ncu_inner] ============ ALL PROFILES COMPLETE ============"
echo "[ncu_inner] Results: ${RESULTS_DIR}"
ls -lah "${RESULTS_DIR}"/*.ncu-rep 2>/dev/null || echo "[ncu_inner] no .ncu-rep files found"
