#!/bin/bash
#SBATCH --job-name=JW_benchmark_mi300_tb
#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH --uenv=prgenv-gnu/7.2.3:2579601092,/capstor/scratch/cscs/ioannmag/cycle37/dace-determ/py_venv_rocm723.squashfs:/capstor/scratch/cscs/ioannmag/cycle37/icon4py_amd/.venv
#SBATCH --view=default
#SBATCH -A csstaff
#SBATCH --partition mi300

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

source setup_amd_env.sh

source venv_mi300/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export DACE_compiler_build_folder_mode="development"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=1000
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"
export ICON4PY_DRIVER_LOGGING_LEVEL="warning"
export DACE_compiler_cuda_chiplet_number=6

export ICON_GRID="icon_benchmark_global"
SUFFIX=""
if [[ "$ICON_GRID" == *"regional"* ]]; then
    SUFFIX="regional120"
elif [[ "$ICON_GRID" == *"global"* ]]; then
    SUFFIX="global120"
fi

PREFIX="VLB_HLB_SWEEP_XCD"
BASE_GT4PY_BUILD_CACHE_DIR="MI300A_${PREFIX}_${SUFFIX}"
PIDS=()

# Launch 4 concurrent benchmark processes, one per GPU.
for worker_id in 0 1 2 3; do
    (
        export GT4PY_BLOCK_SIZE_HEURISTICS="0"
        if [ "$worker_id" == "0" ]; then
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="64,4,1"
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
            export GT4PY_HORIZONTAL_LOOP_BLOCKING="2"
            export GT4PY_VERTICAL_LOOP_BLOCKING="0"
        elif [ "$worker_id" == "1" ]; then
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="64,4,1"
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
            export GT4PY_HORIZONTAL_LOOP_BLOCKING="0"
            export GT4PY_VERTICAL_LOOP_BLOCKING="2"
        elif [ "$worker_id" == "2" ]; then
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="64,4,1"
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
            export GT4PY_HORIZONTAL_LOOP_BLOCKING="0"
            export GT4PY_VERTICAL_LOOP_BLOCKING="4"
        elif [ "$worker_id" == "3" ]; then
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="64,4,1"
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
            export GT4PY_HORIZONTAL_LOOP_BLOCKING="0"
            export GT4PY_VERTICAL_LOOP_BLOCKING="8"
        fi
        export GT4PY_CACHE_SUFFIX="TB2D[${ICON4PY_GPU_THREAD_BLOCK_SIZE_2D}]_TB1D[${ICON4PY_GPU_THREAD_BLOCK_SIZE_1D}]_HLB[${GT4PY_HORIZONTAL_LOOP_BLOCKING}]_VLB[${GT4PY_VERTICAL_LOOP_BLOCKING}]"
        export HIP_VISIBLE_DEVICES="${worker_id}"
        export GT4PY_BUILD_CACHE_DIR="${BASE_GT4PY_BUILD_CACHE_DIR}_${GT4PY_CACHE_SUFFIX}"
        export OUTPUT_PATH=$(pwd)/standalone_driver_output_${GT4PY_BUILD_CACHE_DIR}_wall
        echo "[worker ${worker_id}] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}, GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR} ICON4PY_GPU_THREAD_BLOCK_SIZE_2D=${ICON4PY_GPU_THREAD_BLOCK_SIZE_2D} ICON4PY_GPU_THREAD_BLOCK_SIZE_1D=${ICON4PY_GPU_THREAD_BLOCK_SIZE_1D} GT4PY_HORIZONTAL_LOOP_BLOCKING=${GT4PY_HORIZONTAL_LOOP_BLOCKING} GT4PY_VERTICAL_LOOP_BLOCKING=${GT4PY_VERTICAL_LOOP_BLOCKING}"

        rocprofv3 --kernel-trace on --hip-trace on --marker-trace on --memory-copy-trace on --memory-allocation-trace on --output-format csv -o rocprof_${PREFIX}_${SUFFIX}_${GT4PY_CACHE_SUFFIX} -- \
            pytest -sv \
            -m continuous_benchmarking \
            -p no:tach \
            --benchmark-only \
            --benchmark-warmup=on \
            --benchmark-warmup-iterations=30 \
            --backend=dace_gpu \
            --grid=${ICON_GRID} \
            --benchmark-time-unit=ms \
            --benchmark-min-rounds 1000 \
            model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]
    ) &

    PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

exit "${failed}"
