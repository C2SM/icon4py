#!/bin/bash
#SBATCH --job-name=dycore_benchmark_mi300
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --uenv=prgenv-gnu/7.2.3:2579601092
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
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"
export PYTHONOPTIMIZE=2
# export GT4PY_COLLECT_METRICS_LEVEL=10

export CASE="XCD"

if [ $CASE == "baseline" ]; then
    export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="256,1,1"
    export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
    export GT4PY_VERTICAL_LOOP_BLOCKING="0"
    export DACE_compiler_cuda_chiplet_number=1
    # export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592 # 8Gib external workspace storage per device
elif [ $CASE == "workspace" ]; then
    export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="256,1,1"
    export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
    export GT4PY_VERTICAL_LOOP_BLOCKING="0"
    export DACE_compiler_cuda_chiplet_number=1
    export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592 # 8Gib external workspace storage per device
elif [ $CASE == "VLB4" ]; then
    export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="256,1,1"
    export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
    export GT4PY_VERTICAL_LOOP_BLOCKING="4"
    export DACE_compiler_cuda_chiplet_number=1
    export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592 # 8Gib external workspace storage per device
elif [ $CASE == "XCD" ]; then
    export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="256,1,1"
    export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
    export GT4PY_VERTICAL_LOOP_BLOCKING="0"
    export DACE_compiler_cuda_chiplet_number=6
    export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592 # 8Gib external workspace storage per device
elif [ $CASE == "VLB4_XCD" ]; then
    export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="256,1,1"
    export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
    export GT4PY_VERTICAL_LOOP_BLOCKING="4"
    export DACE_compiler_cuda_chiplet_number=6
    export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592 # 8Gib external workspace storage per device
fi

export GT4PY_CACHE_SUFFIX="TB2D[${ICON4PY_GPU_THREAD_BLOCK_SIZE_2D}]_TB1D[${ICON4PY_GPU_THREAD_BLOCK_SIZE_1D}]_VLB[${GT4PY_VERTICAL_LOOP_BLOCKING}]"
export GT4PY_BUILD_CACHE_DIR=mi300_dycore_global80_${CASE}_${GT4PY_CACHE_SUFFIX}

export ICON_GRID="icon_benchmark_global"

export GT4PY_METRICS_OUTPUT_PATH="gt4py_timers_${GT4PY_BUILD_CACHE_DIR}.json"

echo "GT4PY_BUILD_CACHE_DIR: ${GT4PY_BUILD_CACHE_DIR}"

export ROCPROF_OUTPUT_FILENAME="dycore_v4_${GT4PY_BUILD_CACHE_DIR}"

# rocprofv3 --kernel-trace on --marker-trace on --output-format pftrace -o ${ROCPROF_OUTPUT_FILENAME} -- \
python3 -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --benchmark-only \
    --benchmark-warmup=on \
    --benchmark-warmup-iterations=10 \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    --benchmark-time-unit=ms \
    --benchmark-min-rounds 100 \
    model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]

# rocpd convert -i "${ROCPROF_OUTPUT_FILENAME}.rocpd" --output-format pftrace
