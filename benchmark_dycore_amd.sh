#!/bin/bash
#SBATCH --job-name=dycore_benchmark_mi300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --uenv=prgenv-gnu/7.2.3:v1,/iopsstor/scratch/cscs/ioannmag/ICON4PY/icon4py-amd-benchmarks/py_venv_rocm_mpi4.squashfs:/iopsstor/scratch/cscs/ioannmag/ICON4PY/icon4py-amd-benchmarks/.venv_rocm
#SBATCH --view=default
#SBATCH -A csstaff
#SBATCH --partition mi300
#SBATCH --reservation cscs

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

source setup_amd_env.sh

source .venv_rocm/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export DACE_compiler_build_folder_mode="development"
# export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
# export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"
export PYTHONOPTIMIZE=2
# export GT4PY_COLLECT_METRICS_LEVEL=10

export GT4PY_BUILD_JOBS_MODE=thread               # serial | thread | process (default: process)
export GT4PY_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-8} # number of parallel compile jobs

export DACE_compiler_cuda_chiplet_number=1
export ICON4PY_DACE_PERSISTENT_DISABLE="1"
export ICON4PY_VLB_DISABLE="1"
# export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592 # 8Gib external workspace storage per device

export ICON_GRID="icon_benchmark_global"
GRID_SUFFIX=""
if [[ "$ICON_GRID" == *"global"* ]]; then
    GRID_SUFFIX="global"
elif [[ "$ICON_GRID" == *"regional_large"* ]]; then
    GRID_SUFFIX="regional_large"
elif [[ "$ICON_GRID" == *"regional"* ]]; then
    GRID_SUFFIX="regional"
fi

export GT4PY_BUILD_CACHE_DIR=MI300A_dycore_noexternalworkspace_nopersistent_chiplet${DACE_compiler_cuda_chiplet_number}_VLB1_${GRID_SUFFIX}
export GT4PY_METRICS_OUTPUT_PATH="gt4py_timers_${GT4PY_BUILD_CACHE_DIR}.json"

echo "GT4PY_BUILD_CACHE_DIR: ${GT4PY_BUILD_CACHE_DIR}"

export ROCPROF_OUTPUT_FILENAME="dycore_v4_${GT4PY_BUILD_CACHE_DIR}"

# rocprofv3 --kernel-trace on --marker-trace on --output-format pftrace -o ${ROCPROF_OUTPUT_FILENAME} -- \
python3 -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --benchmark-only \
    --benchmark-warmup=on \
    --benchmark-warmup-iterations=${ICON4PY_STENCIL_TEST_ITERATIONS} \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    --benchmark-time-unit=ms \
    --benchmark-min-rounds ${ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS} \
    model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]

# rocpd convert -i "${ROCPROF_OUTPUT_FILENAME}.rocpd" --output-format pftrace
