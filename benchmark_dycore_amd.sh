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

# MI300A ROCm environment setup
source setup_amd_env.sh

source .venv_rocm/bin/activate

# ICON4Py StencilTest related variables for benchmarking
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100

# ICON4Py performance related options
# Python optimization level
export PYTHONOPTIMIZE=2
# 8Gib external workspace storage per device (one large hip/cuMalloc allocation for temp variables in DaCe - necessary for MI300A perf)
export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592

# Below values necessary to get meaningful runtime per gt4py program
# export GT4PY_COLLECT_METRICS_LEVEL=10
# export GT4PY_DUMP_METRICS_AT_EXIT=1
# Add NVTX/ROCTX markers to each GT4Py program compiled with DaCe
# export GT4PY_ADD_GPU_TRACE_MARKERS="1"

# GT4Py compilation options
# Number of parallel compile jobs
export GT4PY_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-8}
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
# Keep DaCe src files
# export GT4PY_KEEP_DACE_BUILD_FILES=1

export ICON_GRID="icon_benchmark_global"
GRID_SUFFIX=""
if [[ "$ICON_GRID" == *"global"* ]]; then
    GRID_SUFFIX="global"
elif [[ "$ICON_GRID" == *"regional_large"* ]]; then
    GRID_SUFFIX="regional_large"
elif [[ "$ICON_GRID" == *"regional"* ]]; then
    GRID_SUFFIX="regional"
fi

export GT4PY_BUILD_CACHE_DIR=MI300A_dycore_${GRID_SUFFIX}

echo "GT4PY_BUILD_CACHE_DIR: ${GT4PY_BUILD_CACHE_DIR}"

export ROCPROF_OUTPUT_FILENAME="${GT4PY_BUILD_CACHE_DIR}"

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
