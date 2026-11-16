#!/bin/bash
#SBATCH --job-name=dycore_benchmark_gh200
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --uenv=icon/26.7:v1@santis,/iopsstor/scratch/cscs/ioannmag/ICON4PY/icon4py-amd-benchmarks/py_venv_cuda_mpi4.squashfs:/iopsstor/scratch/cscs/ioannmag/ICON4PY/icon4py-amd-benchmarks/.venv_cuda
#SBATCH --view=default
#SBATCH -A csstaff
#SBATCH --partition normal

cd $(git rev-parse --show-toplevel)
source .venv_cuda/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_JOBS_MODE=thread
export GT4PY_BUILD_JOBS=${SLURM_CPUS_PER_TASK}
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export PYTHONOPTIMIZE=2
export CUDA_VISIBLE_DEVICES=0

ICON_GRID=icon_benchmark_global

export CC=$(which gcc)
export CXX=$(which g++) 

# Workspace size in bytes, 0 = no external workspace.
for workspace_size in 0 8589934592; do
    if [[ ${workspace_size} == 0 ]]; then
        unset ICON4PY_BACKEND_WORKSPACE_SIZE # icon4py rejects 0
    else
        export ICON4PY_BACKEND_WORKSPACE_SIZE=${workspace_size}
    fi
    export GT4PY_BUILD_CACHE_DIR=GH200_dycore_${ICON_GRID}_workspace${workspace_size}
    echo "=== ${GT4PY_BUILD_CACHE_DIR}"

    python3 -m pytest -sv -m continuous_benchmarking -p no:tach \
        --benchmark-only --benchmark-warmup=on \
        --benchmark-warmup-iterations=${ICON4PY_STENCIL_TEST_ITERATIONS} \
        --benchmark-min-rounds=${ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS} \
        --benchmark-time-unit=ms \
        --backend=dace_gpu --grid=${ICON_GRID} \
        'model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]'
done

