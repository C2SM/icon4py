#!/bin/bash
#SBATCH --job-name=dycore_granule_profile
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi300

# Go to the root of the icon4py repository to run the script from there
ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

# Set necessasry flags for compilation
source amd_scripts/setup_env.sh

source .venv/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=amd_profiling_granule
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"

pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --benchmark-only \
    --benchmark-warmup=on \
    --benchmark-warmup-iterations=30 \
    --backend=dace_gpu \
    --grid=icon_benchmark_regional \
    --benchmark-time-unit=ms \
    --benchmark-min-rounds 100 \
    model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[True-False]

python amd_scripts/print_gt4py_timers.py dycore_gt4py_program_metrics.json
