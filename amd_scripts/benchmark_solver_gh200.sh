#!/bin/bash
#SBATCH --job-name=solver_benchmark_gh200
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --uenv=icon/25.2:v4
#SBATCH --view=default
#SBATCH -A csstaff

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

# Install if needed (first time only)
# uv sync --extra all --extra cuda12 --python $(which python)

source .venv/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=gh200_profiling_solver_regional
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export CUDAFLAGS="--generate-line-info -Xcompiler -g -O3 -Xcompiler -march=native -Xcompiler -Wno-unused-parameter"

export ICON_GRID="icon_benchmark_regional"

# Run the benchmark
pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --benchmark-only \
    --benchmark-warmup=on \
    --benchmark-warmup-iterations=30 \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    --benchmark-time-unit=ms \
    --benchmark-min-rounds=100 \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"
