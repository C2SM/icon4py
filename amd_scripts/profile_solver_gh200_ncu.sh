#!/bin/bash
#SBATCH --job-name=solver_ncu_gh200
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --uenv=icon/25.2:v4
#SBATCH --view=default
#SBATCH -A csstaff

# Run ncu profiling on GH200 to collect per-kernel metrics for the solver.
# Skips numerical verification since ncu runs only 1 iteration.

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

source .venv_cuda/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=gh200_profiling_solver_regional
export CUDAFLAGS="--generate-line-info -Xcompiler -g -O3 -Xcompiler -march=native -Xcompiler -Wno-unused-parameter"

export ICON_GRID="icon_benchmark_regional"

ncu --set full -k regex:'map.*' --import-source yes -o gh200_solver \
    .venv_cuda/bin/python -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --skip-stenciltest-verification \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"
