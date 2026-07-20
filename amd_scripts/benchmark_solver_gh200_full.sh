#!/bin/bash
#SBATCH --job-name=solver_benchmark
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --uenv=icon/25.2:v4
#SBATCH --view=default
#SBATCH -A csstaff

# Go to the root of the icon4py repository to run the script from there
ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

# Set necessasry flags for compilation

# The following is necessary beforehand
# uv sync --extra all --extra cuda12 --python $(which python)

source .venv_cuda/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=gh200_profiling_solver_regional
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
# export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"
export CUDAFLAGS="--generate-line-info -Xcompiler -g -O3 -Xcompiler -march=native -Xcompiler -Wno-unused-parameter"

export ICON_GRID="icon_benchmark_regional" # TODO(CSCS): Check also `icon_benchmark_global` when the dycore GPU memory issue is fixed

# Run the benchmark and collect the runtime of the whole GT4Py program (see `GT4Py Timer Report` in the output)
# The compiled GT4Py programs will be cached in the directory specified by `GT4PY_BUILD_CACHE_DIR` to be reused for running the profilers afterwards
.venv_cuda/bin/python -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

# Get the kernel names of the GT4Py program so that we can filter them with rocprof-compute
LAST_COMPILED_DIRECTORY=$(realpath $(ls -td ${GT4PY_BUILD_CACHE_DIR}/.gt4py_cache/*/ | head -1))
echo "# Last compiled GT4Py directory: $LAST_COMPILED_DIRECTORY"
LAST_COMPILED_KERNEL_NAMES=$(grep -r -e "__global__ void.*map.*(" ${LAST_COMPILED_DIRECTORY}/src/cuda -o | sed 's/.*\s\([a-zA-Z_][a-zA-Z0-9_]*\)(.*/\1/' | sort -u)
LAST_COMPILED_KERNEL_REGEX=$(echo "$LAST_COMPILED_KERNEL_NAMES" | paste -sd'|' -)
echo "# Last compiled GT4Py kernel names:"
echo "$LAST_COMPILED_KERNEL_NAMES"
NCU_KERNEL_NAME_FILTER="--kernel-name regex:'${LAST_COMPILED_KERNEL_REGEX}'"

# Run rocprof-compute filtering the kernels of interest
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=0
export ICON4PY_STENCIL_TEST_ITERATIONS=1
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=1
ncu --set full ${NCU_KERNEL_NAME_FILTER} --import-source yes -f -o vertically_implicit_solver .venv_cuda/bin/python -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"
