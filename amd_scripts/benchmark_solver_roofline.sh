#!/bin/bash
#SBATCH --job-name=solver_roofline
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi300

# Fixed version of benchmark_solver.sh that generates per-kernel roofline plots
# (avoids the filename-too-long error when all kernel names are concatenated).

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

source amd_scripts/setup_env.sh
source .venv/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=amd_profiling_solver_regional
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"

export ICON_GRID="icon_benchmark_regional"

# First run the benchmark to populate the cache (reuses benchmark_solver.sh step 1)
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100

PYTEST_CMD="$(which python3.12) -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

# Warm up / populate cache
pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

# Get the kernel names of the GT4Py program so that we can filter them with rocprof-compute
LAST_COMPILED_DIRECTORY=$(realpath $(ls -td ${GT4PY_BUILD_CACHE_DIR}/.gt4py_cache/*/ | head -1))
echo "# Last compiled GT4Py directory: $LAST_COMPILED_DIRECTORY"
LAST_COMPILED_KERNEL_NAMES=$(grep -r -e "__global__ void.*map.*(" ${LAST_COMPILED_DIRECTORY}/src/cuda -o | sed 's/.*\s\([a-zA-Z_][a-zA-Z0-9_]*\)(.*/\1/')
echo "# Last compiled GT4Py kernel names:"
echo "$LAST_COMPILED_KERNEL_NAMES"

# --- Step 1: Profile all kernels together (for summary metrics) ---
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=0
export ICON4PY_STENCIL_TEST_ITERATIONS=1
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=1
ROCPROF_COMPUTE_KERNEL_NAME_FILTER="-k $LAST_COMPILED_KERNEL_NAMES"

rocprof-compute profile --name rcu_${GT4PY_BUILD_CACHE_DIR}_all ${ROCPROF_COMPUTE_KERNEL_NAME_FILTER} --kernel-names -- ${PYTEST_CMD}

# Analyze all kernels together (no roofline to avoid filename-too-long issue)
rocprof-compute analyze -p workloads/rcu_${GT4PY_BUILD_CACHE_DIR}_all --report-diff 0

# --- Step 2: Profile and generate roofline per kernel (avoids filename-too-long error) ---
ROOFLINE_OUTPUT_DIR="roofline_${GT4PY_BUILD_CACHE_DIR}"
mkdir -p ${ROOFLINE_OUTPUT_DIR}

for KERNEL_NAME in $LAST_COMPILED_KERNEL_NAMES; do
    echo "# Profiling kernel: $KERNEL_NAME"
    PROFILE_NAME="rcu_${GT4PY_BUILD_CACHE_DIR}_${KERNEL_NAME}"

    rocprof-compute profile --name ${PROFILE_NAME} -k ${KERNEL_NAME} --kernel-names -- ${PYTEST_CMD}

    # Generate roofline for this single kernel
    rocprof-compute analyze -p workloads/${PROFILE_NAME} --roof-only -o ${ROOFLINE_OUTPUT_DIR}/${KERNEL_NAME} || \
        echo "# WARNING: Roofline generation failed for ${KERNEL_NAME}"
done

echo "# Roofline plots saved in ${ROOFLINE_OUTPUT_DIR}/"
