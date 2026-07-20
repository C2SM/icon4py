#!/bin/bash
#SBATCH --job-name=solver_occupancy_sweep
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi300

# Sweep amdgpu_waves_per_eu to control occupancy on MI300X (gfx942).
# This is the AMD equivalent of NVIDIA's --maxrregcount trick.
#
# Since ROCm 7.1.0 / clang 20.0.0 doesn't expose VGPR-limit flags via -mllvm,
# we patch the DaCe-generated HIP source code with:
#   __attribute__((amdgpu_waves_per_eu(min, max)))
# then force recompilation.

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

source amd_scripts/setup_env.sh
source .venv/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON_GRID="icon_benchmark_regional"
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"

TEST_FILE="model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py"
TEST_FILTER="-k test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

BENCHMARK_ARGS="-sv -m continuous_benchmarking -p no:tach --benchmark-only --benchmark-warmup=on --benchmark-warmup-iterations=30 --backend=dace_gpu --grid=${ICON_GRID} --benchmark-time-unit=ms --benchmark-min-rounds=100"

# --- Step 0: Build baseline (no attribute, compiler default) ---
echo "=========================================="
echo "# Baseline: no waves_per_eu attribute"
echo "=========================================="
export GT4PY_BUILD_CACHE_DIR="amd_occupancy_baseline"

pytest ${BENCHMARK_ARGS} ${TEST_FILE} ${TEST_FILTER} 2>&1 | tee occupancy_baseline.log
echo "# Done with baseline"

# --- Step 1..N: Patch and rebuild with different waves_per_eu values ---
for MAX_WAVES in 2 3 4 5 8; do
    echo "=========================================="
    echo "# Testing amdgpu_waves_per_eu(1, ${MAX_WAVES})"
    echo "=========================================="

    CACHE_DIR="amd_occupancy_waves${MAX_WAVES}"
    export GT4PY_BUILD_CACHE_DIR="${CACHE_DIR}"

    # First run to generate the code (will also compile with default settings)
    pytest -sv -m continuous_benchmarking -p no:tach --backend=dace_gpu --grid=${ICON_GRID} ${TEST_FILE} ${TEST_FILTER} 2>&1 | head -50

    # Patch the generated HIP source with the waves_per_eu attribute
    python amd_scripts/set_waves_per_eu.py ${CACHE_DIR} --min-waves 1 --max-waves ${MAX_WAVES}

    # Delete build dir to force recompilation with the patched source
    rm -rf ${CACHE_DIR}/.gt4py_cache/*/build/

    # Re-run benchmark (will recompile from patched source)
    pytest ${BENCHMARK_ARGS} ${TEST_FILE} ${TEST_FILTER} 2>&1 | tee occupancy_waves${MAX_WAVES}.log

    echo "# Done with waves_per_eu(1, ${MAX_WAVES})"
    echo ""
done

echo "# Occupancy sweep complete. Compare results:"
echo "# grep -B1 -A5 'benchmark:' occupancy_*.log"
