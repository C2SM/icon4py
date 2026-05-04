#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --partition=1CN192C4G1H_MI300A_Ubuntu22
#SBATCH --exclusive

# AAC6 rocprof-compute run, parallel to beverin's sbatch_rocprof_compute_rocm72.sh.
# Layered on top of the GT4Py timer flow: compiles + runs once for warmup,
# then runs again under rocprof-compute profile to collect per-kernel pmc data.
#
# Required: $HOME/run_with_patch.py — the cupy NVRTC monkey-patch wrapper that
# injects -DHIP_DISABLE_WARP_SYNC_BUILTINS and #include <cupy/hip_workaround.cuh>
# to work around aac6's CuPy 14.0.1 + ROCm 7.2 hiprtc_runtime.h 64-bit-mask issue.
#
# Pin to a node and set a unique cache dir per node:
#   GT4PY_BUILD_CACHE_DIR=amd_aac6_pmc_26 \
#       sbatch --nodelist=ppac-pl1-s24-26 \
#       --output=rocprof_aac6_node26.out --error=rocprof_aac6_node26.err \
#       amd_scripts/sbatch_rocprof_compute_aac6.sh
#
# Output: rocprof-compute/rcu_<cache_dir>/MI300A_A1/pmc_perf.csv (similar to beverin)

set -eu

# Note: `cupy` is auto-loaded by aac6's prolog and is missing from `module avail`
# on at least ppac-pl1-s24-16 — explicit `module load cupy` errors there.
# Don't reload; the prolog-loaded version is what `python -c "import cupy"` uses.
module load rocm/7.2.0 openmpi mpi4py rocprofiler-compute/7.2.0

source $HOME/icon4py_debug/setup_env.sh
unset PYTHONPATH

cd $HOME/icon4py

if [ "${ICON4PY_ENV_SOURCED:-}" != "1" ]; then
    echo "ERROR: setup_env.sh did not export ICON4PY_ENV_SOURCED." >&2
    exit 1
fi

if [ ! -f $HOME/run_with_patch.py ]; then
    echo "ERROR: $HOME/run_with_patch.py not found — needed for cupy NVRTC fix." >&2
    exit 1
fi

OUTPUT_DIR=$HOME/icon4py/rocprof-compute

# --- GT4Py / pytest config (matches sbatch_gt4py_timer_aac6.sh) ---
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR:-amd_aac6_pmc_$(hostname -s)}
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100

export ICON_GRID="icon_benchmark_regional"

# --- env summary ---
echo "=== env summary ==="
echo "host:         $(hostname)"
echo "rocm module:  $(module list 2>&1 | grep -oE 'rocm/[0-9.]+' | head -1)"
echo "ROCM_VERSION: ${ROCM_VERSION:-unset}"
echo "python:       $(command -v python) ($(python --version 2>&1))"
echo "hipcc:        $(command -v hipcc)"
echo "cupy:         $(python -c 'import cupy; print(cupy.__version__, cupy.__file__)' 2>/dev/null)"
echo "icon4py HEAD: $(git log --oneline -1)"
echo "cache dir:    $GT4PY_BUILD_CACHE_DIR"
echo "==================="

# --- phase 1: warmed-up benchmark to populate the build cache ---
# Look for "GT4Py Timer Report" in the output for the timer median.
python3 $HOME/run_with_patch.py -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

# --- phase 2: extract kernel names from the cache for filtering ---
LAST_COMPILED_DIRECTORY=$(realpath $(ls -td ${GT4PY_BUILD_CACHE_DIR}/.gt4py_cache/*/ | head -1))
echo "# Last compiled GT4Py directory: $LAST_COMPILED_DIRECTORY"
LAST_COMPILED_KERNEL_NAMES=$(grep -r -e "__global__ void.*map.*(" ${LAST_COMPILED_DIRECTORY}/src/cuda -o | sed 's/.*\s\([a-zA-Z_][a-zA-Z0-9_]*\)(.*/\1/')
echo "# Last compiled GT4Py kernel names:"
echo "$LAST_COMPILED_KERNEL_NAMES"
ROCPROF_COMPUTE_KERNEL_NAME_FILTER="-k $(echo "$LAST_COMPILED_KERNEL_NAMES" | paste -sd'|')"

# --- phase 3: rocprof-compute on minimal warmup (1 round each kernel only) ---
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=0
export ICON4PY_STENCIL_TEST_ITERATIONS=1
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=1

rocprof-compute profile \
    --no-roof \
    --name rcu_${GT4PY_BUILD_CACHE_DIR} \
    ${ROCPROF_COMPUTE_KERNEL_NAME_FILTER} \
    --format-rocprof-output csv \
    --kernel-names \
    -R FP64 \
    -p ${OUTPUT_DIR} \
    -- \
    python3 $HOME/run_with_patch.py -m pytest -sv \
        -m continuous_benchmarking \
        -p no:tach \
        --backend=dace_gpu \
        --grid=${ICON_GRID} \
        model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
        -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

echo "# pmc_perf.csv: ${OUTPUT_DIR}/rcu_${GT4PY_BUILD_CACHE_DIR}/MI300A_A1/pmc_perf.csv"
