#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --partition=1CN192C4G1H_MI300A_Ubuntu22
#SBATCH --exclusive

# Power/clock trace during a solver benchmark on aac6.
# Backgrounds rocm-smi sampler at 100 ms, then runs the solver test, then
# kills the sampler. Outputs:
#   <prefix>_<host>_<ts>_power.csv      one row per sample
#   <prefix>_<host>_<ts>_bench.log      benchmark stdout/stderr
#   <prefix>_<host>_<ts>_summary.txt    min/p50/p95/max stats during busy window
#
# Usage — pin to a specific node and supply a prefix:
#   sbatch --nodelist=ppac-pl1-s24-26 \
#       --output=trace_power_node26.out --error=trace_power_node26.err \
#       --export=ALL,TRACE_PREFIX=node26 \
#       amd_scripts/sbatch_trace_power_aac6.sh
#
# Or with a default cache dir:
#   GT4PY_BUILD_CACHE_DIR=amd_aac6_trace_$(date +%H%M) \
#       sbatch --nodelist=ppac-pl1-s24-30 ... amd_scripts/sbatch_trace_power_aac6.sh

set -eu

# --- modules + env ---
# Note: `cupy` is auto-loaded by aac6's prolog and is missing from `module avail`
# on at least ppac-pl1-s24-16 — explicit `module load cupy` errors there.
# Don't reload; the prolog-loaded version is what `python -c "import cupy"` uses.
module load rocm/7.2.0 openmpi mpi4py rocprofiler-compute/7.2.0
source $HOME/icon4py_debug/setup_env.sh
cd $HOME/icon4py

if [ "${ICON4PY_ENV_SOURCED:-}" != "1" ]; then
    echo "ERROR: setup_env.sh did not run" >&2
    exit 1
fi

# --- GT4Py / pytest config (matches sbatch_gt4py_timer_aac6.sh) ---
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR:-amd_aac6_trace_$(hostname -s)}
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export ICON_GRID="icon_benchmark_regional"

PREFIX="${TRACE_PREFIX:-trace_$(hostname -s)}"

echo "=== env summary ==="
echo "host:         $(hostname)"
echo "ROCM_VERSION: ${ROCM_VERSION:-unset}"
echo "cache dir:    $GT4PY_BUILD_CACHE_DIR"
echo "trace prefix: $PREFIX"
echo "==================="

# --- run the trace wrapper around the pytest invocation ---
bash amd_scripts/trace_power.sh "$PREFIX" -- \
    python3 -m pytest -sv \
        -m continuous_benchmarking \
        -p no:tach \
        --backend=dace_gpu \
        --grid=${ICON_GRID} \
        model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
        -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"
