#!/bin/bash
#SBATCH --job-name=solver_trace_power_rocm72
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi300
#SBATCH --exclusive
#SBATCH --uenv=b2550889de318ab5
#SBATCH --view=default

# Power/clock trace during a solver benchmark on beverin.
# Mirrors sbatch_trace_power_aac6.sh but uses beverin's uenv-based env.
# Backgrounds rocm-smi/amd-smi sampler at 100 ms while the solver test runs,
# then kills the sampler. Outputs:
#   <prefix>_<host>_<ts>_power.csv      one row per sample
#   <prefix>_<host>_<ts>_bench.log      benchmark stdout/stderr
#   <prefix>_<host>_<ts>_summary.txt    min/p50/p95/max stats during busy window
#
# Usage — pin to a specific node and supply a prefix:
#   sbatch --nodelist=nid002920 \
#       --output=trace_power_nid002920.out --error=trace_power_nid002920.err \
#       --export=ALL,TRACE_PREFIX=nid002920 \
#       amd_scripts/sbatch_trace_power_rocm72.sh

set -eu

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$ICON4PY_GIT_ROOT"

if [ "$(basename "$PWD")" != "icon4py-rocm72" ]; then
    echo "ERROR: expected to be in icon4py-rocm72/, got $(basename "$PWD")" >&2
    exit 1
fi

# Env: uenv-agnostic setup_env.sh + the rocm72 venv
source amd_scripts/setup_env.sh
source .venv_rocm72/bin/activate

# GT4Py / pytest config — matches sbatch_gt4py_timer_rocm72.sh
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR:-amd_rocm72_trace_$(hostname -s)}
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export HIPFLAGS="-std=c++17 -fPIC -O3 --offload-arch=gfx942 -Wno-unused-parameter"
export ICON_GRID="icon_benchmark_regional"

PREFIX="${TRACE_PREFIX:-trace_$(hostname -s)}"

echo "=== env summary ==="
echo "host:         $(hostname)"
echo "uenv:         ${UENV_VIEW:-${UENV_MOUNT_LIST:-unknown}}"
echo "ROCM_VERSION: ${ROCM_VERSION:-unset}"
echo "python:       $(command -v python) ($(python --version 2>&1))"
echo "hipcc:        $(command -v hipcc)"
echo "hip ver:      $(hipcc --version 2>&1 | head -2 | tail -1)"
echo "gt4py:        $(python -c 'import gt4py; print(gt4py.__file__)' 2>/dev/null)"
echo "cache dir:    $GT4PY_BUILD_CACHE_DIR"
echo "trace prefix: $PREFIX"
echo "==================="

# Run the trace wrapper around the pytest invocation
bash amd_scripts/trace_power.sh "$PREFIX" -- \
    .venv_rocm72/bin/python -m pytest -sv \
        -m continuous_benchmarking \
        -p no:tach \
        --backend=dace_gpu \
        --grid=${ICON_GRID} \
        model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
        -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"
