#!/bin/bash
#SBATCH --job-name=solver_gt4py_timer_rocm72
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi300
#SBATCH --exclusive
#SBATCH --uenv=b2550889de318ab5
#SBATCH --view=default
#SBATCH --output=gt4py_timer_rocm72.out
#SBATCH --error=gt4py_timer_rocm72.err

# GT4Py-timer-only run for the icon4py-rocm72 env (ROCm 7.2 + gt4py PR 2578
# with hlb2noscan patch applied to model_options.py).
#
# Submit from /capstor/scratch/cscs/gandanie/git/icon/icon4py-rocm72/:
#   sbatch amd_scripts/sbatch_gt4py_timer_rocm72.sh
#
# Override cache dir without editing this file:
#   GT4PY_BUILD_CACHE_DIR=amd_rocm72_baseline_v1 sbatch amd_scripts/sbatch_gt4py_timer_rocm72.sh

set -eu

# --- working dir ---
ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$ICON4PY_GIT_ROOT"

if [ "$(basename "$PWD")" != "icon4py-rocm72" ]; then
    echo "ERROR: expected to be in icon4py-rocm72/, got $(basename "$PWD")" >&2
    exit 1
fi

# --- env ---
# setup_env.sh is uenv-agnostic: it auto-detects ROCM_VERSION from hipcc and
# globs the rocprofiler-dev lib path. Works under both 7.1 and 7.2 uenvs.
source amd_scripts/setup_env.sh

# Activate the rocm72 venv (NOT .venv or .venv_rocm)
source .venv_rocm72/bin/activate

# --- GT4Py / pytest config ---
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR:-amd_rocm72_hlb2noscan_v1}
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export HIPFLAGS="-std=c++17 -fPIC -O3 --offload-arch=gfx942 -Wno-unused-parameter"

export ICON_GRID="icon_benchmark_regional"

# --- sanity dump ---
echo "=== env summary ==="
echo "uenv:        ${UENV_VIEW:-${UENV_MOUNT_LIST:-unknown}}"
echo "ROCM_VERSION: ${ROCM_VERSION:-unset}"
echo "python:      $(command -v python) ($(python --version 2>&1))"
echo "hipcc:       $(command -v hipcc)"
echo "hip ver:     $(hipcc --version 2>&1 | head -2 | tail -1)"
echo "gt4py:       $(python -c 'import gt4py; print(gt4py.__file__)' 2>/dev/null)"
echo "dace:        $(python -c 'import dace; print(dace.__version__)' 2>/dev/null)"
echo "cache dir:   $GT4PY_BUILD_CACHE_DIR"
echo "==================="

# --- single-stencil GT4Py timer run ---
# Look for "GT4Py Timer Report" in the output; that's the headline number.
.venv_rocm72/bin/python -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"
