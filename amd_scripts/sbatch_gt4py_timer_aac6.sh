#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --output=gt4py_timer_aac6.out
#SBATCH --error=gt4py_timer_aac6.err
#SBATCH --partition=1CN192C4G1H_MI300A_Ubuntu22
#SBATCH --exclusive

# AAC6 GT4Py-timer-only run, parallel to beverin's sbatch_gt4py_timer_rocm72.sh.
# No rocprof-compute phase — that's a separate sbatch.
#
# Pinned to ROCm 7.2.0 to match the beverin uenv major.minor.
# The setup_env.sh in $HOME/icon4py_debug/ does `module load rocm` (no version),
# which would default to 7.2.0 — but we pin explicitly to be safe.

set -eu

# --- modules ---
# Load rocm 7.2.0 explicitly. setup_env.sh below also runs `module load rocm`
# (no version) which would re-pick the default; the order matters less because
# the second load just confirms the same module, but pinning here makes intent
# explicit.
# Note: `cupy` is auto-loaded by aac6's prolog and is missing from `module avail`
# on at least ppac-pl1-s24-16 — explicit `module load cupy` errors there.
# Don't reload; the prolog-loaded version is what `python -c "import cupy"` uses.
module load rocm/7.2.0 openmpi mpi4py rocprofiler-compute/7.2.0

# --- env (sources $HOME/icon4py_debug/setup_env.sh) ---
source $HOME/icon4py_debug/setup_env.sh

# --- working dir ---
cd $HOME/icon4py

if [ "${ICON4PY_ENV_SOURCED:-}" != "1" ]; then
    echo "ERROR: Environment not set up. setup_env.sh did not export ICON4PY_ENV_SOURCED." >&2
    exit 1
fi

if [ "$(basename "$PWD")" != "icon4py" ] || [ ! -f pyproject.toml ]; then
    echo "ERROR: Must be run from \$HOME/icon4py. Current directory: $PWD" >&2
    exit 1
fi

# --- GT4Py / pytest config ---
# Match beverin's sbatch_gt4py_timer_rocm72.sh exactly so the only intentional
# differences between the two runs are cluster-side (hardware, partition mode,
# power, node sharing, ROCm patch rev).
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR:-amd_aac6_baseline_v1}
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
# HIPFLAGS already set by setup_env.sh on aac6 to:
#   -std=c++17 -fPIC -O3 --offload-arch=gfx942 -Wno-unused-parameter
# Beverin's sbatch_gt4py_timer_rocm72.sh has been aligned to match.

export ICON_GRID="icon_benchmark_regional"

# --- sanity dump ---
echo "=== env summary ==="
echo "host:         $(hostname)"
echo "rocm module:  $(module list 2>&1 | grep -oE 'rocm/[0-9.]+' | head -1)"
echo "ROCM_VERSION: ${ROCM_VERSION:-unset}"
echo "ROCM_PATH:    ${ROCM_PATH:-unset}"
echo "python:       $(command -v python) ($(python --version 2>&1))"
echo "hipcc:        $(command -v hipcc)"
echo "hip ver:      $(hipcc --version 2>&1 | head -2 | tail -1)"
echo "gt4py:        $(python -c 'import gt4py; print(gt4py.__file__)' 2>/dev/null)"
echo "dace:         $(python -c 'import dace; print(dace.__version__)' 2>/dev/null)"
echo "icon4py HEAD: $(git log --oneline -1)"
echo "model_options gpu_block_size_2d:"
grep -n "gpu_block_size_2d" model/common/src/icon4py/model/common/model_options.py | sed 's/^/  /'
echo "HIPFLAGS:     ${HIPFLAGS:-unset}"
echo "cache dir:    $GT4PY_BUILD_CACHE_DIR"
echo "==================="

# --- single-stencil GT4Py timer run ---
# Look for "GT4Py Timer Report" in the output; that's the headline number.
#
# run_with_patch.py monkey-patches cupy's NVRTC compile to inject:
#   1. -DHIP_DISABLE_WARP_SYNC_BUILTINS in compile options
#   2. #include <cupy/hip_workaround.cuh> at the top of every kernel source
# Required to work around aac6's CuPy 14.0.1 + ROCm 7.2 hiprtc_runtime.h
# 64-bit-mask static_assert. See $HOME/run_with_patch.py for details.
python3 $HOME/run_with_patch.py -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"
