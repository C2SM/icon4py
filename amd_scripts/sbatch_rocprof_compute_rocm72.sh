#!/bin/bash
#SBATCH --job-name=solver_rocprof_rocm72
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi300
#SBATCH --exclusive
#SBATCH --uenv=b2550889de318ab5
#SBATCH --view=default

# Run rocprof-compute on the dycore solver on beverin to collect per-kernel
# pmc_perf.csv. Mirrors aac6's job-rocprof-compute-improved.sh logic but
# uses the icon4py-rocm72 venv + uenv.
#
# Pin to a specific node and submit:
#   sbatch --nodelist=nid002510 \
#       --output=rocprof_nid002510.out --error=rocprof_nid002510.err \
#       amd_scripts/sbatch_rocprof_compute_rocm72.sh
#
# Output: rocprof-compute/rcu_<cache_dir>/MI300A_A1/pmc_perf.csv

set -eu

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$ICON4PY_GIT_ROOT"

if [ "$(basename "$PWD")" != "icon4py-rocm72" ]; then
    echo "ERROR: expected to be in icon4py-rocm72/, got $(basename "$PWD")" >&2
    exit 1
fi

source amd_scripts/setup_env.sh
source .venv_rocm72/bin/activate

# Cache dir defaults to amd_rocm72_pmc_<host> so each pinned-node run gets
# its own dir.
export GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR:-amd_rocm72_pmc_$(hostname -s)}
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export HIPFLAGS="-std=c++17 -fPIC -O3 --offload-arch=gfx942 -Wno-unused-parameter"
export ICON_GRID="icon_benchmark_regional"

OUTPUT_DIR=$ICON4PY_GIT_ROOT/rocprof-compute

# rocprof-compute needs the amdsmi Python module which isn't in the venv.
# Add the uenv-installed amdsmi package to PYTHONPATH (glob handles spack hash).
AMDSMI_PY_DIR=$(ls -d /user-environment/linux-zen3/amdsmi-*/share/amd_smi 2>/dev/null | head -1)
if [ -n "$AMDSMI_PY_DIR" ]; then
    export PYTHONPATH="${AMDSMI_PY_DIR}:${PYTHONPATH:-}"
    echo "# amdsmi found at: $AMDSMI_PY_DIR"
else
    echo "# WARN: amdsmi not found under /user-environment/linux-zen3/" >&2
fi

# --- env summary ---
echo "=== env summary ==="
echo "host:         $(hostname)"
echo "uenv:         ${UENV_VIEW:-${UENV_MOUNT_LIST:-unknown}}"
echo "ROCM_VERSION: ${ROCM_VERSION:-unset}"
echo "python:       $(command -v python)"
echo "hipcc:        $(command -v hipcc)"
echo "gt4py:        $(python -c 'import gt4py; print(gt4py.__file__)' 2>/dev/null)"
echo "cache dir:    $GT4PY_BUILD_CACHE_DIR"
echo "==================="

# --- phase 1: warmed-up benchmark run, populates the build cache ---
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100

.venv_rocm72/bin/python -m pytest -sv \
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
    .venv_rocm72/bin/python -m pytest -sv \
        -m continuous_benchmarking \
        -p no:tach \
        --backend=dace_gpu \
        --grid=${ICON_GRID} \
        model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
        -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

echo "# pmc_perf.csv: ${OUTPUT_DIR}/rcu_${GT4PY_BUILD_CACHE_DIR}/MI300A_A1/pmc_perf.csv"
