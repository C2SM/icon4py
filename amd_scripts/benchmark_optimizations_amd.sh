#!/bin/bash
#SBATCH --job-name=dycore_optimizations
#SBATCH --nodes=1
#SBATCH --output=dycore_optimizations_%j_%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --uenv=prgenv-gnu/7.2.3:2804758683
#SBATCH --view=default
#SBATCH -A csstaff
#SBATCH --partition=mi300
#
# Regional/120, original code versus theta compiler fusion and both fused solvers.
set -Eeo pipefail
PHASE="startup"
trap 'rc=$?; if (( rc != 0 )); then echo "FAILED: exit ${rc}; phase=${PHASE}; output=${OUT_DIR:-Slurm stdout}" >&2; fi' EXIT
echo "dycore_optimizations starting: job=${SLURM_JOB_ID:-unset}, node=${SLURM_JOB_NODELIST:-unset}"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit with sbatch on Beverin." >&2
    exit 2
fi
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
REPO_ROOT="$(git -C "${SUBMIT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
PHASE="environment setup"
# This shared setup contains optional discovery pipelines whose nonzero status
# is expected. Disable errexit only while sourcing it, then validate the
# returned status and required tools. This also works with macOS Bash 3.2.
# Read the exact recorded setup without adding it to the review diff.
SETUP_FILE="$(mktemp)"
git show cdc034acb:setup_amd_env.sh > "${SETUP_FILE}"
set +e
source "${SETUP_FILE}"
SETUP_RC=$?
rm -f "${SETUP_FILE}"
set -e
if (( SETUP_RC != 0 )); then
    echo "Could not source setup_amd_env.sh." >&2
    exit 2
fi
PHASE="venv_mi300 activation"
if [[ ! -r venv_mi300/bin/activate ]]; then
    echo "Missing ${REPO_ROOT}/venv_mi300/bin/activate." >&2
    exit 2
fi
source venv_mi300/bin/activate
trap 'rc=$?; echo "ERROR: ${BASH_SOURCE[0]}:${LINENO}, exit ${rc}" >&2' ERR
echo "Environment ready: ${REPO_ROOT}"
set -u

PHASE="runtime environment"
TASK_SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}}"
export CUPY_CACHE_DIR="${TASK_SCRATCH}/.cupy/kernel_cache"
export XDG_CACHE_HOME="${TASK_SCRATCH}/.cache"
export UV_CACHE_DIR="${TASK_SCRATCH}/.cache/uv"
mkdir -p "${CUPY_CACHE_DIR}" "${XDG_CACHE_HOME}"
export ROCM_PATH="${ROCM_PATH:-${ROCM_HOME}}"
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export DACE_compiler_build_folder_mode=development
export GT4PY_ADD_GPU_TRACE_MARKERS=1
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"
export GT4PY_COLLECT_METRICS_LEVEL=10
export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592
export PYTHONOPTIMIZE=2
export PYTHONHASHSEED=0
EXPERIMENT_ROOT="${REPO_ROOT}/amd_scripts"
export PYTHONPATH="${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export GT4PY_BUILD_JOBS=0
export GT4PY_BUILD_JOBS_MODE=serial
export DACE_compiler_cuda_chiplet_number=1
export HIPARCHS=gfx942
export CAUSAL_CMAKE_TIMEOUT_SECONDS=1200
OUT_DIR="${REPO_ROOT}/amd_scripts/optimization_runs/amd_${SLURM_JOB_ID}"
PHASE="validated granule timing"
python "${EXPERIMENT_ROOT}/run_optimization_benchmark.py" --platform amd --comparison "${OPTIMIZATION_COMPARISON:-combined}" --output "${OUT_DIR}"
echo "PASS: ${OUT_DIR}/results/TIMING_SUMMARY.md"
