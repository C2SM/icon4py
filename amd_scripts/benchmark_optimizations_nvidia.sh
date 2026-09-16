#!/bin/bash
#SBATCH --job-name=dycore_optimizations
#SBATCH --nodes=1
#SBATCH --output=dycore_optimizations_%j_%N.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --uenv=icon/26.7:v1@santis
#SBATCH -A csstaff
#SBATCH --view=default
#SBATCH --partition=normal
#
# Regional/120: original versus combined theta and solver fusion.
set -Eeo pipefail
PHASE="startup"
trap 'rc=$?; if (( rc != 0 )); then echo "FAILED: exit ${rc}; phase=${PHASE}; output=${OUT_DIR:-Slurm stdout}" >&2; fi' EXIT
trap 'rc=$?; echo "ERROR: ${BASH_SOURCE[0]}:${LINENO}, exit ${rc}" >&2' ERR
echo "dycore_optimizations starting: job=${SLURM_JOB_ID:-unset}, node=${SLURM_JOB_NODELIST:-unset}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch on Santis." >&2
    exit 2
fi
# Slurm executes a spool copy, so BASH_SOURCE points outside the checkout.
PHASE="repository lookup"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
if ! REPO_ROOT="$(git -C "${SUBMIT_DIR}" rev-parse --show-toplevel)"; then
    echo "Submit from the icon4py checkout; cannot locate it from ${SUBMIT_DIR}." >&2
    exit 2
fi
cd "${REPO_ROOT}"
PHASE="venv_gh200 activation"
if [[ ! -f venv_gh200/bin/activate ]]; then
    echo "Missing ${REPO_ROOT}/venv_gh200/bin/activate." >&2
    exit 2
fi
source venv_gh200/bin/activate
echo "Environment ready: ${REPO_ROOT}"
set -u

PHASE="runtime environment"
TASK_SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}}"
export CUPY_CACHE_DIR="${TASK_SCRATCH}/.cupy/kernel_cache"
export XDG_CACHE_HOME="${TASK_SCRATCH}/.cache"
export UV_CACHE_DIR="${TASK_SCRATCH}/.cache/uv"
mkdir -p "${CUPY_CACHE_DIR}" "${XDG_CACHE_HOME}"
export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=0
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=disabled
export MPICH_GPU_SUPPORT_ENABLED=1
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export DACE_compiler_build_folder_mode=development
export PYTHONOPTIMIZE=2
export PYTHONHASHSEED=0
export GT4PY_COLLECT_METRICS_LEVEL=10
export ICON4PY_DRIVER_LOGGING_LEVEL=warning
export LD_LIBRARY_PATH="${REPO_ROOT}:/user-environment/linux-neoverse_v2/nvhpc-26.1-eyhld4lfk55ld66egsyukpzmvejqlqa2/Linux_aarch64/26.1/compilers/lib:${LD_LIBRARY_PATH:-}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export MPICH_CC="${CC}"
export MPICH_CXX="${CXX}"
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=NVIDIA
export GHEX_GPU_ARCH="80;90"
export GHEX_TRANSPORT_BACKEND=MPI
export CUDA_CACHE_DISABLE=1
export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592
export GT4PY_SKIP_DACE_WARNINGS=0
EXPERIMENT_ROOT="${REPO_ROOT}/amd_scripts"
export PYTHONPATH="${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export GT4PY_ADD_GPU_TRACE_MARKERS=1
export GT4PY_BUILD_JOBS=0
export GT4PY_BUILD_JOBS_MODE=serial
export CAUSAL_CMAKE_TIMEOUT_SECONDS=1200
OUT_DIR="${REPO_ROOT}/amd_scripts/optimization_runs/nvidia_${SLURM_JOB_ID}"
PHASE="validated granule timing"
python "${EXPERIMENT_ROOT}/run_optimization_benchmark.py" --platform nvidia --comparison "${OPTIMIZATION_COMPARISON:-combined}" --output "${OUT_DIR}"
echo "PASS: ${OUT_DIR}/results/TIMING_SUMMARY.md"
