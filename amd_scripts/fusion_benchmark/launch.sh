#!/bin/bash
# Shared environment for the two user-submitted Slurm wrappers.
set -Eeo pipefail
PHASE=startup
trap 'rc=$?; if (( rc != 0 )); then echo "FAILED: exit ${rc}; phase=${PHASE}" >&2; fi' EXIT
trap 'rc=$?; echo "ERROR: ${BASH_SOURCE[0]}:${LINENO}, exit ${rc}" >&2' ERR
PLATFORM="${1:?Expected amd or nvidia}"
shift
: "${SLURM_JOB_ID:?Run inside a GPU allocation}"
echo "dycore_fusion: platform=${PLATFORM} job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST:-unknown}"
PHASE=environment
VENV_PATH="${VENV_PATH:-${VIRTUAL_ENV:-}}"
if [[ ! -r "${VENV_PATH}/bin/activate" ]]; then
    echo "Set VENV_PATH to the GPU venv containing both PR checkouts." >&2
    exit 2
fi
source "${VENV_PATH}/bin/activate"
set -u
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
export MPICH_CC="${CC}" MPICH_CXX="${CXX}"
export PYTHONHASHSEED=0 PYTHONOPTIMIZE=2
export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${SCRATCH:?}/.cupy/kernel_cache}"
mkdir -p "${CUPY_CACHE_DIR}"
if [[ "${PLATFORM}" == amd ]]; then
    export ROCM_PATH="${ROCM_PATH:-${ROCM_HOME:-/user-environment/env/default}}"
    export ROCM_HOME="${ROCM_PATH}"
    export HIPCC="${HIPCC:-$(command -v hipcc)}"
    export HIPARCHS=gfx942 HCC_AMDGPU_TARGET=gfx942 CUPY_ACCELERATORS=cub
    export HUGETLB_ELFMAP=no HUGETLB_MORECORE=no
    export DACE_compiler_cuda_chiplet_number=1
    export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"
    # Match HIPRTC's C++ headers to the selected compiler in the CSCS uenv.
    CXX_HEADERS=$("${CXX}" -x c++ -E -v /dev/null 2>&1 | sed -n '/#include <...> search starts here:/,/End of search list/p' | sed -n '/\/c++\//s/^ *//p' | paste -sd: -)
    if [[ -n "${CXX_HEADERS}" ]]; then
        export CPLUS_INCLUDE_PATH="${CXX_HEADERS}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
    fi
    GCC_INSTALL_DIR=$(dirname "$("${CXX}" -print-libgcc-file-name)")
    export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=${GCC_INSTALL_DIR}${HIPCC_COMPILE_FLAGS_APPEND:+ ${HIPCC_COMPILE_FLAGS_APPEND}}"
    if [[ -r "${ROCM_PATH}/lib/libomp.so" ]]; then
        export LD_PRELOAD="${ROCM_PATH}/lib/libomp.so${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
else
    export CUDA_CACHE_DISABLE=1 CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
    export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=0 FI_CXI_RX_MATCH_MODE=software FI_MR_CACHE_MONITOR=disabled
    export MPICH_GPU_SUPPORT_ENABLED=1
fi
PHASE=controlled_benchmark
python amd_scripts/fusion_benchmark/run.py --platform "${PLATFORM}" "$@"
PHASE=complete
echo "dycore_fusion completed."
