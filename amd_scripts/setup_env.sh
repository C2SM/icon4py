export CC="$(which gcc)"
export MPICH_CC="$(which gcc)"
export CXX="$(which g++)"
export MPICH_CXX="$(which g++)"
export HUGETLB_ELFMAP="no"
export HUGETLB_MORECORE="no"
export PYTHONOPTIMIZE="2"
export HCC_AMDGPU_TARGET="gfx942"
export ROCM_HOME="/user-environment/env/default"
export HIPCC="$(which hipcc)"

# Auto-detect ROCm version from hipcc instead of hardcoding.
# hipcc --version prints e.g. "HIP version: 7.2.26103-9999" → "7.2.26103".
# Falls back to "unknown" if hipcc isn't on PATH (uenv not started).
if command -v hipcc >/dev/null 2>&1; then
    ROCM_VERSION=$(hipcc --version 2>/dev/null | awk -F'[ -]' '/^HIP version:/ {print $3; exit}')
    export ROCM_VERSION="${ROCM_VERSION:-unknown}"
else
    export ROCM_VERSION="unknown"
fi

# Auto-detect rocprofiler-dev lib path. Versioned spack hash differs per uenv,
# so glob to whatever exists under /user-environment/linux-zen3/.
ROCPROF_DEV_LIB=$(ls -d /user-environment/linux-zen3/rocprofiler-dev-*/lib 2>/dev/null | head -1)
if [ -n "$ROCPROF_DEV_LIB" ]; then
    export LD_LIBRARY_PATH="${ROCPROF_DEV_LIB}:${LD_LIBRARY_PATH:-}"
fi
# export LD_LIBRARY_PATH=rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux:$LD_LIBRARY_PATH # TODO(iomaganaris): Add package to uenv properly
export LD_PRELOAD=/user-environment/env/default/lib/libomp.so:${LD_PRELOAD:-}
