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
export CUPY_ACCELERATORS=cub

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

# HIPRTC/comgr's device-JIT compiler (used by cupy for on-the-fly kernel
# compilation, e.g. the first array reduction in grid setup) does not
# reliably auto-detect this uenv's GCC toolchain. It ends up including
# GCC 13's <initializer_list>/<cstddef> without finding the matching
# target-specific bits/c++config.h, so _GLIBCXX_VISIBILITY etc. are left
# undefined and the JIT compile fails with "expected '{'" / "no template
# named 'initializer_list'" errors for gfx942. Pin both the header search
# (CPLUS_INCLUDE_PATH) and comgr's own gcc-toolchain flag so it resolves
# to the exact same GCC install regardless of which internal codepath
# HIPRTC uses.
if command -v g++ >/dev/null 2>&1; then
    GXX_CXX_INCLUDE_DIRS=$(g++ -x c++ -E -v /dev/null 2>&1 \
        | sed -n '/#include <\.\.\.> search starts here:/,/End of search list/p' \
        | grep '/c++/' | sed 's/^ *//' | paste -sd: -)
    if [ -n "$GXX_CXX_INCLUDE_DIRS" ]; then
        export CPLUS_INCLUDE_PATH="${GXX_CXX_INCLUDE_DIRS}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    fi
    GCC_INSTALL_DIR=$(dirname "$(g++ -print-libgcc-file-name)")
    export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=${GCC_INSTALL_DIR}${HIPCC_COMPILE_FLAGS_APPEND:+ $HIPCC_COMPILE_FLAGS_APPEND}"
fi
export LD_PRELOAD=/user-environment/env/default/lib/libomp.so:${LD_PRELOAD:-}
