#!/usr/bin/env bash

source setup_amd_env.sh

# Below are necessary for GHEX
export CC="$(which clang)"
export MPICH_CC="$(which clang)"
export CXX="$(which clang++)"
export MPICH_CXX="$(which clang++)"
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=AMD
export GHEX_GPU_ARCH=gfx942
export GHEX_TRANSPORT_BACKEND=MPI

python -m venv venv_mi300
source venv_mi300/bin/activate

uv sync --no-binary-package mpi4py --extra all --extra distributed --extra rocm7 --python $(which python) --refresh --active

# Patch CuPy hip_workaround.cuh: force mask-stripping for __shfl_*_sync on all ROCm versions.
# CuPy 14.0.1 passes a 32-bit mask to __shfl_xor_sync which requires 64-bit on ROCm 7.0+.
# Stripping the mask is safe because AMD wavefronts execute in lock-step.
# See: https://github.com/cupy/cupy/pull/9748
CUPY_HIP_WORKAROUND=$(python3 -c "import cupy, os; print(os.path.join(os.path.dirname(cupy.__file__), '_core', 'include', 'cupy', 'hip_workaround.cuh'))" 2>/dev/null)
if [ -f "$CUPY_HIP_WORKAROUND" ]; then
    sed -i 's/#if (HIP_VERSION < 60200000) || defined(HIP_DISABLE_WARP_SYNC_BUILTINS)/#if 1  \/\/ Patched: force mask-stripping for all ROCm versions (CuPy 14.0.1 bug)/' "$CUPY_HIP_WORKAROUND"
    echo "# Patched CuPy hip_workaround.cuh: $CUPY_HIP_WORKAROUND"
fi
