#!/bin/bash

set -e

date

# Go to the root of the icon4py repository to run the installation from there
ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

# Set necessasry flags for compilation
source $ICON4PY_GIT_ROOT/amd_scripts/setup_env.sh

# Install uv locally
export PATH="$PWD/bin:$PATH"
if [ ! -x "$PWD/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="$PWD/bin" sh
else
    echo "# uv already installed at $PWD/bin/uv"
fi

# Install icon4py, gt4py, DaCe and other basic dependencies using uv
uv sync --extra rocm7 --python $(which python3.12)

# Activate virtual environment
source .venv/bin/activate

# Install the requirements for rocprofiler-compute so we can run the profiler from the same environment
uv pip install -r /user-environment/linux-zen3/rocprofiler-compute-7.1.0-rjjjgkz67w66bp46jw7bvlfyduzr6vhv/libexec/rocprofiler-compute/requirements.txt

# Patch CuPy hip_workaround.cuh: force mask-stripping for __shfl_*_sync on all ROCm versions.
# CuPy 14.0.1 passes a 32-bit mask to __shfl_xor_sync which requires 64-bit on ROCm 7.0+.
# Stripping the mask is safe because AMD wavefronts execute in lock-step.
# See: https://github.com/cupy/cupy/pull/9748
CUPY_HIP_WORKAROUND=$(python3 -c "import cupy, os; print(os.path.join(os.path.dirname(cupy.__file__), '_core', 'include', 'cupy', 'hip_workaround.cuh'))" 2>/dev/null)
if [ -f "$CUPY_HIP_WORKAROUND" ]; then
    sed -i 's/#if (HIP_VERSION < 60200000) || defined(HIP_DISABLE_WARP_SYNC_BUILTINS)/#if 1  \/\/ Patched: force mask-stripping for all ROCm versions (CuPy 14.0.1 bug)/' "$CUPY_HIP_WORKAROUND"
    echo "# Patched CuPy hip_workaround.cuh: $CUPY_HIP_WORKAROUND"
fi

echo "# install done"
date
