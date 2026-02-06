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
uv sync --python $(which python3.12)

# Activate virtual environment
source .venv/bin/activate

# Install amd-cupy
uv pip install amd-cupy --extra-index-url https://pypi.amd.com/rocm-7.0.2/simple

# Install the requirements for rocprofiler-compute so we can run the profiler from the same environment
uv pip install -r /user-environment/linux-zen3/rocprofiler-compute-7.1.0-rjjjgkz67w66bp46jw7bvlfyduzr6vhv/libexec/rocprofiler-compute/requirements.txt

echo "# install done"
date
