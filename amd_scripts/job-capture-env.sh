#!/bin/bash

#SBATCH --gpus=1

#SBATCH --time=00:10:00

#SBATCH --ntasks=1

#SBATCH --output=capture_env.out

#SBATCH --error=capture_env.err

#SBATCH --partition=1CN192C4G1H_MI300A_Ubuntu22

#SBATCH --exclusive

# cupy is auto-loaded by the prolog; explicit `module load cupy` errors today
# (module file missing from MODULEPATH on aac6, harmless without `set -eu`).
module load rocm/7.2.1 openmpi mpi4py rocprofiler-compute

source $HOME/icon4py_debug/setup_env.sh

cd $HOME/icon4py

if [ "${ICON4PY_ENV_SOURCED:-}" != "1" ]; then
    echo "ERROR: Environment not set up. Source setup_env.sh first." >&2
    exit 1
fi

if [ "$(basename "$PWD")" != "icon4py" ] || [ ! -f pyproject.toml ]; then
    echo "ERROR: Must be run from \$HOME/icon4py. Current directory: $PWD" >&2
    exit 1
fi

bash amd_scripts/capture_env.sh
