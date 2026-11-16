#!/bin/bash
#SBATCH --job-name=venv_squashfs_mpi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
##SBATCH --uenv=prgenv-gnu/7.2.3:v1
##SBATCH --uenv=passthrough
#SBATCH -A csstaff
##SBATCH --view=default
#SBATCH --uenv-passthrough=use
#SBATCH --partition mi300
#SBATCH --exclude=nid[002670,002746,002764,002346,002728]
#SBATCH --reservation=cscs

INITIAL_DIR=$(pwd)
export UV_CACHE_DIR="/dev/shm/$USER/.uv_cache"

source setup_amd_env.sh

# unset PYTHONPATH and set PYTHONUSERBASE to avoid conflicts
unset PYTHONPATH
export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"

# create and activate a new relocatable venv using uv
# in this case we explicitly select the python interpreter from the uenv view
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy /dev/shm/$USER/.venv_rocm
cd /dev/shm/$USER
source .venv_rocm/bin/activate

cd $INITIAL_DIR
export CC="$(which clang)"
export MPICH_CC="$(which clang)"
export CXX="$(which clang++)"
export MPICH_CXX="$(which clang++)"
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=AMD
export GHEX_GPU_ARCH=gfx942
export GHEX_TRANSPORT_BACKEND=MPI
# uv cache clean pymetis # if pymetis is not built with CC/CXX gcc/g++
uv cache clean ghex
uv sync --no-binary-package mpi4py --extra all --extra distributed --extra rocm7 --python $(which python) --refresh --active

cd ../gt4py
uv pip install -e .
cd ..
cd dace
uv pip install -e .
cd ..
cd $INITIAL_DIR

# optionally, to reduce the import times, precompile all
# python modules to bytecode before creating the squashfs image
python -m compileall -j 8 -o 0 -o 1 -o 2 /dev/shm/$USER/.venv_rocm/lib/python3.12/site-packages

mksquashfs /dev/shm/$USER/.venv_rocm py_venv_rocm_mpi4.squashfs \
    -no-recovery -noappend -Xcompression-level 3
