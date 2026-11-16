#!/bin/bash
#SBATCH --job-name=venv_squashfs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
##SBATCH --uenv=icon/26.7:v1@santis
#SBATCH -A csstaff
##SBATCH --view=default
#SBATCH --uenv-passthrough=use
#SBATCH --partition=normal
#SBATCH --reservation maintenance

INITIAL_DIR=$(pwd)
export UV_CACHE_DIR="/dev/shm/$USER/.uv_cache"

# unset PYTHONPATH and set PYTHONUSERBASE to avoid conflicts
unset PYTHONPATH
export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"

# create and activate a new relocatable venv using uv
# in this case we explicitly select the python interpreter from the uenv view
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy /dev/shm/$USER/.venv_cuda
cd /dev/shm/$USER
source .venv_cuda/bin/activate

cd $INITIAL_DIR
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=NVIDIA
export GHEX_GPU_ARCH="80;90"
export GHEX_TRANSPORT_BACKEND=MPI
export CC=$(which gcc)
export CXX=$(which g++)
export MPICH_CXX=$(which g++)
export MPICH_CC=$(which gcc)
# uv cache clean pymetis # if pymetis is not built with CC/CXX gcc/g++
uv sync --no-binary-package mpi4py --extra all --extra distributed --extra cuda13 --python $(which python) --refresh --active

cd ../gt4py
uv pip install -e .
cd ..
cd dace
uv pip install -e .
cd ..
cd ${INITIAL_DIR}

# optionally, to reduce the import times, precompile all
# python modules to bytecode before creating the squashfs image
python -m compileall -j 8 -o 0 -o 1 -o 2 /dev/shm/$USER/.venv_cuda/lib/python3.13/site-packages

mksquashfs /dev/shm/$USER/.venv_cuda py_venv_cuda_mpi4.squashfs \
    -no-recovery -noappend -Xcompression-level 3
