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
