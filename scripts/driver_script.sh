#! /bin/bash
#SBATCH --job-name=icon4py_driver_r2b9_64nodes
#SBATCH --output=/capstor/scratch/cscs/cong/run/r2b9_64nodes/output_%j/log_stdout.txt
#SBATCH --error=/capstor/scratch/cscs/cong/run/r2b9_64nodes/output_%j/log
#SBATCH --account=cwd01
#SBATCH --uenv=icon/25.2:v3:/user-environment
#SBATCH --view=default
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --time=12:00:00


# santis build command:
# GHEX_USE_GPU=ON GHEX_GPU_TYPE=NVIDIA GHEX_GPU_ARCH=90 GHEX_TRANSPORT_BACKEND=MPI CC=$(which gcc) CXX=$(which g++) MPICH_CXX=$(which g++) MPICH_CC=$(which gcc) uv sync --no-binary-package mpi4py --extra all --extra distributed --extra cuda12 --python $(which python) --no-cache

source /capstor/scratch/cscs/cong/icon4py/.venv/bin/activate

export GT4PY_BUILD_CACHE_LIFETIME=PERSISTENT
export GT4PY_BUILD_CACHE_DIR=/capstor/scratch/cscs/cong/tmp_gpu/
# export GT4PY_BUILD_CACHE_DIR=$PWD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nvhpc-25.1-tsfur7lqj6njogdqafhpmj5dqltish7t/Linux_aarch64/25.1/compilers/lib
export CC=$(which gcc)
export CXX=$(which g++)
export MPICH_CC=$(which gcc)
export MPICH_CXX=$(which g++)
export MPICH_GPU_SUPPORT_ENABLED=1
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=NVIDIA
export GHEX_GPU_ARCH=90
export GHEX_TRANSPORT_BACKEND=MPI
export PYTHONOPTIMIZE=1
export OUTPUT_PATH=/capstor/scratch/cscs/cong/run/r2b9_64nodes/output
export INPUT_GRID=/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0015_R02B09_G.nc

python /capstor/scratch/cscs/cong/icon4py/model/standalone_driver/src/icon4py/model/standalone_driver/main.py --output-path $OUTPUT_PATH --grid-file-path $INPUT_GRID --icon4py-backend gtfn_gpu --log-level warning

