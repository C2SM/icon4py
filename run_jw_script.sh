#! /bin/bash
#SBATCH --job-name=icon4py_jw
#SBATCH --output=/capstor/scratch/cscs/cong/icon4py/output_%j/log_stdout.txt
#SBATCH --error=/capstor/scratch/cscs/cong/icon4py/output_%j/log
#SBATCH --account=cwd01
#SBATCH --uenv=icon/26.7:v1:/user-environment
#SBATCH --view=default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
###SBATCH --gpus-per-task=1
###SBATCH --gres-flags=allow-task-sharing
#SBATCH --partition=normal
#SBATCH --time=2:00:00

# gpus-per-node=4 means 4 GPUs per node, and 1 task per node, so each task gets 4 GPUs.

# santis build command:
# GHEX_USE_GPU=ON GHEX_GPU_TYPE=NVIDIA GHEX_GPU_ARCH=90 GHEX_TRANSPORT_BACKEND=MPI CC=$(which gcc) CXX=$(which g++) MPICH_CXX=$(which g++) MPICH_CC=$(which gcc) uv sync --no-binary-package mpi4py --extra all --extra distributed --extra cuda13 --python $(which python) --no-cache

# source /capstor/scratch/cscs/cong/icon4py/.venv/bin/activate
source /capstor/scratch/cscs/cong/icon4py/.venv/bin/activate || { echo "Venv activation failed"; exit 1; }

export GT4PY_BUILD_CACHE_LIFETIME=PERSISTENT
export GT4PY_BUILD_CACHE_DIR=/capstor/scratch/cscs/cong/tmp_cpu/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nvhpc-25.1-tsfur7lqj6njogdqafhpmj5dqltish7t/Linux_aarch64/25.1/compilers/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user-environment/linux-neoverse_v2/nvhpc-26.1-eyhld4lfk55ld66egsyukpzmvejqlqa2/Linux_aarch64/26.1/compilers/lib
export CC=$(which gcc)
export CXX=$(which g++)
export MPICH_CC=$(which gcc)
export MPICH_CXX=$(which g++)
export MPICH_GPU_SUPPORT_ENABLED=1
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=NVIDIA
export GHEX_GPU_ARCH=90
export GHEX_TRANSPORT_BACKEND=MPI
export PYTHONOPTIMIZE=2
# export INPUT_GRID=/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0010_R02B04_G.nc
# export INPUT_GRID=/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0008_R02B05_G.nc
export INPUT_GRID=/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0002_R02B06_G.nc
# export INPUT_GRID=/capstor/store/cscs/userlab/cwd01/cong/grids/icon_grid_0004_R02B07_G.nc
export CONFIG_FILE=/capstor/scratch/cscs/cong/icon4py/model/testing/src/icon4py/model/testing/experiment_configs/jablonowski_williamson.yaml

#  --print-distributed-debug-msg
export CUPY_CACHE_DIR=/capstor/scratch/cscs/cong/test_repo/cupy_cache/rank_$SLURM_PROCID;
srun --cpu-bind=cores bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; /capstor/scratch/cscs/cong/icon4py/.venv/bin/python /capstor/scratch/cscs/cong/icon4py/model/driver/src/icon4py/model/driver/main.py --config-file-path $CONFIG_FILE --grid-file-path $INPUT_GRID --icon4py-backend gtfn_gpu --log-level warning; true'

