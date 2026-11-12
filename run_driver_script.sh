#! /bin/bash
#SBATCH --job-name=icon4py_warm_bubble
#SBATCH --output=/capstor/scratch/cscs/cong/icon4py/output_%j/log_stdout.txt
#SBATCH --error=/capstor/scratch/cscs/cong/icon4py/output_%j/log
#SBATCH --account=cwd01
#SBATCH --uenv=icon/26.7:v1:/user-environment
#SBATCH --view=default
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
###SBATCH --gpus-per-task=1
###SBATCH --gres-flags=allow-task-sharing
#SBATCH --partition=normal
#SBATCH --time=12:00:00


# santis build command:
# GHEX_USE_GPU=ON GHEX_GPU_TYPE=NVIDIA GHEX_GPU_ARCH=90 GHEX_TRANSPORT_BACKEND=MPI CC=$(which gcc) CXX=$(which g++) MPICH_CXX=$(which g++) MPICH_CC=$(which gcc) uv sync --no-binary-package mpi4py --extra all --extra distributed --extra cuda13 --python $(which python) --no-cache

# source /capstor/scratch/cscs/cong/icon4py/.venv/bin/activate
source /capstor/scratch/cscs/cong/icon4py/.venv/bin/activate || { echo "Venv activation failed"; exit 1; }

export GT4PY_BUILD_CACHE_LIFETIME=PERSISTENT
export GT4PY_BUILD_CACHE_DIR=/capstor/scratch/cscs/cong/tmp_gpu/
# export GT4PY_BUILD_CACHE_DIR=/capstor/scratch/cscs/cong/run/r2b6_2nodes/
# export GT4PY_BUILD_CACHE_DIR=$PWD
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
export INPUT_GRID=/capstor/scratch/cscs/cong/icon4py/testdata/grids/torus_100000mx100000m_res100m/torus_100000mx100000m_res100m.nc
# export INPUT_GRID=/capstor/scratch/cscs/cong/icon4py/testdata/grids/torus_100000mx100000m_res500m/torus_100000mx100000m_res500m.nc
# export INPUT_GRID=/capstor/scratch/cscs/cong/icon4py/testdata/grids/torus_100000mx100000m_res5000m/torus_100000mx100000m_res5000m.nc

#  --print-distributed-debug-msg
srun --cpu-bind=cores bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; /capstor/scratch/cscs/cong/icon4py/.venv/bin/python /capstor/scratch/cscs/cong/icon4py/model/driver/src/icon4py/model/driver/main.py --config-file-path /capstor/scratch/cscs/cong/icon4py/model/testing/src/icon4py/model/testing/experiment_configs/warm_bubble.yaml --grid-file-path $INPUT_GRID --icon4py-backend gtfn_gpu --log-level warning; true'
# srun --cpu-bind=cores --gpu-bind=single:1 bash -c '/capstor/scratch/cscs/cong/icon4py/.venv/bin/python /capstor/scratch/cscs/cong/icon4py/model/driver/src/icon4py/model/driver/main.py --config-file-path /capstor/scratch/cscs/cong/icon4py/model/testing/src/icon4py/model/testing/experiment_configs/warm_bubble.yaml --grid-file-path $INPUT_GRID --icon4py-backend gtfn_gpu --log-level warning; true'
# srun --cpu-bind=cores bash -c '/capstor/scratch/cscs/cong/icon4py/.venv/bin/python /capstor/scratch/cscs/cong/icon4py/model/driver/src/icon4py/model/driver/main.py --config-file-path /capstor/scratch/cscs/cong/icon4py/model/testing/src/icon4py/model/testing/experiment_configs/warm_bubble.yaml --grid-file-path $INPUT_GRID --icon4py-backend gtfn_cpu --log-level warning; true'

