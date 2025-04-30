#! /bin/bash
#=============================================================================

# balfrin cpu batch job parameters
# ------------------------------
#SBATCH --partition=debug
#SBATCH --time=0:30:00
#SBATCH --job-name=torus_test
#SBATCH --output=logs/torus_%j.o
#SBATCH --error=logs/torus_%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# OpenMP environment variables
# ----------------------------
export OMP_NUM_THREADS=1

module use /mch-environment/v5/modules
module load netcdf-c/4.8.1-nvhpc
module load hdf5/1.12.2-nvhpc
module load python/3.10.8
module load gcc/11.3.0
module load cuda/11.8.0-nvhpc
export BOOST_ROOT=/scratch/mch/cong/spack-c2sm/spack/opt/spack/linux-sles15-zen3/gcc-11.3.0/boost-1.83.0-irmtcaa6tdqltmaibi7xectycnhowpex/

# Cuda architecture
export CUDAARCHS="80"

# uv sync --extra cuda11

# uenv
# # Path to your boost installation
# export BOOST_ROOT=/user-environment/env/icon/include/boost/
# # Path to your cuda installation
# export CUDA_PATH=/user-environment/linux-sles15-zen3/gcc-12.3.0/cuda-12.3.0-45bjh5clzy5kol7ymjjqzshbn7ymxriu
# # Path to shared libraries in user environment
# export LD_LIBRARY_PATH=/user-environment/env/icon/lib64:$LD_LIBRARY_PATH

rm imgs/*

# source .venv/bin/activate

srun -n 1 --ntasks-per-node 1 --threads-per-core=1 python model/driver/src/icon4py/model/driver/icon4py_driver.py testdata/ser_icondata/mpitask1/gauss3d_torus/ser_data --icon4py_driver_backend=gtfn_gpu --experiment_type=gauss3d_torus --grid_root=2 --grid_level=0 --enable_output
