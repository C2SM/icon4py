#! /bin/bash
#=============================================================================

# santis batch job parameters
# ------------------------------
#SBATCH --partition=debug
#SBATCH --time=0:30:00
#SBATCH --job-name=torus_test
#SBATCH --output=logs/torus_log.o
#SBATCH --error=logs/torus_log.o
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
## #SBATCH --ntasks-per-node=1

# Cuda architecture
#export CUDAARCHS="90" # not needed anymore

# # uenv
# # Path to your boost installation
# export BOOST_ROOT=/user-environment/env/icon/include/boost/
# # Path to your cuda installation
# export CUDA_PATH=/user-environment/linux-sles15-zen3/gcc-12.3.0/cuda-12.3.0-45bjh5clzy5kol7ymjjqzshbn7ymxriu
# # Path to shared libraries in user environment
# export LD_LIBRARY_PATH=/user-environment/env/icon/lib64:$LD_LIBRARY_PATH

# export PATH=$HOME/.local/$(uname -m)/bin:$PATH
# unset -f uenv

export PYTHONOPTIMIZE=2
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=/capstor/scratch/cscs/jcanton/gt4py_cache

source .venv/bin/activate

srun python model/driver/src/icon4py/model/driver/icon4py_driver.py testdata/ser_icondata/mpitask1/gauss3d_torus/ser_data --icon4py_driver_backend=gtfn_gpu --experiment_type=gauss3d_torus --grid_root=2 --grid_level=0 --enable_output

#srun python model/driver/src/icon4py/model/driver/icon4py_driver.py testdata/ser_icondata/mpitask1/jabw_R02B04/ser_data --icon4py_driver_backend=gtfn_gpu --experiment_type=jablonowski_williamson --grid_root=2 --grid_level=0 --enable_output

#srun pytest -sv --backend=gtfn_gpu model/driver/tests/initial_condition_tests/test_jablonowski_williamson.py
