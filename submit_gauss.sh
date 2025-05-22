#! /bin/bash
#=============================================================================

# santis batch job parameters
# ------------------------------
#SBATCH --partition=debug
#SBATCH --time=0:30:00
#SBATCH --job-name=torus_test
#SBATCH --output=logs/torus.log
#SBATCH --error=logs/torus.log
#SBATCH --nodes=1
## #SBATCH --gpus-per-node=4
## #SBATCH --ntasks-per-node=2

export PYTHONOPTIMIZE=2
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=/capstor/scratch/cscs/jcanton/gt4py_cache
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nvhpc-25.1-tsfur7lqj6njogdqafhpmj5dqltish7t/Linux_aarch64/25.1/compilers/lib

source venv_uv/bin/activate

srun \
	python model/driver/src/icon4py/model/driver/icon4py_driver.py \
	/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform200_flat/ser_data \
	--icon4py_driver_backend=gtfn_gpu \
	--experiment_type=gauss3d_torus \
	--grid_root=2 --grid_level=0 --enable_output

# srun \
# 	pytest -sv \
# 	--backend=gtfn_gpu \
# 	model/driver/tests/driver_tests/test_timeloop.py
