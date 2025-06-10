#!/bin/bash

#=============================================================================

#SBATCH --nodes=1
#SBATCH --uenv=icon/25.2:1773507487
#SBATCH --view=default

#SBATCH --time=24:00:00
#SBATCH --partition=normal

#SBATCH --job-name=run62_barray_4x4_nlev800_pert

#SBATCH --output=logs/torus.run62_barray_4x4_nlev800_pert.log
#SBATCH --error=logs/torus.run62_barray_4x4_nlev800_pert.log

export PYTHONOPTIMIZE=2
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=/scratch/mch/jcanton/gt4py_cache
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nvhpc-25.1-tsfur7lqj6njogdqafhpmj5dqltish7t/Linux_aarch64/25.1/compilers/lib

#uenv status

module load gcc-runtime
module load nvhpc

source /scratch/mch/jcanton/icon4py/.venv/bin/activate
#source .venv/bin/activate

# which python
# realpath $(which python)
# python -c "import numpy; print (numpy.__file__)"
# python test_script.py

python \
	model/driver/src/icon4py/model/driver/icon4py_driver.py \
	/scratch/mch/jcanton/ser_data/exclaim_gauss3d.uniform800_flat/ser_data \
	--icon4py_driver_backend=gtfn_gpu \
	--experiment_type=gauss3d_torus \
	--grid_root=2 --grid_level=0 --enable_output

# srun \
# 	pytest -sv \
# 	--backend=gtfn_gpu \
# 	model/driver/tests/driver_tests/test_timeloop.py
