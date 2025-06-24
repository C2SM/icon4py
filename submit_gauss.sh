#!/bin/bash

#=============================================================================

#SBATCH --account=cwd01

#SBATCH --nodes=1
#SBATCH --uenv=icon/25.2:v3
#SBATCH --view=default

#SBATCH --time=00:30:00
#SBATCH --partition=debug

#SBATCH --job-name=runxx_test_wiggles

#SBATCH --output=logs/torus.runxx_test_wiggles.log
#SBATCH --error=logs/torus.runxx_test_wiggles.log

case $CLUSTER_NAME in
balfrin)
	export SCRATCH=/scratch/mch/jcanton
	export PROJECTS_DIR=$SCRATCH
	export ICON4PY_BACKEND="gtfn_gpu"
	module load gcc-runtime
	module load nvhpc
	;;
santis)
	export SCRATCH=/capstor/scratch/cscs/jcanton
	export PROJECTS_DIR=$SCRATCH
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nvhpc-25.1-tsfur7lqj6njogdqafhpmj5dqltish7t/Linux_aarch64/25.1/compilers/lib
	export ICON4PY_BACKEND="gtfn_gpu"
	;;
squirrel)
	export SCRATCH=/scratch/l_jcanton/
	export PROJECTS_DIR=/home/l_jcanton/projects/
	export ICON4PY_BACKEND="gtfn_cpu"
	;;
*)
	echo "cluster name not recognized: ${CLUSTER_NAME}"
	;;
esac
echo "Running on cluster: ${CLUSTER_NAME}"

export PYTHONOPTIMIZE=2
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=$SCRATCH/gt4py_cache

source "$PROJECTS_DIR/icon4py/.venv/bin/activate"

python \
	model/driver/src/icon4py/model/driver/icon4py_driver.py \
	ser_data/exclaim_gauss3d.uniform100_flat/ser_data \
	--icon4py_driver_backend=${ICON4PY_BACKEND} \
	--experiment_type=gauss3d_torus \
	--grid_root=2 --grid_level=0 --enable_output
