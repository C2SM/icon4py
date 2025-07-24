#!/bin/bash

#=============================================================================

#SBATCH --account=cwd01

#SBATCH --nodes=1
#SBATCH --uenv=icon/25.2:v3
#SBATCH --view=default

#SBATCH --partition=debug
#SBATCH --time=00:30:00

#SBATCH --job-name=channel_950x350x100_5m_nlev20_leeMoser

#SBATCH --output=../runs_icon4py/logs/torus.%x.log
#SBATCH --error=../runs_icon4py/logs/torus.%x.log

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
	export SCRATCH=/scratch/l_jcanton
	export PROJECTS_DIR=/home/l_jcanton/projects
	export ICON4PY_BACKEND="gtfn_cpu"
	;;
mac)
	export SCRATCH=/Users/jcanton/projects
	export PROJECTS_DIR=/Users/jcanton/projects
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

export ICON4PY_OUTPUT_DIR=$SCRATCH/runs_icon4py/$SLURM_JOB_NAME
export ICON4PY_SAVEPOINT_PATH="ser_data/exclaim_channel_950x350x100_5m_nlev20/ser_data"
export ICON4PY_GRID_FILE_PATH="testdata/grids/gauss3d_torus/Channel_950m_x_350m_res5m.nc"
export TOTAL_WORKERS=$((SLURM_NNODES * SLURM_TASKS_PER_NODE))

python \
	model/driver/src/icon4py/model/driver/icon4py_driver.py \
	$ICON4PY_SAVEPOINT_PATH \
	--icon4py_driver_backend="$ICON4PY_BACKEND" \
	--experiment_type=gauss3d_torus \
	--grid_root=2 --grid_level=0 --enable_output

## generate vtu files
#python ../python-scripts/plot_vtk.py "$TOTAL_WORKERS" "$ICON4PY_OUTPUT_DIR" "$ICON4PY_SAVEPOINT_PATH" "$ICON4PY_GRID_FILE_PATH"

echo "Finished running job: $SLURM_JOB_NAME, one way or another"
