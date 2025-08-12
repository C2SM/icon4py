#!/bin/bash

# =======================================
# USER-EDITABLE: Slurm job parameters
# =======================================
SLURM_ACCOUNT="cwd01"
SLURM_NODES=1

SLURM_UENV="icon/25.2:v3"
SLURM_UENV_VIEW="default"

SLURM_PARTITION="debug"
SLURM_TIME="00:30:00"

SLURM_JOBNAME="channel_950x350x100_5m_nlev20_leeMoser"
SLURM_LOGDIR="../runs_icon4py/logs"

# =======================================
# USER-EDITABLE: Default booleans
# Positional arguments can override:
#   ./job.sh [run_simulation] [run_postprocess]
#   sbatch job.sh [run_simulation] [run_postprocess]
# =======================================
run_simulation=false
run_postprocess=true

# Override defaults with positional args if provided
if [ -n "$1" ]; then run_simulation="$1"; fi
if [ -n "$2" ]; then run_postprocess="$2"; fi

# =======================================
# Wrapper: If not in Slurm, submit ourselves
# =======================================
if [ -z "$SLURM_JOB_ID" ]; then
	# Timestamp for unique log files
	timestamp=$(date +"%Y%m%d_%H%M%S")

	# Pick log suffix based on booleans
	if $run_simulation && $run_postprocess; then
		log_suffix="both"
	elif $run_simulation; then
		log_suffix="sim"
	elif $run_postprocess; then
		log_suffix="post"
	else
		log_suffix="idle"
	fi

	# Submit self to Slurm with parameters preserved
	sbatch \
		--account="$SLURM_ACCOUNT" \
		--nodes="$SLURM_NODES" \
		--uenv="$SLURM_UENV" \
		--view="$SLURM_UENV_VIEW" \
		--partition="$SLURM_PARTITION" \
		--time="$SLURM_TIME" \
		--job-name="$SLURM_JOBNAME" \
		--output="$SLURM_LOGDIR/%x_${log_suffix}_${timestamp}.log" \
		--error="$SLURM_LOGDIR/%x_${log_suffix}_${timestamp}.log" \
		"$0" "$run_simulation" "$run_postprocess"
	exit
fi

# ============================================================================
# Determine cluster name and options
#
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

# ==============================================================================
# Environment setup
#
export ICON4PY_OUTPUT_DIR=$SCRATCH/runs_icon4py/$SLURM_JOB_NAME
export ICON4PY_SAVEPOINT_PATH="ser_data/exclaim_channel_950x350x100_5m_nlev20/ser_data"
export ICON4PY_GRID_FILE_PATH="testdata/grids/gauss3d_torus/Channel_950m_x_350m_res5m.nc"
export TOTAL_WORKERS=$((SLURM_NNODES * SLURM_TASKS_PER_NODE))

export ICON4PY_DIR=$PROJECTS_DIR/icon4py.ibm
export SCRIPTS_DIR=$PROJECTS_DIR/python-scripts

echo ""
echo ""
echo "Running on cluster   = $CLUSTER_NAME"
echo ""
echo "SLURM_JOB_ID         = $SLURM_JOB_ID"
echo "run_simulation       = $run_simulation"
echo "run_postprocess      = $run_postprocess"
echo ""
echo "ICON4PY:"
echo "    backend          = $ICON4PY_BACKEND"
echo "    run directory    = $ICON4PY_DIR"
echo "    output directory = $ICON4PY_OUTPUT_DIR"
echo ""
echo "SCRIPTS:"
echo "    run directory    = $SCRIPTS_DIR"

# ==============================================================================
# Run simulation
#
if [ "$run_simulation" = true ]; then

	echo "[INFO] Running simulation..."

	cd "$ICON4PY_DIR" || exit

	source .venv/bin/activate

	export PYTHONOPTIMIZE=2
	export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
	export GT4PY_BUILD_CACHE_LIFETIME=persistent
	export GT4PY_BUILD_CACHE_DIR=$SCRATCH/gt4py_cache

	python \
		model/driver/src/icon4py/model/driver/icon4py_driver.py \
		$ICON4PY_SAVEPOINT_PATH \
		--icon4py_driver_backend="$ICON4PY_BACKEND" \
		--experiment_type=gauss3d_torus \
		--grid_root=2 --grid_level=0 --enable_output

fi

# ==============================================================================
# Postprocess
#
if [ "$run_postprocess" = true ]; then

	echo "[INFO] Running postprocess..."

	if [ -n "$VIRTUAL_ENV" ]; then
		# deactivate simulation venv if active
		deactivate
	fi
	source "$SCRIPTS_DIR/.venv/bin/activate"

	# generate vtu files
	python "$SCRIPTS_DIR/plot_vtk.py" "$TOTAL_WORKERS" "$ICON4PY_OUTPUT_DIR" "$ICON4PY_SAVEPOINT_PATH" "$ICON4PY_GRID_FILE_PATH"

	# compute temporal averages
	python "$SCRIPTS_DIR/temporal_average.py" "$TOTAL_WORKERS" "$ICON4PY_OUTPUT_DIR"

	echo "Finished running job: $SLURM_JOB_NAME"

fi
