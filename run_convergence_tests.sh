#! /bin/bash
#SBATCH --job-name=icon4py_convergence
#SBATCH --output=/capstor/scratch/cscs/cong/icon4py/convergence_%j/log_stdout.txt
#SBATCH --error=/capstor/scratch/cscs/cong/icon4py/convergence_%j/log
#SBATCH --account=cwd01
#SBATCH --uenv=icon/26.7:v1:/user-environment
#SBATCH --view=default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --time=4:00:00

# Runs the tracer advection convergence tests
# (model/driver/tests/driver/validation_tests/test_tracer_advection_convergence.py) and keeps
# their errors: every test case writes <test>_<experiment_case>.json into
# convergence_<jobid>/errors, then plot_tracer_advection_convergence.py plots them into
# convergence_<jobid>/plots. The pytest result (pass/fail of the order checks) is in the log;
# the errors are written even when a check fails.
#
# Submit from the repo root:  sbatch run_convergence_tests.sh
# Optional: BACKEND=gtfn_cpu sbatch run_convergence_tests.sh
#           TESTS="-k horizontal" sbatch run_convergence_tests.sh

REPO=/capstor/scratch/cscs/cong/icon4py
cd $REPO || exit 1
source $REPO/.venv/bin/activate || { echo "Venv activation failed"; exit 1; }

export GT4PY_BUILD_CACHE_LIFETIME=PERSISTENT
export GT4PY_BUILD_CACHE_DIR=/capstor/scratch/cscs/cong/tmp_cpu/
export GT4PY_BUILD_JOBS=16
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user-environment/linux-neoverse_v2/nvhpc-26.1-eyhld4lfk55ld66egsyukpzmvejqlqa2/Linux_aarch64/26.1/compilers/lib
export CC=$(which gcc)
export CXX=$(which g++)
export PYTHONOPTIMIZE=2
export CUPY_CACHE_DIR=/capstor/scratch/cscs/cong/test_repo/cupy_cache/rank_0

OUT=$REPO/convergence_${SLURM_JOB_ID:-local}
export ICON4PY_CONVERGENCE_OUTPUT_DIR=$OUT/errors
mkdir -p $ICON4PY_CONVERGENCE_OUTPUT_DIR
BACKEND=${BACKEND:-gtfn_gpu}
echo "backend $BACKEND, errors -> $ICON4PY_CONVERGENCE_OUTPUT_DIR"

srun --cpu-bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID; \
  python -m pytest -n0 -v -rA --benchmark-disable --level validation --backend $BACKEND $TESTS \
  model/driver/tests/driver/validation_tests/test_tracer_advection_convergence.py"
echo "pytest exit code $?"

python $REPO/plot_tracer_advection_convergence.py $ICON4PY_CONVERGENCE_OUTPUT_DIR --out-dir $OUT/plots
