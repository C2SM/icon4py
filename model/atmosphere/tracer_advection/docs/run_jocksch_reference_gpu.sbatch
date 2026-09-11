#!/bin/bash
# Run the Jocksch-cylinder reference tests (L1/L2/trajectory, test_jocksch_reference.py) on
# one GPU backend, on santis (uenv icon/26.7:v1, husk sandbox):
#
#   sbatch --partition=debug run_jocksch_reference_gpu.sbatch gtfn_gpu          # contracted (default)
#   sbatch --partition=debug run_jocksch_reference_gpu.sbatch dace_gpu nofma    # FMA contraction off
#
# The second argument selects the floating-point contraction of the build ('fma', the
# compilers' default, or 'nofma', see tests/tracer_advection/conftest.py); each (backend,
# mode) pair gets its own build cache, since the cache key ignores compiler flags.
# One backend per job: a first-time build of the driver's static-field factories plus the
# advection stencils has to fit the debug partition's 30 min (a cold gtfn_cpu build of the
# same test takes 6 min with 8 build jobs on the login node). Every environment variable is set
# inside the script (the broker does not forward the submitting shell's exports); the uenv
# is the one husk forces (printed in the job banner). Adjust ROOT to the checkout. The log
# is weno_data/slurm/<jobid>.out; husk forces stderr (its banner, tracebacks, pytest
# warnings) to <workspace>/slurm-<jobid>.err. Inside the compute cage the home directory
# is hidden and everything but the workspace is read-only, so the job uses a copy of the
# 'uv' binary staged in weno_data/bin/ (cp -L ~/.local/bin/uv weno_data/bin/) and a
# TMPDIR inside weno_data if the node's /tmp is not writable.
#
# The job is a pytest run, so the submitter holds the workspace's pytest lock for the job's
# whole lifetime (notes/workflow.md: one pytest / GT4Py execution at a time), not just for
# the sbatch call; no --wait (the broker refuses it), poll instead:
#
#   WS=<workspace>
#   until mkdir $WS/weno_data/pytest.lock 2>/dev/null; do sleep 30; done
#   echo "<agent> sbatch run_jocksch_reference_gpu.sbatch <backend>" > $WS/weno_data/pytest.lock/owner
#   cd $WS   # husk confines --output to the submitting directory and below: submit from the
#            # workspace root, not from icon4py/, or the broker refuses weno_data/slurm/%j.out
#   id=$(sbatch --partition=debug --parsable icon4py/model/atmosphere/tracer_advection/docs/run_jocksch_reference_gpu.sbatch <backend>)
#   until ! squeue -j $id -h | grep -q .; do sleep 30; done
#   rm -rf $WS/weno_data/pytest.lock
#
# If the job times out (30 min), resubmit: the persistent build cache resumes the build
# where it stopped.
#SBATCH --job-name=jocksch_reference_gpu
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/cmueller/tracer_advection_port/icon-exclaim/weno_data/slurm/%j.out
set -euo pipefail

BACKEND=${1:-gtfn_gpu}
MODE=${2:-fma}
ROOT=/capstor/scratch/cscs/cmueller/tracer_advection_port/icon-exclaim
WENO_DATA=$ROOT/weno_data
cd $ROOT/icon4py

export PATH=$WENO_DATA/bin:$PATH
command -v uv >/dev/null || { echo "uv not found: stage it with 'cp -L ~/.local/bin/uv $WENO_DATA/bin/'" >&2; exit 127; }
if ! [ -w "${TMPDIR:-/tmp}" ]; then
    export TMPDIR=$WENO_DATA/tmp
    mkdir -p $TMPDIR
fi
export ICON4PY_TEST_DATA_PATH=$WENO_DATA/testdata
export UV_PYTHON=$PWD/.venv/bin/python
export UV_PYTHON_INSTALL_DIR=${TMPDIR:-$WENO_DATA}/uvpy
export UV_CACHE_DIR=$WENO_DATA/uv_cache
export CXX=/user-environment/env/default/bin/g++
export CC=/user-environment/env/default/bin/gcc
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_JOBS=8
# fallback if the parallel build dies with BrokenProcessPool (memory pressure):
# export GT4PY_BUILD_JOBS_MODE=serial
case $MODE in
    fma) export GT4PY_BUILD_CACHE_DIR=$WENO_DATA/gt4py_cache/$BACKEND ;;
    nofma)
        export ICON4PY_FP_CONTRACT_OFF=1
        export GT4PY_BUILD_CACHE_DIR=$WENO_DATA/gt4py_cache/${BACKEND}_nofma
        ;;
    *) echo "unknown mode '$MODE' (fma | nofma)" >&2; exit 2 ;;
esac

echo "backend $BACKEND, mode $MODE, cache $GT4PY_BUILD_CACHE_DIR, uv $(command -v uv), TMPDIR ${TMPDIR:-unset}"
nvidia-smi -L || true
date
uv run --group test --frozen pytest -n0 -v -s --backend=$BACKEND --benchmark-disable \
    model/atmosphere/tracer_advection/tests/tracer_advection/integration_tests/test_jocksch_reference.py
date
