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
# is weno_data/slurm/<jobid>.out.
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

export ICON4PY_TEST_DATA_PATH=$WENO_DATA/testdata
export UV_PYTHON=$PWD/.venv/bin/python
export UV_PYTHON_INSTALL_DIR=${TMPDIR:-/tmp}/uvpy
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

echo "backend $BACKEND, mode $MODE, cache $GT4PY_BUILD_CACHE_DIR"
nvidia-smi -L || true
date
uv run --group test --frozen pytest -n0 -v -s --backend=$BACKEND --benchmark-disable \
    model/atmosphere/tracer_advection/tests/tracer_advection/integration_tests/test_jocksch_reference.py
date
