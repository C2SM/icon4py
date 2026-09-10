#!/bin/bash
# Run the Jocksch-cylinder reference tests (L1/L2/trajectory, test_jocksch_reference.py) on
# one GPU backend with FMA contraction off, on santis (uenv icon/26.7:v1, husk sandbox):
#
#   sbatch run_jocksch_reference_gpu.sbatch gtfn_gpu
#   sbatch run_jocksch_reference_gpu.sbatch dace_gpu
#
# One backend per job: a first-time stencil build of the driver's static-field factories
# plus the advection stencils does not fit two backends into the debug partition's 30 min.
# Every environment variable is set inside the script (the broker does not forward the
# submitting shell's exports). Adjust ROOT to the checkout.
#SBATCH --job-name=jocksch_reference_gpu
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/cmueller/tracer_advection_port/icon-exclaim/weno_data/slurm/%j.out
set -euo pipefail

BACKEND=${1:-gtfn_gpu}
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
export GT4PY_BUILD_JOBS_MODE=serial
# contraction off (see tests/tracer_advection/conftest.py); its own cache directory
export ICON4PY_FP_CONTRACT_OFF=1
export GT4PY_BUILD_CACHE_DIR=$WENO_DATA/gt4py_cache/${BACKEND}_nofma

echo "backend $BACKEND, cache $GT4PY_BUILD_CACHE_DIR"
nvidia-smi -L || true
uv run --group test --frozen pytest -n0 -v -s --backend=$BACKEND \
    model/atmosphere/tracer_advection/tests/tracer_advection/integration_tests/test_jocksch_reference.py
