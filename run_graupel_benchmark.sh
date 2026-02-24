#!/bin/bash
#SBATCH --job-name=graupel_benchmark
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi300


source .venv/bin/activate

source amd_scripts/setup_env.sh

PREFIX="beverin_test"

export GT4PY_BUILD_CACHE_LIFETIME=PERSISTENT
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export PYTHONOPTIMIZE=2
export GT4PY_BUILD_CACHE_DIR="gt4py_cache_${PREFIX}"
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"

GRID="/capstor/store/cscs/userlab/d126/muphys_grids/inputs/atm_R2B06.nc" #mbe2605_atm_3d_input_20200122T000000Z.nc" #atm_R2B06.nc" #atm_R2B07_maxFrac.nc"
SIM_ITERATIONS=100
BENCHMARK_ITERATIONS=10

pytest --pdb -v model/atmosphere/subgrid_scale_physics/muphys/tests/muphys/integration_tests/test_graupel_only.py --backend=dace_gpu -k mini

for i in $(seq 1 ${BENCHMARK_ITERATIONS})
do
    echo "Benchmark iteration ${i} / ${BENCHMARK_ITERATIONS}"

    ITERATION_PREFIX="${PREFIX}_iter${i}"
    export GT4PY_BUILD_CACHE_DIR="gt4py_cache_${ITERATION_PREFIX}"

    python model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/driver/run_graupel_only.py -o junk_${ITERATION_PREFIX}.nc -b dace_gpu ${GRID} ${SIM_ITERATIONS}
    # rocprofv3 --kernel-trace on --hip-trace on --marker-trace on --memory-copy-trace on --memory-allocation-trace on --output-format pftrace -o rocprofv3_${GT4PY_BUILD_CACHE_DIR} -- \
    # python model/atmosphere/subgrid_scale_physics/muphys/src/icon4py/model/atmosphere/subgrid_scale_physics/muphys/driver/run_graupel_only.py -o junk_${ITERATION_PREFIX}.nc -b dace_gpu ${GRID} ${SIM_ITERATIONS}

    rm junk_${ITERATION_PREFIX}.nc
done

