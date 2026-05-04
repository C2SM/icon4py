#!/bin/bash

#SBATCH --gpus=1

#SBATCH --time=01:00:00

#SBATCH --ntasks=4

#SBATCH --output=icon4py.out

#SBATCH --error=icon4py.err

#SBATCH --partition=1CN192C4G1H_MI300A_Ubuntu22

#SBATCH --exclusive

rm -rf $HOME/icon4py/amd_profiling_solver_regional/.gt4py_cache

# cupy is auto-loaded by the prolog; explicit `module load cupy` errors today
# (module file missing from MODULEPATH on aac6, harmless without `set -eu`).
module load rocm/7.2.1 openmpi mpi4py rocprofiler-compute

source $HOME/icon4py_debug/setup_env.sh

cd $HOME/icon4py

OUTPUT_DIR=$HOME/icon4py/rocprof-compute

sed -i '0,/optimization_args\["make_persistent"\] = True/s//&\n        optimization_args["gpu_block_size_2d"] =  (256, 1, 1)/' "$HOME/icon4py/model/common/src/icon4py/model/common/model_options.py"

if [ "${ICON4PY_ENV_SOURCED:-}" != "1" ]; then
    echo "ERROR: Environment not set up. Source setup_env.sh first." >&2
    exit 1
fi

if [ "$(basename "$PWD")" != "icon4py" ] || [ ! -f pyproject.toml ]; then
    echo "ERROR: Must be run from \$HOME/icon4py. Current directory: $PWD" >&2
    exit 1
fi

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=amd_profiling_solver_regional
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100

export ICON_GRID="icon_benchmark_regional"

python3 -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

# Get the kernel names of the GT4Py program so that we can filter them with rocprof-compute
LAST_COMPILED_DIRECTORY=$(realpath $(ls -td ${GT4PY_BUILD_CACHE_DIR}/.gt4py_cache/*/ | head -1))
echo "# Last compiled GT4Py directory: $LAST_COMPILED_DIRECTORY"
LAST_COMPILED_KERNEL_NAMES=$(grep -r -e "__global__ void.*map.*(" ${LAST_COMPILED_DIRECTORY}/src/cuda -o | sed 's/.*\s\([a-zA-Z_][a-zA-Z0-9_]*\)(.*/\1/')
echo "# Last compiled GT4Py kernel names:"
echo "$LAST_COMPILED_KERNEL_NAMES"
#ROCPROF_COMPUTE_KERNEL_NAME_FILTER="-k $LAST_COMPILED_KERNEL_NAMES"
ROCPROF_COMPUTE_KERNEL_NAME_FILTER="-k $(echo "$LAST_COMPILED_KERNEL_NAMES" | paste -sd'|')"

# Run rocprof-compute filtering the kernels of interest
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=0
export ICON4PY_STENCIL_TEST_ITERATIONS=1
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=1
rocprof-compute profile --no-roof --name rcu_${GT4PY_BUILD_CACHE_DIR} ${ROCPROF_COMPUTE_KERNEL_NAME_FILTER} --format-rocprof-output csv --kernel-names -R FP64 -p ${OUTPUT_DIR} -- \
    python3 -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

git checkout model/common/src/icon4py/model/common
