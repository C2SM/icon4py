#!/bin/bash
#SBATCH --job-name=thetarho_benchmark
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi300
#SBATCH --uenv=prgenv-gnu/7.2.3:2579601092@beverin%mi200
#SBATCH --view=default
##SBATCH --nodelist=nid002410
#
# Default uenv above is the ROCm 7.1 image. To run under a different ROCm
# uenv (e.g. ROCm 7.2 for testing), override at submission time:
#   sbatch --uenv=<image-id> --view=default amd_scripts/benchmark_solver.sh
# install_icon4py_venv.sh and setup_env.sh both auto-detect the active
# ROCm version, so the same .venv works across uenvs as long as you
# rebuild via install_icon4py_venv.sh after switching.

# Go to the root of the icon4py repository to run the script from there
ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

# Set necessasry flags for compilation
source amd_scripts/setup_env.sh

source .venv/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=amd_profiling_thetarho_regional_rocm723_dacecodegen2
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"

export ICON_GRID="icon_benchmark_regional" # TODO(CSCS): Check also `icon_benchmark_global` when the dycore GPU memory issue is fixed

# Run the benchmark and collect the runtime of the whole GT4Py program (see `GT4Py Timer Report` in the output)
# The compiled GT4Py programs will be cached in the directory specified by `GT4PY_BUILD_CACHE_DIR` to be reused for running the profilers afterwards
.venv/bin/python -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_compute_theta_rho_face_values_and_pressure_gradient_and_update_vn.py \
    -k "test_TestComputeThetaRhoPressureGradientAndUpdateVn[is_iau_active[False]-compile_time_domain]"

# Run the benchmark and collect its trace
# TODO(AMD): Generating `rocpd` output fails with segfaults
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=30
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
# Can also add `--att` for thread tracing
rocprofv3 --kernel-trace on --hip-trace on --marker-trace on --memory-copy-trace on --memory-allocation-trace on --output-format csv -o rocprofv3_${GT4PY_BUILD_CACHE_DIR} -- \
    .venv/bin/python -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_compute_theta_rho_face_values_and_pressure_gradient_and_update_vn.py \
    -k "test_TestComputeThetaRhoPressureGradientAndUpdateVn[is_iau_active[False]-compile_time_domain]"
# Alternatively, export the data to `csv` and print kernel runtimes with the following command
python amd_scripts/median_rocprof_csv.py rocprofv3_${GT4PY_BUILD_CACHE_DIR}_kernel_trace.csv

# Get the kernel names of the GT4Py program so that we can filter them with rocprof-compute
LAST_COMPILED_DIRECTORY=$(realpath $(ls -td ${GT4PY_BUILD_CACHE_DIR}/.gt4py_cache/*/ | head -1))
echo "# Last compiled GT4Py directory: $LAST_COMPILED_DIRECTORY"
LAST_COMPILED_KERNEL_NAMES=$(grep -r -e "__global__ void.*map.*(" ${LAST_COMPILED_DIRECTORY}/src/cuda -o | sed 's/.*\s\([a-zA-Z_][a-zA-Z0-9_]*\)(.*/\1/')
echo "# Last compiled GT4Py kernel names:"
echo "$LAST_COMPILED_KERNEL_NAMES"
ROCPROF_COMPUTE_KERNEL_NAME_FILTER="-k $LAST_COMPILED_KERNEL_NAMES"

# Run rocprof-compute filtering the kernels of interest
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=0
export ICON4PY_STENCIL_TEST_ITERATIONS=1
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=1
rocprof-compute profile --name rcu_${GT4PY_BUILD_CACHE_DIR} ${ROCPROF_COMPUTE_KERNEL_NAME_FILTER} --format-rocprof-output rocpd --kernel-names -R FP64 --device 0 -- \
    .venv/bin/python -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_compute_theta_rho_face_values_and_pressure_gradient_and_update_vn.py \
    -k "test_TestComputeThetaRhoPressureGradientAndUpdateVn[is_iau_active[False]-compile_time_domain]"
