#!/bin/bash
#SBATCH --job-name=solver_benchmark
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi300

# Go to the root of the icon4py repository to run the script from there
ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

# Set necessasry flags for compilation
source amd_scripts/setup_env.sh

source .venv/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=amd_profiling_solver_regional
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"

export ICON_GRID="icon_benchmark_regional" # TODO(CSCS): Check also `icon_benchmark_global` when the dycore GPU memory issue is fixed

# Run the benchmark and collect the runtime of the whole GT4Py program (see `GT4Py Timer Report` in the output)
# The compiled GT4Py programs will be cached in the directory specified by `GT4PY_BUILD_CACHE_DIR` to be reused for running the profilers afterwards
pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

# Run the benchmark and collect its trace
# TODO(AMD): Generating `rocpd` output fails with segfaults
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=30
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
# Can also add `--att` for thread tracing
rocprofv3 --kernel-trace on --hip-trace on --marker-trace on --memory-copy-trace on --memory-allocation-trace on --output-format pftrace -o rocprofv3_${GT4PY_BUILD_CACHE_DIR} -- \
    $(which python3.12) -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"
# Alternatively, export the data to `csv` and print kernel runtimes with the following command
# python amd_scripts/median_rocprof_csv.py rocprofv3_${GT4PY_BUILD_CACHE_DIR}_kernel_trace.csv

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
rocprof-compute profile --name rcu_${GT4PY_BUILD_CACHE_DIR} ${ROCPROF_COMPUTE_KERNEL_NAME_FILTER} --format-rocprof-output rocpd --kernel-names -R FP64 -- \
    $(which python3.12) -m pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

# TODO(AMD): Roofline generation fails with
#   File "/user-environment/linux-zen3/rocprofiler-compute-7.1.0-rjjjgkz67w66bp46jw7bvlfyduzr6vhv/libexec/rocprofiler-compute/roofline.py", line 998, in standalone_roofline
#    self.empirical_roofline(ret_df=t_df)
#  File "/user-environment/linux-zen3/rocprofiler-compute-7.1.0-rjjjgkz67w66bp46jw7bvlfyduzr6vhv/libexec/rocprofiler-compute/utils/logger.py", line 66, in wrap_function
#    result = function(*args, **kwargs)
#             ^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "/user-environment/linux-zen3/rocprofiler-compute-7.1.0-rjjjgkz67w66bp46jw7bvlfyduzr6vhv/libexec/rocprofiler-compute/roofline.py", line 463, in empirical_roofline
#    flops_figure.write_image(
#  File "/capstor/scratch/cscs/ioannmag/HPCAIAdvisory/icon4py/.venv/lib/python3.12/site-packages/plotly/basedatatypes.py", line 3895, in write_image
#    return pio.write_image(self, *args, **kwargs)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "/capstor/scratch/cscs/ioannmag/HPCAIAdvisory/icon4py/.venv/lib/python3.12/site-packages/plotly/io/_kaleido.py", line 555, in write_image
#    path.write_bytes(img_data)
#  File "/user-environment/linux-zen3/python-3.12.12-jpkfwhqo6njvbpw7gjcs22qkvxwexnv5/lib/python3.12/pathlib.py", line 1036, in write_bytes
#    with self.open(mode='wb') as f:
#         ^^^^^^^^^^^^^^^^^^^^
# File "/user-environment/linux-zen3/python-3.12.12-jpkfwhqo6njvbpw7gjcs22qkvxwexnv5/lib/python3.12/pathlib.py", line 1013, in open
#    return io.open(self, mode, buffering, encoding, errors, newline)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# OSError: [Errno 36] File name too long: '/capstor/scratch/cscs/ioannmag/HPCAIAdvisory/icon4py/workloads/rcu_amd_profiling_solver/MI300A_A1/empirRoof_gpu-0_FP64_map_0_fieldop_0_0_500_map_100_fieldop_0_0_0_514_map_100_fieldop_1_0_0_0_520_map_115_fieldop_0_0_0_516_map_115_fieldop_1_0_0_518_map_13_fieldop_0_0_498_map_31_fieldop_0_0_0_512_map_35_fieldop_0_0_503_map_60_fieldop_0_0_504_map_85_fieldop_0_0_506_map_90_fieldop_0_0_508_map_91_fieldop_0_0_510.pdf'
