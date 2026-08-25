#!/bin/bash
#SBATCH --job-name=solver_benchmark_mi300
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --uenv=prgenv-gnu/7.2.3:2579601092
#SBATCH --view=default
#SBATCH -A csstaff
#SBATCH --partition mi300

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

source setup_amd_env.sh

source venv_mi300/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_CACHE_SUFFIX="v2"
export GT4PY_BUILD_CACHE_DIR=mi300_solver_global_amdheurfixes2_daceXCD_sweep_${GT4PY_CACHE_SUFFIX}
export DACE_compiler_build_folder_mode="development"
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"
export PYTHONOPTIMIZE=2
export DACE_compiler_cuda_chiplet_number=6

export ICON_GRID="icon_benchmark_global"

export ICON4PY_BACKEND_WORKSPACE_SIZE=8589934592 # 8Gib external workspace storage per device

rocprofv3 --kernel-trace on --hip-trace on --marker-trace on --memory-copy-trace on --memory-allocation-trace on --output-format pftrace -o vids_predictor_${GT4PY_CACHE_SUFFIX} -- \
    pytest -sv \
    -m continuous_benchmarking \
    -p no:tach \
    --benchmark-only \
    --benchmark-warmup=on \
    --benchmark-warmup-iterations=30 \
    --backend=dace_gpu \
    --grid=${ICON_GRID} \
    --benchmark-time-unit=ms \
    --benchmark-min-rounds=100 \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[at_first_substep[False]__is_iau_active[False]__divdamp_type[32]-compile_time_domain]"
