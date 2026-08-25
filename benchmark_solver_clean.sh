#!/bin/bash
#SBATCH --job-name=wall_JW4Py
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
##SBATCH --gres=gpu:1
#SBATCH --uenv=icon/26.7:v1@santis,/capstor/scratch/cscs/ioannmag/cycle38/icon4py/py_venv.squashfs:/capstor/scratch/cscs/ioannmag/cycle38/icon4py/.venv
#SBATCH -A csstaff
#SBATCH --view=default
#SBATCH --partition=normal

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

source .venv/bin/activate

export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=0
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=disabled
export MPICH_GPU_SUPPORT_ENABLED=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export DACE_compiler_build_folder_mode="development"
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export PYTHONOPTIMIZE=2

export ICON4PY_DRIVER_LOGGING_LEVEL="warning"

export LD_LIBRARY_PATH=/user-environment/linux-neoverse_v2/nvhpc-26.1-eyhld4lfk55ld66egsyukpzmvejqlqa2/Linux_aarch64/26.1/compilers/lib:$LD_LIBRARY_PATH
export CC=$(which gcc)
export CXX=$(which g++)
export MPICH_CC=$(which gcc)
export MPICH_CXX=$(which g++)
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=NVIDIA
export GHEX_GPU_ARCH="80;90"
export GHEX_TRANSPORT_BACKEND=MPI
export CUDA_CACHE_DISABLE=1

export ICON_GRID="icon_benchmark_global"

SUFFIX=""
if [[ "$ICON_GRID" == *"global"* ]]; then
    SUFFIX="global"
elif [[ "$ICON_GRID" == *"regional"* ]]; then
    SUFFIX="regional"
fi

export GT4PY_BUILD_CACHE_DIR_PREFIX="GH200_solver_128x2_256x1_${SUFFIX}"
export VERSION_SUFFIX="v2"

export GT4PY_SKIP_DACE_WARNINGS=0

export LD_LIBRARY_PATH=$(pwd):${LD_LIBRARY_PATH}

echo "Executing vertical solver on ${SLURM_NNODES} GH200 nodes to check the WALL CLOCK timer reported at the end of the run"

export GT4PY_BUILD_CACHE_DIR="${GT4PY_BUILD_CACHE_DIR_PREFIX}_${VERSION_SUFFIX}"
export GT4PY_METRICS_OUTPUT_PATH="gt4py_timers_${GT4PY_BUILD_CACHE_DIR}.json"

nsys profile -t cuda,nvtx,osrt -o ${GT4PY_BUILD_CACHE_DIR} --stats true -f true \
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
