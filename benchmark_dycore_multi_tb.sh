#!/bin/bash
#SBATCH --job-name=wall_JW4Py
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
#SBATCH --uenv=icon/26.7:v1@santis,/capstor/scratch/cscs/ioannmag/cycle37/icon4py/py313_venv_determ.squashfs:/capstor/scratch/cscs/ioannmag/cycle37/icon4py/.venv
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
export CUDAFLAGS="--generate-line-info -Xcompiler -g -O3 -Xcompiler -march=native -Xcompiler -Wno-unused-parameter"

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export DACE_compiler_build_folder_mode="development"
export PYTHONOPTIMIZE=2

export ICON4PY_DRIVER_LOGGING_LEVEL="warning"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user-environment/linux-neoverse_v2/nvhpc-26.1-eyhld4lfk55ld66egsyukpzmvejqlqa2/Linux_aarch64/26.1/compilers/lib
export CC=$(which gcc)
export CXX=$(which g++)
export MPICH_CC=$(which gcc)
export MPICH_CXX=$(which g++)
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=NVIDIA
export GHEX_GPU_ARCH=90
export GHEX_TRANSPORT_BACKEND=MPI
export CUDA_CACHE_DISABLE=1

export GT4PY_BUILD_CACHE_DIR_PREFIX="GH200_dycore_${SUFFIX}_1rank"
export VERSION_SUFFIX="v1"
export GT4PY_SKIP_DACE_WARNINGS=0

export LD_LIBRARY_PATH=$(pwd):${LD_LIBRARY_PATH}

echo "Executing dycore on ${SLURM_NNODES} GH200 nodes"

PIDS=()

# Launch 4 concurrent benchmark processes, one per GPU.
for worker_id in 0 1 2 3; do
    (
        if [ "$worker_id" == "0" ]; then
            export GT4PY_BLOCK_SIZE_HEURISTICS="5"
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="64,1,1"
            export GT4PY_CACHE_SUFFIX="H5_64x1_${VERSION_SUFFIX}"
        elif [ "$worker_id" == "1" ]; then
            export GT4PY_BLOCK_SIZE_HEURISTICS="3"
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="64,1,1"
            export GT4PY_CACHE_SUFFIX="H3_64x1_${VERSION_SUFFIX}"
        elif [ "$worker_id" == "2" ]; then
            export GT4PY_BLOCK_SIZE_HEURISTICS="0"
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="128,2,1"
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="64,1,1"
            export GT4PY_CACHE_SUFFIX="128x2_64x1_${VERSION_SUFFIX}"
        elif [ "$worker_id" == "3" ]; then
            export GT4PY_BLOCK_SIZE_HEURISTICS="0"
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="128,2,1"
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
            export GT4PY_CACHE_SUFFIX="128x2_256x1_${VERSION_SUFFIX}"
        fi
        export CUDA_VISIBLE_DEVICES="${worker_id}"
        export GT4PY_BUILD_CACHE_DIR="${GT4PY_BUILD_CACHE_DIR_PREFIX}_${GT4PY_CACHE_SUFFIX}"
        export ICON_GRID="icon_benchmark_regional"
        export GT4PY_COLLECT_METRICS_LEVEL=10
        export DYCORE_GT4PY_PROGRAMS_TIMER_FILE="dycore_gt4py_program_metrics_${GT4PY_BUILD_CACHE_DIR}.json"

        # Run the benchmark
        pytest -sv \
            -m continuous_benchmarking \
            -p no:tach \
            --benchmark-only \
            --benchmark-warmup=on \
            --benchmark-warmup-iterations=30 \
            --backend=dace_gpu \
            --grid=${ICON_GRID} \
            --benchmark-time-unit=ms \
            --benchmark-min-rounds 100 \
            model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]
    ) &

    PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

exit "${failed}"
