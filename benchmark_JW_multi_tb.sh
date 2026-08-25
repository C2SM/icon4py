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

# export ICON_GRID="./icon_grid_0025_R02B08_G.nc"
# export ICON_GRID="./icon_grid_0004_R02B07_G.nc"
export ICON_GRID="./icon_grid_0021_R02B06_G.nc"
# export ICON_GRID="./icon_grid_0013_R02B04_R.nc"
SUFFIX=""
if [[ "$ICON_GRID" == *"R02B04"* ]]; then
    SUFFIX="R02B04"
elif [[ "$ICON_GRID" == *"R02B06"* ]]; then
    SUFFIX="R02B06"
elif [[ "$ICON_GRID" == *"R02B07"* ]]; then
    SUFFIX="R02B07"
elif [[ "$ICON_GRID" == *"R02B08"* ]]; then
    SUFFIX="R02B08"
fi

export GT4PY_BUILD_CACHE_DIR_PREFIX="GH200_JW_${SUFFIX}_1rank"
export VERSION_SUFFIX="v5"
export GT4PY_SKIP_DACE_WARNINGS=0

export GT4PY_BLOCK_SIZE_HEURISTICS="5"

export LD_LIBRARY_PATH=$(pwd):${LD_LIBRARY_PATH}

echo "Executing JW4Py on ${SLURM_NNODES} GH200 nodes to check the WALL CLOCK timer reported at the end of the run"

PIDS=()

# Launch 4 concurrent benchmark processes, one per GPU.
for worker_id in 0 1 2 3; do
    (
        # if [ "$worker_id" == "0" ]; then
        #     export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="32,8,1"
        #     export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="64,1,1"
        #     export GT4PY_CACHE_SUFFIX="32x8_64x1_${VERSION_SUFFIX}"
        # elif [ "$worker_id" == "1" ]; then
        #     export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="64,4,1"
        #     export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="64,1,1"
        #     export GT4PY_CACHE_SUFFIX="64x4_64x1_${VERSION_SUFFIX}"
        # elif [ "$worker_id" == "2" ]; then
        #     export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="128,2,1"
        #     export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="64,1,1"
        #     export GT4PY_CACHE_SUFFIX="128x2_64x1_${VERSION_SUFFIX}"
        # elif [ "$worker_id" == "3" ]; then
        #     export ICON4PY_GPU_THREAD_BLOCK_SIZE_2D="256,1,1"
        #     export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="64,1,1"
        #     export GT4PY_CACHE_SUFFIX="256x1_64x1_${VERSION_SUFFIX}"
        # fi
        if [ "$worker_id" == "0" ]; then
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="32,1,1"
            export GT4PY_CACHE_SUFFIX="H5_32x1_${VERSION_SUFFIX}"
        elif [ "$worker_id" == "1" ]; then
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="64,1,1"
            export GT4PY_CACHE_SUFFIX="H5_64x1_${VERSION_SUFFIX}"
        elif [ "$worker_id" == "2" ]; then
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="128,1,1"
            export GT4PY_CACHE_SUFFIX="H5_128x1_${VERSION_SUFFIX}"
        elif [ "$worker_id" == "3" ]; then
            export ICON4PY_GPU_THREAD_BLOCK_SIZE_1D="256,1,1"
            export GT4PY_CACHE_SUFFIX="H5_256x1_${VERSION_SUFFIX}"
        fi
        export CUDA_VISIBLE_DEVICES="${worker_id}"
        export GT4PY_BUILD_CACHE_DIR="${GT4PY_BUILD_CACHE_DIR_PREFIX}_${GT4PY_CACHE_SUFFIX}"
        export OUTPUT_PATH=$(pwd)/standalone_driver_output_${GT4PY_BUILD_CACHE_DIR}_wall_${worker_id}
        echo "[worker ${worker_id}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR} ICON4PY_GPU_THREAD_BLOCK_SIZE_2D=${ICON4PY_GPU_THREAD_BLOCK_SIZE_2D} ICON4PY_GPU_THREAD_BLOCK_SIZE_1D=${ICON4PY_GPU_THREAD_BLOCK_SIZE_1D}"
        nsys profile -t cuda,nvtx,osrt -o ${GT4PY_BUILD_CACHE_DIR} --stats true -f true \
            icon4py-standalone-driver \
            --config-file-path exclaim_nh35_tri_jws_r2b7_${SLURM_NNODES}nodes \
            --grid-file-path $(realpath ${ICON_GRID}) \
            --icon4py-backend dace_gpu \
            --log-level ${ICON4PY_DRIVER_LOGGING_LEVEL} \
            --output-path ${OUTPUT_PATH} \
            --no-enable-output
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

rm -rf ${OUTPUT_PATH}