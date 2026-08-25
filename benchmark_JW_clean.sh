#!/bin/bash
#SBATCH --job-name=wall_JW4Py
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
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

# export ICON_GRID="./icon_grid_0025_R02B08_G.nc"
export ICON_GRID="./icon_grid_0004_R02B07_G.nc"
# export ICON_GRID="./icon_grid_0021_R02B06_G.nc"
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

export GT4PY_BUILD_CACHE_DIR_PREFIX="GH200_JW_128x2_256x1_${SUFFIX}"
export VERSION_SUFFIX="v2"

export GT4PY_SKIP_DACE_WARNINGS=0

export LD_LIBRARY_PATH=$(pwd):${LD_LIBRARY_PATH}

echo "Executing JW4Py on ${SLURM_NNODES} GH200 nodes to check the WALL CLOCK timer reported at the end of the run"

export GT4PY_BUILD_CACHE_DIR="${GT4PY_BUILD_CACHE_DIR_PREFIX}_${VERSION_SUFFIX}"
export OUTPUT_PATH=$(pwd)/standalone_driver_output_${GT4PY_BUILD_CACHE_DIR}_wall
srun -u --cpu-bind=cores \
    bash -c 'printenv TMPDIR; export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}; export DACE_compiler_build_folder_mode="development"; echo "SLURM_LOCALID: ${SLURM_LOCALID}: GPU ${CUDA_VISIBLE_DEVICES}"; icon4py-driver \
    --config-file-path exclaim_nh35_tri_jws_r2b7_${SLURM_NNODES}nodes \
    --grid-file-path $(realpath ${ICON_GRID}) \
    --icon4py-backend dace_gpu \
    --log-level ${ICON4PY_DRIVER_LOGGING_LEVEL} \
    --output-path ${OUTPUT_PATH} \
    --no-enable-output'

rm -rf ${OUTPUT_PATH}