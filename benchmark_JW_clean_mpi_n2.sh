#!/bin/bash
#SBATCH --job-name=JW4Py_mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
#SBATCH --uenv=icon/26.7:v1@santis,/capstor/scratch/cscs/ioannmag/cycle38/icon4py-new/py_venv_mpi4.squashfs:/capstor/scratch/cscs/ioannmag/cycle38/icon4py-new/.venv
#SBATCH -A csstaff
#SBATCH --view=default
#SBATCH --partition=normal

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

source .venv/bin/activate

# Options from ICON
export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=0
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=disabled

export CUDA_CACHE_DISABLE=1
export CUPY_CACHE_IN_MEMORY=1

export MPICH_GPU_SUPPORT_ENABLED=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
# export DACE_compiler_build_folder_mode="development"
export PYTHONOPTIMIZE=2
# export GT4PY_ADD_COMPILATION_TRACE_MARKERS=1

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
# export ICON_GRID="./icon_grid_0054_R02B08_G.nc"
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

# export ICON4PY_PROFILE_MARKERS=nvtx
# export GT4PY_COLLECT_METRICS_LEVEL=10
# export GT4PY_ADD_GPU_TRACE_MARKERS="1"

GIBS_EXTERNAL=48
export ICON4PY_BACKEND_WORKSPACE_SIZE=$((${GIBS_EXTERNAL}*1073741824)) # 64Gib external workspace storage per device

export GT4PY_BUILD_CACHE_DIR_PREFIX="GH200_JW_external${GIBS_EXTERNAL}Gib_mpi_${SUFFIX}_120lvl_${SLURM_NNODES}nodes"
export VERSION_SUFFIX="v5"

export GT4PY_SKIP_DACE_WARNINGS=0

export LD_LIBRARY_PATH=$(pwd):${LD_LIBRARY_PATH}

echo "Executing JW4Py on ${SLURM_NNODES} GH200 nodes to check the WALL CLOCK timer reported at the end of the run (MPI)"

export GT4PY_BUILD_CACHE_DIR="${GT4PY_BUILD_CACHE_DIR_PREFIX}_${VERSION_SUFFIX}"
export OUTPUT_PATH=$(pwd)/standalone_driver_output_${GT4PY_BUILD_CACHE_DIR}_wall
export CONFIG_DIR="exclaim_nh35_tri_jws_r2b7_120lev_${SLURM_NNODES}nodes"

echo "Variables used for this run:"
printenv | grep -E '^(SLURM_|ICON4PY_|GT4PY_|GHEX_|NCCL_|CUDA_|CUPY_|FI_)' | sort

srun -u --cpu-bind=cores --gres-flags=allow-task-sharing \
    bash -c 'printenv TMPDIR; export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}; echo "SLURM_LOCALID: ${SLURM_LOCALID}: GPU ${CUDA_VISIBLE_DEVICES}"; \
    icon4py-driver \
    --config-file-path ${CONFIG_DIR}/config.yml \
    --grid-file-path $(realpath ${ICON_GRID}) \
    --icon4py-backend dace_gpu \
    --log-level ${ICON4PY_DRIVER_LOGGING_LEVEL} \
    --output-path ${OUTPUT_PATH} \
    --no-enable-output'

rm -rf ${OUTPUT_PATH}

