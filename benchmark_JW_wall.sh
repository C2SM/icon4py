#!/bin/bash
#SBATCH --job-name=wall_JW4Py
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
#SBATCH --uenv=icon/25.2:v4
#SBATCH -A cwp03
#SBATCH --view=default

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
export PYTHONOPTIMIZE=2

export ICON4PY_DRIVER_LOGGING_LEVEL="warning"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nvhpc-25.1-tsfur7lqj6njogdqafhpmj5dqltish7t/Linux_aarch64/25.1/compilers/lib
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
export ICON_GRID="./icon_grid_0004_R02B07_G.nc"
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

export GT4PY_BUILD_CACHE_DIR="GH200_JW_${SUFFIX}_persistent_ntasks${SLURM_NTASKS}"

export GT4PY_SKIP_DACE_WARNINGS=0

export LD_LIBRARY_PATH=$(pwd):${LD_LIBRARY_PATH}

export OUTPUT_PATH=$(pwd)/standalone_driver_output_${GT4PY_BUILD_CACHE_DIR}_wall

echo "Executing JW4Py on ${SLURM_NNODES} GH200 nodes to check the WALL CLOCK timer reported at the end of the run"

rm -rf ${OUTPUT_PATH}*

srun -u --cpu-bind=cores \
    bash -c 'printenv TMPDIR; CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}; echo "SLURM_LOCALID: ${SLURM_LOCALID}: GPU ${CUDA_VISIBLE_DEVICES}"; icon4py-standalone-driver \
    --config-file-path exclaim_nh35_tri_jws_r2b7_${SLURM_NNODES}nodes \
    --grid-file-path $(realpath ${ICON_GRID}) \
    --icon4py-backend dace_gpu \
    --log-level ${ICON4PY_DRIVER_LOGGING_LEVEL} \
    --output-path ${OUTPUT_PATH} \
    --no-enable-output'

rm -rf ${OUTPUT_PATH}