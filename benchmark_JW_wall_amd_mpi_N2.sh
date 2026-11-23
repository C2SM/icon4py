#!/bin/bash
#SBATCH --job-name=JW4Py_mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
##SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --uenv=prgenv-gnu/7.2.3:2950075422,/ritom/scratch/cscs/ioannmag/cycle39/icon4py-benchmarks/py_venv_mpi4.squashfs:/ritom/scratch/cscs/ioannmag/cycle39/icon4py-benchmarks/.venv
#SBATCH -A csstaff
#SBATCH --view=default
#SBATCH --partition mi300
#SBATCH --reservation cscs
#SBATCH --exclude=nid[002668,002676],nid[002936,002956]

ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

source setup_amd_env.sh

source .venv/bin/activate

# ROCm + cray-mpich related options
export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_ENABLE_IPC_MODE_LEGACY=1
# Options from Santis
export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=0
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=disabled

# Avoid caching CUDA/COMGR kernels to avoid inode limit
export CUDA_CACHE_DISABLE=1
export AMD_COMGR_CACHE=0

# ICON4Py performance related options
# Python optimization level
export PYTHONOPTIMIZE=2
# External workspace storage per device (one large hip/cuMalloc allocation for temp variables in DaCe - necessary for MI300A perf)
GIBS_EXTERNAL=48
export ICON4PY_BACKEND_WORKSPACE_SIZE=$((${GIBS_EXTERNAL}*1073741824))
# Set the logging level for ICON4Py driver to warning to avoid printing overhead when measuring. Requires the changes in `show_timer_report`
export ICON4PY_DRIVER_LOGGING_LEVEL="warning"

# Below values necessary to get meaningful runtime per gt4py program
# export GT4PY_COLLECT_METRICS_LEVEL=10
# export GT4PY_DUMP_METRICS_AT_EXIT=1
# Add NVTX/ROCTX markers to each GT4Py program compiled with DaCe
# export GT4PY_ADD_GPU_TRACE_MARKERS="1"

# GT4Py compilation options
# Number of parallel compile jobs
export GT4PY_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-8}
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
# Keep DaCe src files
# export GT4PY_KEEP_DACE_BUILD_FILES=1
# Skip verbose DaCe warnings during compilation
export GT4PY_SKIP_DACE_WARNINGS=1

# export ICON_GRID="./icon_grid_0004_R02B08_G.nc"
# export ICON_GRID="./icon_grid_0013_R02B04_R.nc"
# export ICON_GRID="./icon_grid_0054_R02B08_G.nc"
export ICON_GRID="./icon_grid_0004_R02B07_G.nc"
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

export GT4PY_BUILD_CACHE_DIR="MI300A_JW_external${GIBS_EXTERNAL}GiB_mpi_${SUFFIX}_120lvl_ntasks${SLURM_NTASKS}"
export OUTPUT_PATH=$(pwd)/standalone_driver_output_${GT4PY_BUILD_CACHE_DIR}_wall
export CONFIG_DIR="exclaim_nh35_tri_jws_r2b7_120lev_${SLURM_NNODES}nodes"

echo "Variables used for this run:"
printenv | grep -E '^(SLURM_|ICON4PY_|GT4PY_|GHEX_|NCCL_|CUDA_|HIP_|CUPY_|FI_|HSA_|AMD_)' | sort

echo "Executing JW4Py on ${SLURM_NNODES} MI300A nodes to check the WALL CLOCK timer reported at the end of the run (MPI)"

srun -u --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores \
    bash -c 'export CUPY_CACHE_DIR=$(pwd)/cupy_rocm_cache/rank_$SLURM_PROCID; export HIP_VISIBLE_DEVICES=${SLURM_LOCALID}; echo "SLURM_LOCALID: ${SLURM_LOCALID}: GPU ${HIP_VISIBLE_DEVICES}"; \
    icon4py-driver \
    --config-file-path ${CONFIG_DIR}/config.yml \
    --grid-file-path $(realpath ${ICON_GRID}) \
    --icon4py-backend dace_gpu \
    --log-level ${ICON4PY_DRIVER_LOGGING_LEVEL}'
rm -rf ${OUTPUT_PATH}
