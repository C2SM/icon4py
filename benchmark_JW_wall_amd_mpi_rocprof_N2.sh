#!/bin/bash
#SBATCH --job-name=rocprof_JW4Py_mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
##SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --uenv=prgenv-gnu/7.2.3:2950075422,/ritom/scratch/cscs/ioannmag/cycle39/icon4py-benchmarks/py_venv_mpi4.squashfs:/ritom/scratch/cscs/ioannmag/cycle39/icon4py-benchmarks/.venv
#SBATCH -A csstaff
#SBATCH --view=default
#SBATCH --partition mi300
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

export CUDA_CACHE_DISABLE=1
export CUPY_CACHE_IN_MEMORY=1
export AMD_COMGR_CACHE=0

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export DACE_compiler_build_folder_mode="development"
export PYTHONOPTIMIZE=2
# export DACE_compiler_cuda_chiplet_number=6

export ICON4PY_PROFILE_MARKERS=roctx
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_ADD_GPU_TRACE_MARKERS="1"

export ICON4PY_DRIVER_LOGGING_LEVEL="warning"

export CC=$(which gcc)
export CXX=$(which g++)
export MPICH_CC=$(which gcc)
export MPICH_CXX=$(which g++)
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=AMD
export GHEX_GPU_ARCH=gfx942
export GHEX_TRANSPORT_BACKEND=MPI

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

export GT4PY_SKIP_DACE_WARNINGS=0

export LD_LIBRARY_PATH=$(pwd):${LD_LIBRARY_PATH}

# Optimization flags
GIBS_EXT=48
export ICON4PY_BACKEND_WORKSPACE_SIZE=$((${GIBS_EXT}*1073741824)) # 80Gib external workspace storage per device

export GT4PY_BUILD_CACHE_DIR="MI300A_JW_external${GIBS_EXT}GiB_mpi_roctx_${SUFFIX}_120lvl_ntasks${SLURM_NTASKS}_danieloptclean"
export ROCPROF_OUTPUT_FILENAME="rocprofv3_${GT4PY_BUILD_CACHE_DIR}"
export OUTPUT_PATH=$(pwd)/standalone_driver_output_${GT4PY_BUILD_CACHE_DIR}_wall
export CONFIG_DIR="exclaim_nh35_tri_jws_r2b7_120lev_${SLURM_NNODES}nodes"

echo "Variables used for this run:"
printenv | grep -E '^(SLURM_|ICON4PY_|GT4PY_|GHEX_|NCCL_|CUDA_|CUPY_|FI_)' | sort

echo "Executing JW4Py on ${SLURM_NNODES} MI300A nodes to check the WALL CLOCK timer reported at the end of the run (MPI)"

# if [[ ! -d ${GT4PY_BUILD_CACHE_DIR} ]]; then
    srun -u --cpu-bind=cores \
        bash -c 'printenv TMPDIR; export HIP_VISIBLE_DEVICES=${SLURM_LOCALID}; echo "SLURM_LOCALID: ${SLURM_LOCALID}: GPU ${HIP_VISIBLE_DEVICES}"; \
        icon4py-driver \
        --config-file-path ${CONFIG_DIR}/config.yml \
        --grid-file-path $(realpath ${ICON_GRID}) \
        --icon4py-backend dace_gpu \
        --log-level ${ICON4PY_DRIVER_LOGGING_LEVEL} \
        --output-path ${OUTPUT_PATH} \
        --no-enable-output'
    rm -rf ${OUTPUT_PATH}
# fi

srun -u --cpu-bind=cores \
    bash -c 'printenv TMPDIR; export HIP_VISIBLE_DEVICES=${SLURM_LOCALID}; echo "SLURM_LOCALID: ${SLURM_LOCALID}: GPU ${HIP_VISIBLE_DEVICES}"; rocprofv3 --kernel-trace on --marker-trace on --rccl-trace on --output-format pftrace -o ${ROCPROF_OUTPUT_FILENAME}_rank${SLURM_PROCID} --perfetto-buffer-size 2097152 --perfetto-shmem-size-hint 16384 --perfetto-buffer-fill-policy discard -- \
    icon4py-driver \
    --config-file-path ${CONFIG_DIR}/config.yml \
    --grid-file-path $(realpath ${ICON_GRID}) \
    --icon4py-backend dace_gpu \
    --log-level ${ICON4PY_DRIVER_LOGGING_LEVEL} \
    --output-path ${OUTPUT_PATH} \
    --no-enable-output'
rm -rf ${OUTPUT_PATH}
