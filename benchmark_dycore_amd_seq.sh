#!/bin/bash
#SBATCH --job-name=dycore_benchmark_mi300
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH -A csstaff
#SBATCH --partition mi300
#SBATCH --reservation cscs
#SBATCH --uenv=prgenv-gnu/7.2.3:v1,/iopsstor/scratch/cscs/ioannmag/ICON4PY/icon4py-amd-benchmarks/py_venv_rocm_mpi4.squashfs:/iopsstor/scratch/cscs/ioannmag/ICON4PY/icon4py-amd-benchmarks/.venv_rocm
#SBATCH --view=default

cd $(git rev-parse --show-toplevel)

# Snapshot of the node's Cray power/energy counters, compared with Slurm's ConsumedEnergy.
PM_DIR=pm_counters_${SLURM_JOB_ID}
mkdir -p ${PM_DIR}
cp -a /sys/cray/pm_counters ${PM_DIR}/t0

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_KEEP_DACE_BUILD_FILES=1
export GT4PY_BUILD_JOBS_MODE=thread
export GT4PY_BUILD_JOBS=${SLURM_CPUS_PER_TASK}
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100
export PYTHONOPTIMIZE=2
export ROCR_VISIBLE_DEVICES=0

ICON_GRID=icon_benchmark_global

export VENV=/iopsstor/scratch/cscs/ioannmag/ICON4PY/icon4py-amd-benchmarks/.venv_rocm

# run <chiplet_number> <persistent_disable> <vlb_disable> <workspace_size, 0 = no external workspace>
run() {
    export DACE_compiler_cuda_chiplet_number=$1
    export ICON4PY_DACE_PERSISTENT_DISABLE=$2
    export ICON4PY_VLB_DISABLE=$3
    if [[ $4 == 0 ]]; then
        unset ICON4PY_BACKEND_WORKSPACE_SIZE # icon4py rejects 0
    else
        export ICON4PY_BACKEND_WORKSPACE_SIZE=$4
    fi
    export GT4PY_BUILD_CACHE_DIR=MI300A_dycore_${ICON_GRID}_chiplet$1_persistentdisable$2_vlbdisable$3_workspace$4
    echo "=== ${GT4PY_BUILD_CACHE_DIR}"

    bash -c "
        source setup_amd_env.sh
        source ${VENV}/bin/activate
        python3 -m pytest -sv -m continuous_benchmarking -p no:tach \
            --benchmark-only --benchmark-warmup=on \
            --benchmark-warmup-iterations=${ICON4PY_STENCIL_TEST_ITERATIONS} \
            --benchmark-min-rounds=${ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS} \
            --benchmark-time-unit=ms \
            --backend=dace_gpu --grid=${ICON_GRID} \
            --benchmark-json=${GT4PY_BUILD_CACHE_DIR}/benchmark_results_${SLURM_JOBID}.json \
            'model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]'
    "
}

run 1 1 1 0
run 1 0 1 0
run 1 0 1 8589934592
run 1 0 0 8589934592
run 6 0 1 8589934592
run 6 0 0 8589934592

cp -a /sys/cray/pm_counters ${PM_DIR}/t1
