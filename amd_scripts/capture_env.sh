#!/usr/bin/env bash
# Capture full GPU-node environment for cross-cluster comparison.
# Run inside a GPU allocation on each cluster (beverin, aac6, etc.):
#
#   srun --partition=mi300 --gres=gpu:1 --ntasks=1 --time=00:10:00 \
#       bash amd_scripts/capture_env.sh
#
# or:
#
#   salloc ... ; bash amd_scripts/capture_env.sh
#
# Output: env_<hostname>_<date>.txt in the current directory.
# Diff the two files to find the variables that differ between clusters.

set -u

OUT="env_$(hostname -s)_$(date +%Y%m%d_%H%M%S).txt"

section() {
    echo
    echo "===== $* ====="
}

run() {
    # Run a command, suppress 'command not found' but keep real errors.
    if command -v "$1" >/dev/null 2>&1; then
        "$@" 2>&1
    else
        echo "[skip: $1 not on PATH]"
    fi
}

{
    section "host / kernel"
    uname -a
    hostname -f 2>/dev/null || hostname

    section "/etc/os-release"
    cat /etc/os-release 2>/dev/null

    section "cpu"
    run lscpu

    section "numa topology"
    run numactl -H

    section "memory"
    run free -h

    section "gpu visible to job"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
    echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
    echo "GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-unset}"

    section "rocm version files"
    cat /opt/rocm/.info/version 2>/dev/null || echo "[no /opt/rocm/.info/version]"
    cat /opt/rocm/.info/version-dev 2>/dev/null
    ls -1 /opt/rocm-* 2>/dev/null

    section "hipconfig"
    run hipconfig --full

    section "compilers"
    run amdclang++ --version
    run amdclang --version
    run clang++ --version
    run hipcc --version
    run gcc --version

    section "rocm-smi: showallinfo"
    run rocm-smi --showallinfo

    section "rocm-smi: hardware identity"
    run rocm-smi --showproductname --showserial --showuniqueid --showvbios --showdriverversion

    section "rocm-smi: clocks / power / perf"
    run rocm-smi --showclocks --showperflevel
    run rocm-smi -P
    run rocm-smi --showmaxpower

    section "rocm-smi: memory"
    run rocm-smi --showmeminfo vram --showmeminfo vis_vram --showmeminfo gtt

    section "rocm-smi: partitioning (MI300 specific)"
    run rocm-smi --showcomputepartition
    run rocm-smi --showmemorypartition
    run rocm-smi --showxgmierr

    section "rocm-smi: temperature / fan (state of node)"
    run rocm-smi --showtemp --showfan

    section "rocm-smi: pids on GPU (concurrency check)"
    run rocm-smi --showpids

    section "amd-smi: static"
    run amd-smi static
    section "amd-smi: metric"
    run amd-smi metric
    section "amd-smi: firmware"
    run amd-smi firmware
    section "amd-smi: bad-pages / ras"
    run amd-smi bad-pages
    run amd-smi ras

    section "rocminfo (HSA agents, queues, xnack)"
    run rocminfo

    section "kernel modules (amdgpu)"
    /sbin/modinfo amdgpu 2>&1 | head -30 || echo "[no modinfo]"

    section "dmesg amdgpu (last 50)"
    dmesg 2>/dev/null | grep -i amdgpu | tail -50 || echo "[dmesg not readable]"

    section "relevant env vars"
    env | grep -iE '^(rocm|hip|hsa|hcc|gpu|amd|llvm|hcc|omp|gt4py|dace)' | sort

    section "ulimits"
    ulimit -a

    section "loaded modules (Lmod / spack uenv)"
    module list 2>&1 || true
    echo "UENV_MOUNT_LIST=${UENV_MOUNT_LIST:-unset}"
    echo "UENV_VIEW=${UENV_VIEW:-unset}"

    section "slurm context"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
    echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
    echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
    echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
    echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset}"

    section "done"
    date
} > "$OUT" 2>&1

echo "Wrote $OUT"
echo "Tip: rsync this file off the cluster and diff against the other cluster's capture."
