#!/bin/bash
#SBATCH --job-name=dycore_fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --account=csstaff
#SBATCH --partition=mi300
#SBATCH --uenv=prgenv-gnu/7.2.3:2804758683
#SBATCH --view=default
#SBATCH --output=dycore_fusion_%j_%N.out

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the ICON4Py PR checkout}"
exec bash amd_scripts/fusion_benchmark/launch.sh amd "$@"
