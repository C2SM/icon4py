#!/bin/bash
#SBATCH --job-name=dycore_fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --uenv=icon/26.7:v1@santis
#SBATCH --view=default
#SBATCH --output=dycore_fusion_%j_%N.out
#SBATCH --gres=gpu:1

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the ICON4Py PR checkout}"
exec bash amd_scripts/fusion_benchmark/launch.sh nvidia "$@"
