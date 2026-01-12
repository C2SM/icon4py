#!/usr/bin/env bash

# Log all output to separate logfiles, stored as artifacts in gitlab. Output to
# stdout only from rank 0.

set -euo pipefail

log_file="${CI_PROJECT_DIR:+${CI_PROJECT_DIR}/}pytest-log-rank-${SLURM_PROCID}.txt"

if [[ "${SLURM_PROCID}" -eq 0  ]]; then
    echo "Starting pytest on rank ${SLURM_PROCID}, logging to stdout and ${log_file}"
    $@ |& tee "${log_file}"
else
    echo "Starting pytest on rank ${SLURM_PROCID}, logging to ${log_file}"
    $@ >& "${log_file}"
fi
