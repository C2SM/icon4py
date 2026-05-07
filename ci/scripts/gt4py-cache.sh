#!/usr/bin/env bash

# Sets up a persistent gt4py cache directory based on backend and week to start
# with a fresh cache every week. ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR is set as
# the root and GT4PY_BUILD_CACHE_DIR is set to
# ${ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR}/icon4py/${BACKEND}/uv-lock-<hash of uv.lock>-job-<hash of job name>-${DATE}.

set -euo pipefail

# First clean up files and directories older than 7 days in the base cache
# directory. The backend directories can stay, but the date-based directories
# are removed. There may be concurrent cleanup, ignore failures.
find "${ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR}/icon4py" -mindepth 2 -maxdepth 2 -type d -mtime +7 -exec rm -rf {} + || true

uv_lock_hash=$(sha256sum "./uv.lock" | awk '{print $1}')
job_name_hash=$(echo -n "${CI_JOB_NAME}" | sha256sum | awk '{print $1}')

# Then set the cache directory for this run based on the backend and current date.
DATE=$(date +%Y-%W)
export GT4PY_BUILD_CACHE_DIR="${ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR}/icon4py/${BACKEND}/uv-lock-${uv_lock_hash}-job-${job_name_hash}-${DATE}"
mkdir -p "${GT4PY_BUILD_CACHE_DIR}"

echo "Using GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR}"

# TODO: This is here just for debugging, probably remove?
if [[ "${ICON4PY_CI_WIPE_GT4PY_CACHE:-}" == "true" ]]; then
    echo "Wiping cache in ${GT4PY_BUILD_CACHE_DIR}"
    rm -rf "${GT4PY_BUILD_CACHE_DIR}"
fi
