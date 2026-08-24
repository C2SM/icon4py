#!/usr/bin/env bash

# Sets up a persistent gt4py cache directory based on the base image, uv.lock
# hash, compiler flags, job name and week to start with a fresh cache every week.
# ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR is set as the root and
# GT4PY_BUILD_CACHE_DIR is set to
# ${ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR}/icon4py/gt4py-cache/base-<hash>-uv-lock-<hash of uv.lock>-flags-<hash of CXXFLAGS=${CXXFLAGS} NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS}>-job-<job name hash>-${DATE}.

set -euo pipefail

# First clean up files and directories older than 7 days in the base cache
# directory. There may be concurrent cleanup, ignore failures.
find "${ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR}/icon4py/gt4py-cache" -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} + || true

uv_lock_hash=$(sha256sum "./uv.lock" | awk '{print substr($1,1,32)}')
job_name_hash=$(echo -n "${CI_JOB_NAME}" | sha256sum | awk '{print substr($1,1,32)}')
if [[ -z "${BASE_IMAGE:-}" ]]; then
	echo "BASE_IMAGE must be set and non-empty" >&2
	exit 1
fi
base_image_hash=$(echo -n "${BASE_IMAGE}" | sha256sum | awk '{print substr($1,1,32)}')
flags_hash=$(echo -n "CXXFLAGS=${CXXFLAGS:-} NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-}" | sha256sum | awk '{print substr($1,1,32)}')

# Then set the cache directory for this run based on the backend and current date.
DATE=$(date +%Y-%W)
# TEMPORARY (experiment branch, revert before merge): hardcoded cache-busting
# suffix so proof runs always get a fresh cache directory. Bump the counter
# for each new proof run (phase 2/3), remove the whole line before merge.
BUST="cleanupproof-1"
export GT4PY_BUILD_CACHE_DIR="${ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR}/icon4py/gt4py-cache/base-${base_image_hash}-uv-lock-${uv_lock_hash}-flags-${flags_hash}-job-${job_name_hash}-${DATE}-${BUST}"
mkdir -p "${GT4PY_BUILD_CACHE_DIR}"

echo "Using GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR}"
