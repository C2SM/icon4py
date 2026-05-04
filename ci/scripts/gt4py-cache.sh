#!/usr/bin/env bash

# Sets up a persistent gt4py cache directory based on backend and week to start
# with a fresh cache every week. GT4PY_BUILD_CACHE_BASE_DIR is set as the root
# and GT4PY_BUILD_CACHE_DIR is set to
# ${GT4PY_BUILD_CACHE_BASE_DIR}/${BACKEND}/uv-lock-<hash of uv.lock>-${DATE}.

set -euo pipefail

if [[ "${ICON4PY_CI_DISABLE_PERSISTENT_GT4PY_CACHE:-}" == "true" ]]; then
    echo "Using non-persistent gt4py cache because ICON4PY_CI_DISABLE_PERSISTENT_GT4PY_CACHE is set"
    export GT4PY_BUILD_CACHE_LIFETIME="session"
    exit 0
fi

# First clean up files and directories older than 7 days in the base cache
# directory. The backend directories can stay, but the date-based directories
# are removed. There may be concurrent cleanup, ignore failures.
find "${GT4PY_BUILD_CACHE_BASE_DIR}/icon4py" -mindepth 2 -maxdepth 2 -type d -mtime +7 -exec rm -rf {} + || true

uv_lock_hash=$(sha256sum "./uv.lock" | awk '{print $1}')

# Then set the cache directory for this run based on the backend and current date.
DATE=$(date +%Y-%W)
export GT4PY_BUILD_CACHE_DIR="${GT4PY_BUILD_CACHE_BASE_DIR}/icon4py/${BACKEND}/uv-lock-${uv_lock_hash}-${DATE}"
mkdir -p "${GT4PY_BUILD_CACHE_DIR}"

echo "Using GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR}"
