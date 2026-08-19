#!/usr/bin/env bash

# Sets up a persistent gt4py cache directory based on the base image, uv.lock
# hash, job name and week to start with a fresh cache every week.
# ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR is set as the root and
# GT4PY_BUILD_CACHE_DIR is set to
# ${ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR}/icon4py/gt4py-cache/base-<hash>-uv-lock-<hash of uv.lock>-job-<job name>-${DATE}.

set -euo pipefail

# First clean up files and directories older than 7 days in the base cache
# directory. There may be concurrent cleanup, ignore failures.
find "${ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR}/icon4py/gt4py-cache" -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} + || true

uv_lock_hash=$(sha256sum "./uv.lock" | awk '{print substr($1,1,32)}')
job_name="${CI_JOB_NAME_SLUG}"
if [[ -z "${BASE_IMAGE:-}" ]]; then
    echo "BASE_IMAGE must be set and non-empty" >&2
    exit 1
fi
base_image_hash=$(echo -n "${BASE_IMAGE}" | sha256sum | awk '{print substr($1,1,32)}')
flags_hash=$(echo -n "CXXFLAGS=${CXXFLAGS:-} NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-}" | sha256sum | awk '{print substr($1,1,32)}')

# Then set the cache directory for this run based on the backend and current date.
DATE=$(date +%Y-%W)

# TEMPORARY (experiment branch, revert before merge): cache placement probe.
#   ICON4PY_CI_GT4PY_CACHE_MODE=ram      (default) build into container-local
#                                        /icon4py (in-memory overlay), no reuse.
#   lustre                             build directly on Lustre in a probe dir
#                                        (fresh due to CACHE_BUST), published
#                                        normally (well: written directly).
#   staged                             build into node-local dir; stage-in from
#                                        a Lustre probe dir at start; publish
#                                        back at job end (atomic per-entry).
#   ICON4PY_CI_GT4PY_CACHE_BUST=<tag>    scopes probe dirs so runs never reuse
#                                        already-populated caches.
MODE="${ICON4PY_CI_GT4PY_CACHE_MODE:-ram}"
BUST="${ICON4PY_CI_GT4PY_CACHE_BUST:-nobust}"
PROBE_ROOT="${ICON4PY_CI_GT4PY_BUILD_CACHE_BASE_DIR}/icon4py/gt4py-cache-probe"

case "${MODE}" in
    ram)
        export GT4PY_BUILD_CACHE_DIR=/icon4py
        ;;
    lustre)
        export GT4PY_BUILD_CACHE_DIR="${PROBE_ROOT}/lustre/base-${base_image_hash}-uv-lock-${uv_lock_hash}-flags-${flags_hash}-${BUST}"
        mkdir -p "${GT4PY_BUILD_CACHE_DIR}"
        ;;
    staged)
        LUSTRE_STAGE_DIR="${PROBE_ROOT}/staged/base-${base_image_hash}-uv-lock-${uv_lock_hash}-flags-${flags_hash}-${BUST}"
        mkdir -p "${LUSTRE_STAGE_DIR}"
        export GT4PY_BUILD_CACHE_DIR=/icon4py
        echo "stage-in: copying ${LUSTRE_STAGE_DIR} -> ${GT4PY_BUILD_CACHE_DIR}"
        t0=${SECONDS}
        rsync -a --ignore-existing "${LUSTRE_STAGE_DIR}/" "${GT4PY_BUILD_CACHE_DIR}/" || true
        echo "stage-in took $((SECONDS - t0))s"

        _compile_cache_publish() {
            local target="${LUSTRE_STAGE_DIR}"
            local src="${GT4PY_BUILD_CACHE_DIR}/.gt4py_cache"
            if [[ ! -d "${src}" ]]; then
                echo "publish: nothing to publish (${src} missing)"
                return 0
            fi
            echo "publish: merging ${src} -> ${target}"
            local s=${SECONDS}
            (
                exec {lockfd}>"${target}/.publish.lock"
                flock -x -w 900 "${lockfd}" || { echo "publish: lock timeout, skipping"; exit 0; }
                local n_new=0 n_skip=0
                # entry = top-level content-hashed dir or file inside .gt4py_cache
                for entry in "${src}"/*; do
                    local name; name=$(basename "${entry}")
                    local dst="${target}/.gt4py_cache/${name}"
                    if [[ -e "${dst}" ]]; then
                        n_skip=$((n_skip + 1))
                        continue
                    fi
                    # atomic per entry: copy to temp then rename
                    local tmp="${dst}.tmp.${CI_JOB_ID:-$$}"
                    rm -rf "${tmp}"
                    cp -a "${entry}" "${tmp}" && mv -T "${tmp}" "${dst}" && n_new=$((n_new + 1)) || rm -rf "${tmp}"
                done
                echo "publish: done, new=${n_new} existing=${n_skip} in $((SECONDS - s))s"
            )
        }
        trap _compile_cache_publish EXIT
        ;;
    *)
        echo "ICON4PY_CI_GT4PY_CACHE_MODE=${MODE} unknown" >&2
        exit 1
        ;;
esac

mkdir -p "${GT4PY_BUILD_CACHE_DIR}"
echo "Using GT4PY_BUILD_CACHE_DIR=${GT4PY_BUILD_CACHE_DIR} (mode=${MODE}, bust=${BUST})"
