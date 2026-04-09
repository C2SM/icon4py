#!/usr/bin/env bash
# Setup local development environment.
# Invoked via:  ./scripts/run setup-env [--verbose]

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

VERBOSE=0
for arg in "$@"; do
    case "$arg" in
        --verbose|-v) VERBOSE=1 ;;
        --help|-h)
            echo "Usage: setup-env [--verbose]"
            echo "  Bootstrap the local development environment."
            exit 0
            ;;
        *)
            log_error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

log_info "Repository root: $REPO_ROOT"

# ── Check prerequisites ─────────────────────────────────────────────────────
require_cmd python3
require_cmd uv
require_cmd git

[[ "$VERBOSE" -eq 1 ]] && log_info "python3 → $(command -v python3)"
[[ "$VERBOSE" -eq 1 ]] && log_info "uv      → $(command -v uv)"

# ── Sync dependencies ───────────────────────────────────────────────────────
log_info "Syncing project dependencies …"
(cd "$REPO_ROOT" && uv sync)

log_info "Environment setup complete."
