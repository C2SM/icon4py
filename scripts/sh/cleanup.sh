#!/usr/bin/env bash
# Remove common build/cache artifacts from the repo tree.
# Invoked via:  ./scripts/run cleanup [--dry-run]

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        --help|-h)
            echo "Usage: cleanup [--dry-run]"
            echo "  Remove build/cache artifacts from the repository."
            exit 0
            ;;
        *)
            log_error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

PATTERNS=(
    "__pycache__"
    ".mypy_cache"
    ".pytest_cache"
    ".ruff_cache"
    "*.egg-info"
    "dist"
    "build"
)

for pat in "${PATTERNS[@]}"; do
    while IFS= read -r -d '' target; do
        if [[ "$DRY_RUN" -eq 1 ]]; then
            log_info "[dry-run] would remove: $target"
        else
            log_info "Removing: $target"
            rm -rf "$target"
        fi
    done < <(find "$REPO_ROOT" -name "$pat" -not -path '*/.git/*' -print0 2>/dev/null)
done

[[ "$DRY_RUN" -eq 0 ]] && log_info "Cleanup complete."
