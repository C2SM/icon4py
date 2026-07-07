#!/usr/bin/env bash
# [help] Generate and optionally post the weekly Slack activity summary.
# Invoked via: ./scripts/run weekly-slack-summary [--dry-run] [--skip-opencode] [--offline]

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

DRY_RUN=""
SKIP_OPENCODE=""
OFFLINE=""
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN="--dry-run" ;;
        --skip-opencode) SKIP_OPENCODE="--skip-opencode" ;;
        --offline) OFFLINE="--offline" ;;
        --help|-h)
            echo "Usage: weekly-slack-summary [--dry-run] [--skip-opencode] [--offline] [extra-args]"
            echo "  Generate and optionally post the weekly Slack activity summary."
            echo "  Pass --dry-run to generate the Markdown file without posting to Slack."
            echo "  Pass --skip-opencode to use raw context as the summary (no OpenCode step)."
            echo "  Pass --offline to use synthetic data instead of calling external APIs."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

require_cmd uv

log_info "Running weekly Slack summary generator..."
uv run -q --frozen --isolated --python 3.12 --only-group scripts \
    python3 "${REPO_ROOT}/scripts/python/weekly_slack_summary.py" generate \
    ${DRY_RUN} ${SKIP_OPENCODE} ${OFFLINE} "${EXTRA_ARGS[@]}"

log_info "Weekly Slack summary complete."
