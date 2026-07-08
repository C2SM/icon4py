# SPEC — Weekly Slack Activity Summary

## Goal

Create an automated weekly Slack activity summary for the icon4py repository.

## Scope

- GitHub Actions workflow in `.github/workflows/weekly-slack-summary.yml`.
- Workflow runs at 08:00 UTC every Monday (`0 8 * * 1`).
- Targets `C2SM/icon4py` on GitHub.
- Produces a Markdown summary of repository activity and the latest GitLab weekly CI status.
- Posts the Markdown to a Slack channel via an incoming webhook.
- Helper scripts follow the existing `scripts/` conventions (`scripts/python/...`, tests).
- OpenCode batch run generates the polished Markdown from collected static inputs.

## Out of scope

- Actual Slack credentials are not available; the workflow uses a placeholder secret (`SLACK_WEBHOOK_URL`) and the integration is left for manual testing.
- Full natural-language summarization of every PR/issue is delegated to the OpenCode batch run; static collection only gathers raw facts.

## Static inputs collected

1. **GitHub PRs** for the previous 7 days (Monday 00:00 UTC to Sunday 23:59 UTC).
   - Closed PRs: number, title, author, URL, merged/closed status, body, list of commits, recent review/comments activity.
   - Open PRs with activity: title, author, URL, last updated, recent comments/reviews/commits.
   - Inactive open PRs: title, author, URL, last updated.
2. **GitHub issues** for the previous 7 days.
   - Opened issues: title, author, URL, body snippet.
   - Closed issues: title, author, URL, closure reason if available.
3. **GitLab weekly CI** (`https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/5125340235196978/2255149825504677/-/pipelines`).
   - Latest pipeline scheduled/started within the previous ~30 hours (to cover the Monday 01:00 UTC run).
   - Status: success, failed, running, canceled, no pipeline found.
   - If failed: list failed jobs with names, failure reasons, and links.
   - If running: report jobs still running.

## OpenCode batch run

- A Python helper script collects the static inputs and writes them as structured JSON/Markdown context.
- It then invokes `opencode` in batch mode with an instruction file, passing the collected context.
- The instruction tells OpenCode to generate the final Markdown summary with the sections requested by the user.
- The instruction emphasizes brevity, links, one-sentence summaries, and separating CI infrastructure failures from test failures.

## Deliverables

1. `.github/workflows/weekly-slack-summary.yml`
2. `scripts/python/weekly_slack_summary.py` (collects static inputs, runs opencode, posts to Slack)
3. Entry point through `scripts/run` (the Python script exposes a `cli` object)
4. `scripts/data/weekly_slack_summary_instructions.md` (OpenCode instructions)
5. `scripts/tests/python/test_weekly_slack_summary.py` (unit tests for collectors and CLI)
6. `.opencode/config.json` (OpenCode provider/model configuration for CSCS Inference)
7. This workflow must be runnable locally with dummy flags (no Slack posting, no external APIs).

## Acceptance criteria

- [x] Workflow file is syntactically valid and can be loaded by GitHub Actions.
- [x] Python helper script passes its unit tests and can run with `--dummy-input-data --dummy-summarization --dummy-output` without credentials or external APIs.
- [x] The default CLI invocation (no flags) uses live data, invokes OpenCode, writes files, and posts to Slack.
- [x] `--dummy-input-data` replaces live GitHub/GitLab API calls with synthetic fixture data.
- [x] `--dummy-summarization` bypasses OpenCode and uses the deterministic direct Markdown formatter.
- [x] `--dummy-output` skips the Slack webhook post and prints the final summary to stdout while still writing files.
- [x] The script checks that `opencode` is installed and fails with a clear message if not.
- [x] `opencode run` is invoked with the prompt before `--file` attachments and the unit test asserts this order.
- [x] OpenCode instructions explicitly tell the agent to return the summary in its response and not to create files or code blocks.
- [x] Every PR and issue listed in the summary has a brief one-sentence description of what it is about.
- [x] Inactive open PRs are presented as a single deduplicated highlights list (up to 15 PRs) tagged by why they are included (newest / oldest / most active inactive).
- [x] The summary style avoids em-dashes and mentions blockers when present.
- [x] GitLab CI section distinguishes: all-good, no recent pipeline, jobs still running, failed jobs (with links).
- [x] Pre-commit checks pass.
- [x] Two consecutive reviewer passes with no actionable findings.
- [x] Missing Slack webhook in non-dry-run mode fails with a non-zero exit code (R4).
- [x] OpenCode batch invocation is covered by a unit test or documented CLI reference (R7).
- [x] `.opencode/config.json` configures the `cscs-inference` provider, uses the Kimi K2.7 Code model, and reads the API key from the `CSCS_INFERENCE_API_KEY` environment variable.
- [x] The workflow installs the `@ai-sdk/openai-compatible` provider package and sets `OPENCODE_CONFIG`, `OPENCODE_MODEL`, and `CSCS_INFERENCE_API_KEY` so OpenCode uses the CSCS Inference endpoint.
- [x] `SLACK_CHANNEL` support is removed from the script and workflow because Slack incoming webhooks do not allow channel override.

## Progress

- [x] Phase 0/1 — Goal elicitation and SPEC drafting
- [x] Phase 2 — Exploration
- [x] Phase 5 — Initial implementation
- [x] Phase 5 — CLI redesign (dummy flags, drop shell wrapper, workflow simplification)
- [x] Phase 5 — Content and invocation fixes (opencode order, descriptions, inactive PR highlights, style rules)
- [x] Phase 5 — OpenCode provider config and Slack channel cleanup
- [ ] Phase 6 — Local verification and review convergence
- [ ] Phase 7 — Human review / PR

## Decisions

- Use the GitHub REST API (no GraphQL) to reduce credential and complexity requirements.
- Use the GitLab REST API for pipeline/job status.
- Use the Python standard library (`urllib.request`, `json`, `datetime`) for all external HTTP calls to avoid new runtime dependencies.
- The script follows the existing project convention of using `typer` for the CLI.
- Collect data for the previous calendar week (Monday–Sunday UTC).
- The OpenCode instruction file is kept in `scripts/data/` alongside the script.
- Slack posting uses a single incoming webhook URL; channel override is not supported because Slack incoming webhooks are bound to the channel chosen during installation.
- `opencode` is installed in the workflow via `npm install -g opencode-ai`; the helper script checks that the `opencode` executable is present and fails early if not.
- The generated Markdown is uploaded as a GitHub Actions artifact in every run, even when not in dry-run mode.
- The helper script runs in the repo's isolated `scripts` dependency group (`--only-group scripts`).
- The script supports an `--offline` mode that uses synthetic data so it can be tested without API credentials.
- The workflow accepts a `workflow_dispatch` `offline` boolean input for manual test runs.
- The CLI redesign uses three orthogonal opt-out flags (`--dummy-input-data`, `--dummy-summarization`, `--dummy-output`) so the scheduled/production invocation needs no flags.
- The shell wrapper is removed; the Python script is registered with `scripts/run` by exposing a `cli` object.

## Open questions

- None currently.
