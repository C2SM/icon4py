# OpenCode Instructions: Weekly Slack Summary

You are generating a concise weekly activity summary for the icon4py repository.
The summary will be posted to a Slack channel. Use the provided Markdown context
file (`weekly_slack_summary_context.md`) as the only source of facts.

## Output format

Produce a single Markdown document with exactly these sections, in order:

1. **Title**: "icon4py weekly summary: <start-date> - <end-date>"
2. **Closed PRs** — list each closed/merged PR with a one-sentence summary,
   author, and link. Separate infrastructure-only PRs from substantive changes
   when possible.
3. **Ongoing PRs** — active open PRs with a one-sentence status and link.
4. **Inactive open PRs** — short list of open PRs with no recent activity.
5. **Opened issues** — new issues, one line each with link.
6. **Closed issues** — resolved issues, one line each with link.
7. **Weekly CI status** — report from GitLab.

## Tone and style

- Keep the whole summary short (aim for under 40 lines).
- Use full links; Slack will unfurl them if configured.
- One-sentence summaries only.
- Be factual; do not invent details not present in the context.
- For CI, explicitly distinguish these cases:
  - all-good (green)
  - no recent pipeline found
  - pipeline still running
  - failed jobs; list each failed job with name, link, and whether it looks like
    an infrastructure failure (runner, network, container) or a test failure.
- Do not include raw JSON or commit hashes unless they add meaningful context.
