# Weekly Slack Summary Instructions

You are generating a concise weekly activity summary for the icon4py repository.
The summary will be posted to a Slack channel. Use the provided Markdown context
file (`weekly_slack_summary_context.md`) as the only source of facts.

## Ground rules

- Return the final Markdown summary directly in your response.
- Do not create files and do not wrap the output in a code block.
- Use only facts present in the provided context.
- Do not use em-dashes (`—`); use hyphens (`-`) or colons instead.

## Output format

Produce a single Markdown document with exactly these sections, in order:

1. **Title**: "icon4py weekly summary: <start-date> - <end-date>"
2. **Closed PRs** — list each closed/merged PR with a one-sentence description
   of what it is about, author, and link. Separate infrastructure-only PRs from
   substantive changes when possible.
3. **Ongoing PRs** — active open PRs with a one-sentence status, description,
   and link.
4. **Inactive open PRs** — a single deduplicated highlights list of up to 15
   PRs. Each entry must include a one-sentence description and be tagged with
   why it is included:
   - `(newest inactive)` — recently updated but still inactive
   - `(oldest inactive)` — stalest
   - `(most active inactive)` — most discussion despite no recent update
   - Combinations like `(newest & most active inactive)` are allowed.
5. **Opened issues** — new issues, one line each with a one-sentence description
   and link.
6. **Closed issues** — resolved issues, one line each with a one-sentence
   description and link.
7. **Weekly CI status** — report from GitLab.

## Tone and style

- Keep the whole summary short (aim for under 40 lines).
- Use full links; Slack will unfurl them if configured.
- One-sentence summaries only.
- Mention blockers briefly when present, for example requested changes, failing
  CI, unresolved conflicts, or draft status.
- For CI, explicitly distinguish these cases:
  - all-good (green)
  - no recent pipeline found
  - pipeline still running
  - failed jobs; list each failed job with name, link, and whether it looks like
    an infrastructure failure (runner, network, container) or a test failure.
- Do not include raw JSON or commit hashes unless they add meaningful context.

## Weekly Easter Egg

End the summary with a short, lighthearted sign-off for the week: a joke, quote,
or nerdy observation. It should relate in some way to icon4py's world —
icosahedral grids, weather/climate simulations, Fortran, Python, stencil
computations, or parallel computing. Keep it to one or two lines, make it silly
or witty, and do not attribute a quote to a real person unless it actually is
one.
