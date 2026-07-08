# Weekly Slack Summary Instructions

You are generating a concise weekly activity summary for the icon4py repository.
The summary will be posted to a Slack channel as `mrkdwn` text.

Use the provided Markdown context file (`weekly_slack_summary_context.md`) as the
primary source of facts.

## Ground rules

- Return the final mrkdwn summary directly in your response.
- Do not create files and do not wrap the output in a code block.
- Use only facts present in the provided context.
- Do not use em-dashes (`—`); use hyphens (`-`) or colons instead.
- Do not use Markdown headings (`#`, `##`, etc.). Use `*Section Title*` for
  section titles (Slack renders single asterisks as bold).
- Format every link as Slack mrkdwn: `<URL|display text>`. Do not use raw URLs
  and do not use Markdown `[text](url)` links.
- One-sentence summaries only.

## Output format

Produce a single document with exactly these sections, in order:

1. **Title line**: "*icon4py weekly summary: <start-date> - <end-date>*"
2. **Closed PRs** — for each PR: `<PR URL|title>` by author: one-sentence description.
3. **Ongoing PRs** — `<PR URL|title>` by author: one-sentence status/description.
4. **Inactive open PRs** — highlights list (up to 15), each tagged by reason.
5. **Opened issues** — `<issue URL|title>` by author: one-sentence description.
6. **Closed issues** — `<issue URL|title>` by author: one-sentence description.
7. **Weekly CI status** — short report.

For every PR and issue, include a brief one-sentence description of what it is
actually about. Do not just repeat the title. If the title is unclear, derive the
description from the body, comments, or commits in the context.

## Tone and style

- Keep the whole summary short (aim for under 40 lines).
- Mention blockers briefly when present.
- For CI, explicitly distinguish: all-good, no recent pipeline, still running,
  failed jobs (with name, link, and infra vs test failure).
- Do not include raw JSON or commit hashes unless they add meaningful context.

## Weekly Easter Egg

End with a short, silly/nerdy sign-off related to icon4py's world — icosahedral
grids, weather/climate simulations, Fortran, Python, stencil computations, or
parallel computing. One or two lines max.
