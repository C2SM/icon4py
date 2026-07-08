# Weekly Slack Summary Instructions

You are generating a concise weekly activity summary for the icon4py repository.
The summary will be posted to a Slack channel as `mrkdwn` text.

Use the provided Markdown context file (`weekly_slack_summary_context.md`) as the
primary source of facts.

## Ground rules

- Return the final Markdown/mrkdwn summary directly in your response.
- Do not create files and do not wrap the output in a code block.
- Use only facts present in the provided context.
- Do not use em-dashes (`—`); use hyphens (`-`) or colons instead.
- Do not use Markdown headings (`#`, `##`, etc.). Use `*Section Title*` for
  section titles (Slack renders single asterisks as bold).
- Use raw full links; Slack will unfurl them.
- One-sentence summaries only.

## Output format

Produce a single document with exactly these sections, in order:

1. **Title line**: "*icon4py weekly summary: <start-date> - <end-date>*"
2. **Closed PRs** — list each closed/merged PR with:
   - title as a link
   - author
   - one-sentence description of what it is about
   - link
3. **Ongoing PRs** — active open PRs with title, author, status, description, link.
4. **Inactive open PRs** — highlights list (up to 15) tagged by reason.
5. **Opened issues** — title, author, description, link.
6. **Closed issues** — title, author, description, link.
7. **Weekly CI status** — short report.

For every PR and issue, include a brief one-sentence description of what it is
actually about. Do not just repeat the title. If the title is unclear, derive the
description from the body, comments, or commits in the context.

## Tone and style

- Keep the whole summary short (aim for under 40 lines).
- Mention blockers briefly when present (requested changes, failing CI,
  unresolved conflicts, draft status).
- For CI, explicitly distinguish these cases:
  - all-good (green)
  - no recent pipeline found
  - pipeline still running
  - failed jobs; list each failed job with name, link, and whether it looks like
    an infrastructure failure or a test failure.
- Do not include raw JSON or commit hashes unless they add meaningful context.

## Weekly Easter Egg

End the summary with a short, lighthearted sign-off for the week: a joke, quote,
or nerdy observation. It should relate in some way to icon4py's world —
icosahedral grids, weather/climate simulations, Fortran, Python, stencil
computations, or parallel computing. Keep it to one or two lines, make it silly
or witty, and do not attribute a quote to a real person unless it actually is
one.
