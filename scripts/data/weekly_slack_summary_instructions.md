# Weekly Slack Summary Instructions

You are generating a concise weekly activity summary for the icon4py repository.
The summary will be posted to a Slack channel as `mrkdwn` text.

The attached JSON file (`weekly_slack_summary_context.json`) contains the raw
collected data for the week. Use it as the only source of facts.

JSON schema overview:

- `week_start`, `week_end`: ISO timestamps for the reporting week.
- `repository`: repo name.
- `github_prs`: contains `closed_prs`, `active_prs`, `inactive_pr_highlights`.
  Each PR object has `number`, `title`, `author`, `url`, `body`, `commits`,
  `comments`, `review_comments`, and date fields.
- `github_issues`: contains `opened_issues` and `closed_issues`.
- `gitlab_ci`: status, URL, failed/running jobs.

## Ground rules

- Return the final mrkdwn summary directly in your response.
- Do not create files and do not wrap the output in a code block.
- Use only facts present in the JSON context.
- Do not use em-dashes (`—`); use hyphens (`-`) or colons instead.
- Do not use Markdown headings. Use `*Section Title*` for section titles.
- Format links as raw URLs.
- One-sentence summaries only.

## Output format

Produce a single document with these sections, in order:

1. Title: "*icon4py weekly summary: <start-date> - <end-date>*"
2. *Closed PRs* — title, author, one-sentence description, link.
3. *Ongoing PRs* — title, author, one-sentence status/description, link.
4. *Inactive open PRs* — highlights list (up to 15) tagged by reason.
5. *Opened issues* — title, author, description, link.
6. *Closed issues* — title, author, description, link.
7. *Weekly CI status* — short report.

For every PR and issue, include a brief one-sentence description of what it is
actually about. Do not just repeat the title. Use `body`, `comments`,
`review_comments`, or `commits` to derive the description when needed.

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
