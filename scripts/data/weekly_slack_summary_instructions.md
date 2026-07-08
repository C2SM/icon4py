# Weekly Slack Summary Instructions

You are generating a concise weekly activity summary for the icon4py repository.
The summary will be posted to a Slack channel as `mrkdwn` text.

The attached JSON file contains the raw collected data. Use it as the only
source of facts.

## Ground rules

- Return the final mrkdwn summary directly in your response.
- Do not create files and do not wrap the output in a code block.
- Use only facts present in the JSON.
- Do not use em-dashes (`—`); use hyphens (`-`) or colons instead.
- Do not use Markdown headings. Use `*Section Title*` for section titles.
- Format links as raw URLs.
- One-sentence summaries only.

## Required sections and exact JSON paths

Use these exact top-level keys in the JSON:

1. *Title*: from `week_start` and `week_end`.
2. *Closed PRs*: from `github_prs.closed_prs`. For each item include `title`,
   `author`, `url`, and a one-sentence description based on `body` or `commits`.
3. *Ongoing PRs*: from `github_prs.active_prs`. Same fields as closed PRs.
4. *Inactive open PRs*: from `github_prs.inactive_pr_highlights`. Each item has
   `title`, `author`, `url`, `updated_at`, and `reasons`. If the list is empty,
   write "No inactive PR highlights this week."
5. *Opened issues*: from `github_issues.opened_issues`. Include `title`,
   `author`, `url`, and a one-sentence description from `body`.
6. *Closed issues*: from `github_issues.closed_issues`. Same fields.
7. *Weekly CI status*: from `gitlab_ci`. Report `status`, `url`, and any
   `failed_jobs` or `running_jobs`.

If a section's list is empty, write "(None)" or a short note. Do not omit the
section entirely.

## Tone and style

- Keep the whole summary short (aim for under 40 lines).
- For every PR and issue, include a one-sentence description of what it is
  actually about. Do not just repeat the title.
- Mention blockers briefly when present.
- For CI, explicitly distinguish: all-good, no recent pipeline, still running,
  failed jobs.
- Do not include raw JSON or commit hashes unless they add meaningful context.

## Weekly Easter Egg

End with a short, silly/nerdy sign-off related to icon4py's world — icosahedral
grids, weather/climate simulations, Fortran, Python, stencil computations, or
parallel computing. One or two lines max.
