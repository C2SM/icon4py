# Weekly Slack Summary Instructions

You are generating a concise weekly activity summary for the icon4py repository.
The summary will be posted to a Slack channel as `mrkdwn` text.

Use the provided Markdown context file (`weekly_slack_summary_context.md`) as the
source of facts.

## Ground rules

- Return the final mrkdwn summary directly in your response.
- Do not create files and do not wrap the output in a code block.
- Use only facts present in the context.
- Do not use em-dashes (`—`); use hyphens (`-`) or colons instead.
- Do not use Markdown headings. Use `*Section Title*` for section titles.
- Format links as raw URLs.
- One-sentence summaries only.

## Output format

1. `*icon4py weekly summary: <start-date> - <end-date>*`
2. `*Closed PRs*`
   - Under `*Infrastructure / Tooling*`, list PRs about CI, build, packaging,
     tests, linting, or developer tooling.
   - Under `*Model / Science*`, list PRs about physics, numerics, dycore,
     advection, microphysics, or grid behavior.
   - Each entry: title, author, one-sentence description, link.
3. `*Ongoing PRs*` — title, author, one-sentence status, link.
4. `*Inactive open PRs*` — up to 15 highlights with reason tags.
5. `*Opened issues*` — title, author, description, link.
6. `*Closed issues*` — title, author, description, link.
7. `*Weekly CI status*` — short report.

For every PR and issue, include a one-sentence description of what it is
actually about. Do not just repeat the title.

## Tone and style

- Keep the whole summary under 45 lines.
- Mention blockers briefly when present.
- For CI, distinguish: all-good, no recent pipeline, still running, failed jobs.
- Do not include raw JSON or commit hashes.

## Weekly Easter Egg

End with a short, silly/nerdy sign-off related to icon4py's world. One or two
lines max.
