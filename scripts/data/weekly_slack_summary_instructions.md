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

## Required content for every PR and issue

For each PR or issue you list, you MUST include all of:

1. Title
2. Author (the user who opened it)
3. Link
4. A brief LLM-summarized description of what it is actually about

The description MUST NOT simply restate the title. Read the body, commit
messages, review comments, and issue comments to produce a meaningful summary.
If the title is vague (e.g. "fix bug" or "update foo"), explain the concrete
change. If the body is empty or unhelpful, infer the purpose from commit
messages. If there is truly no information beyond the title, state that
explicitly.

Example bad summary: "Update driver readme with configuration details" (this is
just the title).
Example good summary: "Adds a configuration section to the driver readme
covering mixed-precision and backend settings."

## Output format

1. `*icon4py weekly summary: <start-date> - <end-date>*`
2. `*Closed PRs*` — title, author, link, description.
3. `*Ongoing PRs*` — title, author, link, status/description.
4. `*Inactive open PRs*` — up to 15 highlights with reason tags.
5. `*Opened issues*` — title, author, link, description.
6. `*Closed issues*` — title, author, link, description.
7. `*Weekly CI status*` — short report.

## Tone and style

- Keep the whole summary under 45 lines.
- Mention blockers briefly when present.
- For CI, distinguish: all-good, no recent pipeline, still running, failed jobs.
- Do not include raw JSON or commit hashes.

## Weekly Easter Egg

End with a short, silly/nerdy sign-off related to icon4py's world. One or two
lines max.
