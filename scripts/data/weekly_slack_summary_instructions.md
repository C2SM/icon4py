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

## Output format

Produce a numbered summary with a horizontal separator (`---`) between sections.

01. `*icon4py weekly summary: <start-date> - <end-date>*`
02. `---`
03. `*Closed PRs*`
    1. `<title>` by <author> — <one-sentence summary> <url>
    2. ...
04. `---`
05. `*Ongoing PRs*`
    1. `<title>` by <author> — <one-sentence status> <url>
    2. ...
06. `---`
07. `*Inactive open PRs*`
    1. `<title>` (<reason tag>) <url>
    2. ...
08. `---`
09. `*Opened issues*`
    1. `<title>` by <author> — <one-sentence summary> <url>
10. `---`
11. `*Closed issues*`
    1. `<title>` by <author> — <one-sentence summary> <url>
12. `---`
13. `*Weekly CI status*` — short report.

For every PR and issue, include a one-sentence description of what it is
actually about. Do not just repeat the title. Use body, comments, or commits to
infer content when the title is unclear.

## Tone and style

- Keep the whole summary under 45 lines.
- Mention blockers briefly when present.
- For CI, distinguish: all-good, no recent pipeline, still running, failed jobs.
- Do not include raw JSON or commit hashes.

## Weekly Easter Egg

End with a short, silly/nerdy sign-off related to icon4py's world. One or two
lines max.
