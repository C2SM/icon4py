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
- No Easter egg at the end.

## Output format

Produce one compact block with these sections:

1. `*icon4py weekly summary: <start-date> - <end-date>*`
2. `*Closed PRs*` — one line per PR using this exact format:
   `:white_check_mark: <title> — <one-sentence summary> (<author>) <url>`
3. `*Ongoing PRs*` — one line per PR:
   `:arrows_counterclockwise: <title> — <one-sentence summary> (<author>) <url>`
4. `*Inactive open PRs*` — one line per highlight:
   `:sleeping: <title> — <reason tag> <url>`
5. `*Opened issues*` — one line per issue:
   `:new: <title> — <one-sentence summary> (<author>) <url>`
6. `*Closed issues*` — one line per issue:
   `:white_check_mark: <title> — <one-sentence summary> (<author>) <url>`
7. `*Weekly CI status*` — one line:
   `:large_green_circle:`, `:yellow_circle:`, `:red_circle:`, or `:gray_circle:`
   followed by a short note.

If a section has no entries, write `*Section Title*`: `(none)`.

For every PR and issue, the one-sentence summary must describe what it is
actually about, not just restate the title.

## Tone and style

- Keep the whole summary under 25 lines.
- Mention blockers briefly when present.
- Do not include raw JSON or commit hashes.
