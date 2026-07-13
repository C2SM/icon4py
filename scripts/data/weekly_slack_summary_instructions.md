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
- Do not use Markdown headings (`#`, `##`, etc.). Use `*Section Title*` for
  section titles (Slack renders single asterisks as bold).
- Format every link as Slack mrkdwn: `<URL|display text>`. Do not use raw URLs
  or Markdown `[text](url)`.

## Required content for every PR and issue

For each listed item, include all of:

1. Title as a clickable `<URL|title>` link
2. Author (the user who opened it)
3. A one-sentence description of what the item is actually about

The description must not simply restate the title. Use the body, commit
messages, review comments, or issue comments to produce a meaningful summary.
If the title is vague (e.g. "fix bug", "update foo"), explain the concrete
change. If there is no additional information beyond the title, say so.

## Output format

1. `*icon4py weekly summary: <start-date> - <end-date>*`
2. `*Merged PRs*`
   - `<PR URL|title>` by <author>: <one-sentence summary>
3. `*Closed PRs (without merging)*`
   - Same format as merged PRs. Omit this section entirely if there are none.
4. `*Ongoing PRs*`
   - `<PR URL|title>` by <author>: \<one-sentence status/summary>
5. `*Inactive open PRs*`
   - List up to 9 highlighted inactive PRs: 3 newest, 3 oldest, 3 most active.
   - Format: `<PR URL|title>` by <author> \[<reason tags>\]
   - Mention how many more inactive PRs exist and that the listed ones are a
     selection.
6. `*Opened issues*`
   - `<issue URL|title>` by <author>: <one-sentence summary>
7. `*Closed issues*`
   - `<issue URL|title>` by <author>: <one-sentence summary>
8. `*Weekly CI status*`
   - Report the status for the current week (Monday to Sunday), even though the
     PRs and issues above are from the previous week.
   - Use a single compact line when possible: `<status> — <pipeline URL|pipeline>`.
   - For failed jobs, group them by failure category (e.g., runner failures,
     test failures, infrastructure failures). Show up to 5 categories. For each
     category, give the total count, one representative clickable
     `<job URL|job name>` link, and a short (few-word) summary. Present the
     category line compactly, e.g.
     `Runner failure: 100 jobs — <job URL|job name> (compute nodes failed to connect)`.
   - If there are more than 5 categories, state the total number of categories
     and point to the pipeline link for details.
   - Distinguish green / no recent pipeline / running / failed jobs.
9. `*Nightly benchmarking CI status*`
   - Report the status of the nightly benchmarking pipeline for the last 24
     hours, using the same compact format as the weekly CI status.

If a section is empty, write `(none)`.

## Tone and style

- Keep the whole summary under 45 lines and **under 3,500 characters** so it
  stays within a single Slack message. Be terse in CI sections.
- Mention blockers briefly when present.
- Do not include raw JSON or commit hashes.

## Weekly Easter Egg

End with a short, silly joke or humorous observation related to icon4py's world
— icosahedral grids, weather/climate simulations, Fortran, Python, stencil
computations, parallel computing, Slurm, or distributed filesystems. One or two
lines max.

**Strictly forbidden:** "Why did the X" or "Why does the X" joke format. No setup-punchline jokes of any kind. Use wordplay, puns, absur observations, or deadpan humor instead.

**Strictly forbidden:** "Why did the X" or "Why does the X" joke format. No setup-punchline jokes of any kind. Use wordplay, puns, absur observations, or deadpan humor instead.
