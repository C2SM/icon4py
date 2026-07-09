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
2. `*Closed PRs*`
   - `<PR URL|title>` by <author>: <one-sentence summary>
3. `*Ongoing PRs*`
   - `<PR URL|title>` by <author>: \<one-sentence status/summary>
4. `*Inactive open PRs*`
   - `<PR URL|title>` by <author> \[<reason tags>\]
5. `*Opened issues*`
   - `<issue URL|title>` by <author>: <one-sentence summary>
6. `*Closed issues*`
   - `<issue URL|title>` by <author>: <one-sentence summary>
7. `*Weekly CI status*`
   - Report the status for the current week (Monday to Sunday), even though the
     PRs and issues above are from the previous week.
   - Always include the pipeline URL as a clickable `<URL|pipeline>` link.
   - For failed jobs, list up to 12 representative failed jobs. Each entry must
     have a clickable `<job URL|job name>` link, the `failure_reason` in
     parentheses, and a one-sentence plain-English summary of what the job was
     trying to do and what went wrong. Example:
     `<job URL|test_model_stencils_aarch64: [gtfn_gpu, icon_global, dycore]>`
     (script_failure): the aarch64 stencil tests for the dycore on the global
     icon grid using the gtfn_gpu backend failed to run their test script.
   - Use the truncated log tail in the context when available; if no log tail is
     provided, infer the summary from the job name and failure reason. Do not
     copy raw log text verbatim.
   - If there are more than 12 failed jobs, state the total count and point to
     the pipeline link for the full list.
   - Distinguish green / no recent pipeline / running / failed jobs.

If a section is empty, write `(none)`.

## Tone and style

- Keep the whole summary under 45 lines.
- Mention blockers briefly when present.
- Do not include raw JSON or commit hashes.

## Weekly Easter Egg

End with a short, silly/nerdy sign-off related to icon4py's world — icosahedral
grids, weather/climate simulations, Fortran, Python, stencil computations,
parallel computing, Slurm, or distributed filesystems. One or two lines max.
