# Weekly Slack Summary Instructions

You are generating a concise weekly activity summary for the icon4py repository.
The summary will be posted to a Slack channel as `mrkdwn` text.

Use the provided Markdown context file (`weekly_slack_summary_context.md`) as the
source of facts.

## Ground rules

- Write the final summary to the file path given in the prompt. This file is
  what gets posted to Slack.
- Use only facts present in the context.
- Do not use em-dashes (`—`); use simple sentences separated by periods (`.`) or if necessary joined by colons (`:`) or semicolons (`;`).
- Do not use Markdown headings (`#`, `##`, etc.). Use `*Section Title*` for
  section titles (Slack renders single asterisks as bold).
- Format every link as Slack mrkdwn: `<URL|display text>`. Do not use raw URLs
  or Markdown `[text](url)`.
- Use `-` as list marker.

## Length limit

**The summary must be under 3,800 characters and under 50 lines.**
Slack's hard limit is 4,000 characters. Stay under 3,800 to leave a safety
margin. Be informative: use the full budget, but do not exceed 3,800.

After writing the summary to the file, verify its length with a shell command
(e.g. `wc -m <path>`). If it exceeds 3,800 characters, edit the file to trim
content (drop descriptions before titles, reduce entry counts), then verify
again. Repeat until it fits. Never guess the length, always measure it.

Guidelines:

- Merged PRs: list all merged PRs (combine rest into a group bullet if >10)
- Closed PRs: list all PRs closed without merging (combine rest into group bullet if >5)
- Ongoing PRs: list notable ongoing PRs (up to ~10), ongoing means commits in the last two weeks
- Inactive PRs: list notable inactive PRs (up to ~5), inactive means no commits in the last two weeks
- Issues: list all, keep short
- CI sections: one line each, include pipeline URL

## Output format

1. `*icon4py weekly summary: <start-date> - <end-date>*`
2. `*Merged PRs*`
3. `*Closed PRs (without merging)*`: Omit section if none.
4. `*Ongoing PRs*`
5. `*Inactive open PRs*`: Mention total count of unlisted ones.
6. `*Opened issues*`
7. `*Closed issues*`
8. `*Weekly CI status*`
9. `*Nightly benchmarking CI status*`

If a section is empty, write `(none)`.

## Content

PR entries: `<URL|short-title>` by author: brief summary (5-10 words).
Closed PRs: `<URL|short-title>` by author: brief summary (5-10 words).
Inactive PRs: `<URL|short-title>` by author: no description.
Issues: `<URL|short-title>` by author: brief summary (5-10 words).
CI: `<status>: <pipeline URL|pipeline> - failure note`

## Trimming priority

If the content is too long, prioritize removing content as follows:

- Shorten the closed and inactive PRs lists
- Remove descriptions from closed and inactive PRs
- Remove URLs from closed and inactive PRs
- Only refer to closed and inactive PRs by PR number
- Shorten other lists, combining the overflow into a combined point saying how many other PRs/issues there are

## Weekly Easter Egg

End with a short joke or humorous observation related to icon4py's world
(icosahedral grids, weather/climate, Fortran, Python, stencils, MPI, Slurm).
One line only.

**Strictly forbidden:** "Why did the X" / "Why does the X" format. No setup-punchline jokes. Use wordplay, puns, absurd observations, or deadpan humor.
