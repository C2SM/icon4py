# PR Agent

The `PR Agent` workflow ([`.github/workflows/pr-agent.yml`](../../.github/workflows/pr-agent.yml)) runs [PR Agent](https://github.com/The-PR-Agent/pr-agent) on the [CSCS inference service](https://docs.cscs.ch/services/inference/) to post an advisory review on pull requests. PR Agent reads the diff through the GitHub API with one model call per command, so it never sees the rest of the repository; for a review by an agent that explores the repository, see [agent_review.md](agent_review.md). It is **never a required status check**: if PR Agent fails or times out, the run stays green, and a comment on the PR links to the failed run. A later successful run removes that comment.

The ICON4Py workflow is a small caller of the reusable workflow in [GridTools/ci-workflows](https://github.com/GridTools/ci-workflows), which holds the pipeline (authorization, the model chain, the PR Agent image pinned by digest, the failure comment), its security invariants, its remaining risks and its tests; see its [documentation](https://github.com/GridTools/ci-workflows/blob/main/docs/pr-agent.md). Changes to the pipeline are made and reviewed there.

## Usage

- `/review` runs automatically when a non-draft PR is opened, reopened or marked ready, if its author has write access. Pushes to the PR do not trigger a review.
- Users with write access can comment `/review`, `/improve`, `/ask "<question>"`, `/describe` or `/help` on any PR, including a draft or one from an outside contributor. The commenter's permission is checked. `/describe` posts its summary as a comment and never edits the PR description.
- Comments with setting arguments (`--section.key=value`) are refused with a notice in the run, since they could override the workflow's settings. Single-dash arguments such as `/review -i` work.
- Any comment from a writer that starts with `/` starts PR Agent: a `/word` that is not a PR Agent command ends in the failure comment.
- Write access comes from the `EXCLAIM`, `Icon4py-Admin` and `CI Users` teams, from organization ownership and from direct collaborators. The CI machine accounts are of type `User`, so the workflow's bot filters do not exclude them. Every organization member would also pass if the organization's base permission were ever raised to `write`. `triage` is not enough: the permission API reports it as `read`.

The review policy is in [`.pr_agent.toml`](../../.pr_agent.toml). The review rules are in [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md), shared with GitHub Copilot's reviewer and the agent review, followed by `CODING_GUIDELINES.md` and `AGENTS.md`. Extend those files rather than adding guidance to `.pr_agent.toml`. PR Agent reads all of them from `main` through the API, so a PR cannot change the policy or the rules for its own review, and changes take effect once merged.

## What ICON4Py configures

In the workflow file:

- The **triggers**: `pull_request_target` (`opened`, `reopened`, `ready_for_review`) and `issue_comment` (`created`); not `pull_request`, since PRs from forks get no secrets there. The concurrency group never cancels a run in progress.
- The **pinned version** of the reusable workflow (see [Updating the pinned version](#updating-the-pinned-version)).
- The **permission ceiling** for every called job, `contents: read`, `issues: write` and `pull-requests: write` (no `contents: write`), and the explicit mapping of the key. `secrets: inherit` is never used, and would not work anyway: the reusable workflow is in another organization.
- The **link to this page**, shown in errors and in the failure comment.

The inference endpoint is not configurable here: the reusable workflow sends the key only to an endpoint on its central allowlist (today only the CSCS one), and no repository variable can redirect it. The key's environment is the default, `pr-agent`; never take it from a variable either.

[Repository variables](https://github.com/C2SM/icon4py/settings/variables/actions), changeable without a PR by anyone with write access, and validated by the reusable workflow:

| Variable                     | Default      | Purpose                                                     |
| ---------------------------- | ------------ | ----------------------------------------------------------- |
| `PR_AGENT_MODEL`             | **required** | CSCS model id, as listed by `GET /v1/models`                |
| `PR_AGENT_FALLBACK_MODELS`   | **required** | Comma-separated ids, tried in order; `none` for no fallback |
| `PR_AGENT_TOKEN_BUDGET`      | `262144`     | Context window PR Agent assumes                             |
| `PR_AGENT_MAX_OUTPUT_TOKENS` | `100000`     | Tokens reserved for the answer, reasoning included          |
| `PR_AGENT_AI_TIMEOUT`        | `720`        | Seconds one model may take before the next one is tried     |

- Since any writer can change these, what bounds them is the key itself: its model restriction and its token budget.
- Model ids are stored without the `openai/` prefix, which the workflow adds. CSCS retires models on short notice, so check them against the [model table](https://docs.cscs.ch/services/inference/api/).
- The token limits apply to every model in the chain, so they must fit the smallest one: the output reserve is sent as `max_tokens`, which a model with a lower output limit rejects. Check a fallback before adding it, since it is only reached when the models before it fail. Too small an output reserve makes reasoning models return empty responses.
- A diff that does not fit in the token budget is reviewed in up to three chunks, sent to the model in parallel and merged into one review (`enable_large_pr_chunking` in `.pr_agent.toml`). Each chunk is billed as a separate call; files that still do not fit are listed in the review's coverage footer.
- The PR Agent step's limit is one `PR_AGENT_AI_TIMEOUT` per model plus two minutes, so a model that times out is not retried: the next one is tried instead. The reusable workflow rejects a chain that does not fit in its 45-minute job with two minutes to spare.
- The review names the model that produced it, so a fallback answer is visible.

## Required repository settings

1. **The `pr-agent` environment, restricted to the default branch** (**Settings → Environments → pr-agent → Deployment branches and tags → Selected branches**: `main` only, with **Allow administrators to bypass configured protection rules** off), holding the secret **`PR_AGENT_INFERENCE_API_KEY`**. Only the called job that runs PR Agent uses it, and its branch rule is evaluated against this repository's run, which is on `main` for both events. If the environment is ever deleted, recreate it before the next run: GitHub creates an unknown environment on first use, unprotected and without the key. Never create a repository- or organization-level secret of that name: the reusable workflow fails if the key resolves without the environment.
2. **A dedicated CSCS key** in that secret, not shared with the [agent review](agent_review.md) or another tool: created in the [inference UI](https://ui.inference.cscs.ch/login) with a token budget, a reset period and a restriction to the models in the chain. Its budget is the hard spending cap. The `CSCS_INFERENCE_API_KEY` repository secret belongs to the weekly Slack summary and is not used here.
3. **The Actions event policy** "Allow PR Agent events" (see [Event policy](#event-policy)).
4. The repository variables above.
5. If the organization or the repository ever restricts which actions may run (**Settings → Actions → General**), the allowlist needs `GridTools/ci-workflows/.github/workflows/pr-agent.yml` and what it runs under this repository's policy; at the current pin that is only the `docker://pragent/pr-agent` image.

## Event policy

GitHub blocks `pull_request_target` by default in public repositories from **2 December 2026**, unless an Actions event policy allows it; PR Agent's automatic review depends on that event. The repository therefore has the policy "Allow PR Agent events" under **Settings → Actions → Policies**, scoped by workflow path to `.github/workflows/pr-agent.yml` (this repository's caller file, not the path in GridTools/ci-workflows) and allowing `pull_request_target` and `issue_comment`. Its event list is an allowlist, so a trigger added to the workflow must be added to the policy too, and renaming the workflow file takes it out of the policy's scope. The policy covers no other workflow (see [agent_review.md](agent_review.md#required-repository-settings)). See [Control workflow execution](https://docs.github.com/actions/how-tos/administer/control-workflow-execution).

## Security invariants

The full list is in the reusable workflow's [documentation](https://github.com/GridTools/ci-workflows/blob/main/docs/pr-agent.md#security-invariants). For this caller:

- **Never check out or run PR code**, and never grant `contents: write`. PR Agent reads the diff through the GitHub API.
- **The called workflow's authorization is the only gate.** It requires write access from the PR author (automatic review) or the commenter (commands), and fails closed. `pull_request_target` ignores the fork-approval setting, so without it anyone could spend the inference budget.
- **Review labels stay disabled and `/describe` posts a comment**, since their output is derived from untrusted PR text. The reusable workflow sets these in PR Agent's environment, so a failed fetch of `.pr_agent.toml` cannot re-enable them; a reviewed change to `.pr_agent.toml` still can, so review it like code.
- **The key, the environment and the endpoint are never taken from a variable**, and the key is mapped explicitly from the `pr-agent` environment.

## Updating the pinned version

The `uses:` line pins a release of GridTools/ci-workflows by full commit SHA, with the tag in a comment; never a branch or a tag name. The [agent review](agent_review.md#updating-the-pinned-version) uses the same release; update both pins together. Before merging an update, whether from Dependabot or by hand:

1. Read the [release notes](https://github.com/GridTools/ci-workflows/releases). A new permission, a new required input, a new endpoint or a new PR Agent image is flagged there. A new permission must also be added to the calling job's `permissions:`, or every run fails at startup.

2. Check that the commit is on its `main` and that the tag points to it:

   ```bash
   gh api repos/GridTools/ci-workflows/compare/main...<sha> --jq .status   # expect 'identical' or 'behind'
   gh api repos/GridTools/ci-workflows/git/ref/tags/<tag> --jq .object.sha  # expect <sha> (annotated tags: dereference)
   ```

3. After merging, test as below.

## Testing changes

The workflow file and `.pr_agent.toml` are read from `main`, so a change takes effect once merged. After merging, check one automatic review (a writer's non-draft PR) and one `/review` comment: the `authorize` job's notice names the pinned commit of GridTools/ci-workflows, and the PR Agent job's log shows `Applying repo settings (sections: [...])` when `.pr_agent.toml` was applied (the bare `Applying repo settings` appears in every run).

Before merging, the call can be tested in the reusable workflow's [caller test mode](https://github.com/GridTools/ci-workflows/blob/main/docs/pr-agent.md#caller-test-mode): a temporary, separate workflow file (for example `.github/workflows/pr-agent-test.yml`) triggered by `pull_request`, with the same `uses:`, `permissions:`, `with:` and `secrets:` plus `dry_run: true`, which uses no key and no environment and posts nothing. Never add `pull_request` to `pr-agent.yml` itself, since the event policy does not allow it. The called workflow only acts on a non-draft PR being opened, reopened or marked ready; on a push or on a draft PR every called job is skipped, so a green run then proves only that GitHub accepted the call (the pin, the input and secret names and the permission ceiling). Delete the test file before merging. Changes to `.pr_agent.toml` and the review rules cannot be tested from a branch.

Changes to the pipeline itself are tested in GridTools/ci-workflows.
