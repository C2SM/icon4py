# Agent Review

The `Agent Review` workflow ([`.github/workflows/agent-review.yml`](../../.github/workflows/agent-review.yml)) posts an advisory review of a pull request written by a coding agent (a _harness_ such as opencode or Pi) that can explore the repository with tools. The agent runs inside the [AWF](https://github.com/github/gh-aw-firewall) sandbox, without network access, and uses the [CSCS inference service](https://docs.cscs.ch/services/inference/) through AWF's API proxy, so that neither the inference key nor a write-capable token is readable by the agent or by code from the PR. It is **never a required status check**. It runs independently of [PR Agent](pr_agent.md) and uses no comment commands, so it does not interfere with PR Agent's.

The ICON4Py workflow is a small caller of the reusable workflow in [GridTools/ci-workflows](https://github.com/GridTools/ci-workflows), which holds the pipeline (its four jobs, the harness adapters, the AWF configuration and the prompt), its security invariants, its remaining risks and its tests; see its [documentation](https://github.com/GridTools/ci-workflows/blob/main/docs/agent-review.md). Changes to the pipeline are made and reviewed there, and the design decisions are recorded in its [ADR 0001](https://github.com/GridTools/ci-workflows/blob/main/docs/adr/0001-sandboxed-agent-pr-review.md) and [ADR 0002](https://github.com/GridTools/ci-workflows/blob/main/docs/adr/0002-reusable-workflow.md).

## Why ICON4Py uses it

- [PR Agent](pr_agent.md) reviews a PR from its diff with a single model call and never sees the rest of the repository. A coding agent can read related code, follow call sites across the workspace packages and run `git log`. The review rule ICON4Py cares most about, generic code that belongs in `model/common` or duplicates something already there, cannot be checked from a diff alone.
- It complements PR Agent and does not replace it: both are advisory and independent. They use different keys, different workflow files (and event policies) and different comment markers.

**Risks ICON4Py accepts**: a sandbox escape would expose the dedicated key, which is why it is low-budget and quick to rotate; an injected agent can spend that key's budget, which is the only hard cap across runs; and prompt injection can still steer the review's text. The review is therefore advisory, never a status check, and names the commit it reviewed. The full list and the mitigations are in the called workflow's [documentation](https://github.com/GridTools/ci-workflows/blob/main/docs/agent-review.md#what-remains-risky).

## Usage

- Anyone with write access can request a review of any PR not opened by a bot, including a fork or draft PR, by adding the **`agent-review` label**. The **labeller's** permission is checked, not the author's. To review again, remove and re-add the label. Pushes to the PR do not trigger a review.
- **Run workflow** on the Actions tab (`workflow_dispatch`, from `main`) reviews a given PR number, optionally with another harness or as a dry run that only shows the comment in the run summary.
- There is no automatic review when a PR is opened. The called workflow supports one, but its trigger types are not enabled here.
- Write access comes from the same teams and accounts as for PR Agent (see [pr_agent.md](pr_agent.md#usage)). `triage` is not enough: the permission API reports it as `read`.

Each harness keeps one comment per PR, updated in place by later runs. The comment names the harness, its version, the model and the reviewed commit; pushes after that commit are not reviewed. A review can take up to about 20 minutes.

## What ICON4Py configures

In the workflow file:

- The **rule files** the agent follows, in this order: `.github/copilot-instructions.md`, `AGENTS.md`, `CODING_GUIDELINES.md`. Each is reset to the PR's base version before the agent runs, so a PR cannot change the rules it is reviewed by. To change what the agent checks, extend these files rather than the prompt.
- The **prompt addendum** [`.github/agent-review/prompt-addendum.md`](../../.github/agent-review/prompt-addendum.md): ICON4Py-specific notes added to the prompt, read from the workflow's commit (`main`), never from the PR. Every file it names in backticks (with an extension such as `.md`, `.toml` or `.yaml`) must be one of the rule files, or the review fails; directories and tool names are fine.
- The **pinned version** of the reusable workflow (see [Updating the pinned version](#updating-the-pinned-version)).
- The **permission ceiling** for every called job, `contents: read` and `pull-requests: write`, and the explicit mapping of the key. `secrets: inherit` is never used, and would not work anyway: the reusable workflow is in another organization.

The called workflow also resets every agent instruction file a PR touches (`AGENTS.md`, `CLAUDE.md`, `.claude/`, `.agents/`, `.opencode/` and similar, at any depth), and gives no review if one of them is a symlink to a path it does not reset. `CLAUDE.md -> AGENTS.md` is accepted because both ends are reset. Before adding another symlinked instruction file, read "What remains risky" in the called workflow's documentation: its target must be added to the `instruction_files` input.

The inference endpoint is not configurable here: it is fixed centrally in GridTools/ci-workflows, so the key can only be sent to an allowlisted host and no repository variable can redirect it.

[Repository variables](https://github.com/C2SM/icon4py/settings/variables/actions), changeable without a PR by anyone with write access:

| Variable               | Meaning                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `AGENT_REVIEW_HARNESS` | `opencode` (recommended) or `pi`; `stub` for tests. A dispatch can choose another one.                                     |
| `AGENT_REVIEW_MODEL`   | Model id on the CSCS inference service, e.g. `zai-org/GLM-5.3-Flash`. No default: without it, a review with a model fails. |

Only these two inputs come from variables: a harness from the central allowlist, which a writer can choose by dispatch anyway, and a model id for the fixed endpoint, which the key's model restriction and budget bound. Every other input is set in the workflow file, so changing it needs a reviewed PR.

The **audit upload** (`upload_audit`, AWF's audit and proxy logs as an artifact) is off and is not mapped from a variable: in this public repository any signed-in user can download the artifact. Enable it only for debugging, by a reviewed PR, and remove it afterwards.

## Required repository settings

Without these, the workflow does not run, or runs without a key.

1. **The `agent-review` environment, restricted to the default branch** (**Settings → Environments → agent-review → Deployment branches and tags → Selected branches**: `main` only, with **Allow administrators to bypass configured protection rules** off), holding the secret **`AGENT_REVIEW_INFERENCE_API_KEY`**. Only the called `review` job uses it, and its branch rule is evaluated against this repository's run, so the label and dispatches from `main` get the key while dispatches of other refs are denied. If the environment is ever deleted, recreate it before the next run: GitHub creates an unknown environment on first use, unprotected and without the key. Never create a repository- or organization-level secret of that name: the called workflow fails if the key resolves without the environment.
2. **A dedicated CSCS key** in that secret, never PR Agent's: created in the [inference UI](https://ui.inference.cscs.ch/login) for this workflow alone, with its own token budget and reset period, and restricted to the configured model. Its budget is the only hard spending cap.
3. **An Actions event policy** (**Settings → Actions → Policies**) scoped by workflow path to `.github/workflows/agent-review.yml`, this repository's caller file (not the path in GridTools/ci-workflows), allowing **`pull_request_target` and `workflow_dispatch`**. GitHub blocks `pull_request_target` by default in public repositories from **2 December 2026**, and a policy's event list is an exhaustive allowlist. PR Agent's policy (see [pr_agent.md](pr_agent.md#event-policy)) covers only its own file.
4. The repository variables above and the label `agent-review`.
5. If the organization or the repository ever restricts which actions may run (**Settings → Actions → General**), the allowlist needs `GridTools/ci-workflows/.github/workflows/agent-review.yml` and the actions it runs under this repository's policy; at the current pin these are `actions/upload-artifact`, `actions/download-artifact` and `github/gh-aw-firewall`.

## Updating the pinned version

The `uses:` line pins a release of GridTools/ci-workflows by full commit SHA, with the tag in a comment; never a branch or a tag name. [PR Agent](pr_agent.md#updating-the-pinned-version) uses the same release; update both pins together. Before merging an update, whether from Dependabot or by hand:

1. Read the [release notes](https://github.com/GridTools/ci-workflows/releases). A new permission, a new required input, a raised cap or a harness or AWF change is marked as breaking there. A new permission must also be added to the calling job's `permissions:`, or every run fails at startup.

2. Check that the commit is on its `main` and that the tag points to it:

   ```bash
   gh api repos/GridTools/ci-workflows/compare/main...<sha> --jq .status   # expect 'identical' or 'behind'
   gh api repos/GridTools/ci-workflows/git/ref/tags/<tag> --jq .object.sha  # expect <sha> (annotated tags: dereference)
   ```

3. After merging, test as below.

## Testing changes

The workflow file and the addendum are read from `main`, so a change takes effect once merged. Then:

1. **Run workflow** with harness `stub` and dry run: the run summary shows the stub's comment, and the `authorize` job's notices name the pinned commit of GridTools/ci-workflows.
2. The same with the default harness, dry run (this uses the key).
3. The label on a PR.

Before merging, the plumbing can be tested in the reusable workflow's [caller test mode](https://github.com/GridTools/ci-workflows/blob/main/docs/agent-review.md#caller-test-mode): on a branch of this repository with a PR to `main` (a draft works), temporarily add a `pull_request` trigger and a second job with the same `uses:`, `permissions:`, `project_name`, `rule_files` and `prompt_addendum`, plus `harness: stub`, `pr: ${{ github.event.pull_request.number }}` and `dry_run: true`, and no `secrets:`. It uses no key and no environment and posts nothing. Under `pull_request` the workflow file, the rule files and the addendum are PR content, so the reset of the rule files is not tested. While it is enabled, the Actions event policy for this file must also allow `pull_request`. Remove the trigger and the job before merging.

Changes to the pipeline itself are tested in GridTools/ci-workflows.
