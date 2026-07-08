#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --only-group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect repository activity and generate a weekly Slack summary."""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated, Any, Final

import typer


cli = typer.Typer(
    name=__name__.split(".")[-1].replace("_", "-"),
    no_args_is_help=True,
    help=__doc__,
)


_SCRIPTS_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[1]


GITHUB_REPO_OWNER: Final[str] = "C2SM"
GITHUB_REPO_NAME: Final[str] = "icon4py"
GITLAB_PROJECT_ID: Final[str] = "5125340235196978"
GITLAB_PIPELINE_URL_TEMPLATE: Final[str] = (
    "https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/"
    f"{GITLAB_PROJECT_ID}/2255149825504677/-/pipelines"
)
GITLAB_API_BASE: Final[str] = "https://gitlab.com/api/v4"
INSTRUCTIONS_FILE: Final[pathlib.Path] = (
    _SCRIPTS_DIR / "data" / "weekly_slack_summary_instructions.md"
)


def _iso_date(d: datetime.date) -> str:
    return d.isoformat()


def _previous_week_bounds(
    now: datetime.datetime | None = None,
) -> tuple[datetime.datetime, datetime.datetime]:
    """Return the previous Monday 00:00 UTC to Sunday 23:59:59 UTC."""
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    # Monday of current week
    current_monday = now - datetime.timedelta(days=now.weekday())
    current_monday = current_monday.replace(hour=0, minute=0, second=0, microsecond=0)
    previous_monday = current_monday - datetime.timedelta(days=7)
    previous_sunday = current_monday - datetime.timedelta(seconds=1)
    return previous_monday, previous_sunday


def _github_api_request(
    url: str,
    *,
    token: str | None = None,
    method: str = "GET",
    data: bytes | None = None,
) -> dict[str, Any] | list[Any]:
    """Make a GitHub API request and return the parsed JSON response."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "icon4py-weekly-slack-summary",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API request failed ({e.code}): {e.reason}\n{error_body}") from e
    return json.loads(body.decode("utf-8"))


def _github_search_issues(
    query: str,
    *,
    token: str | None = None,
    per_page: int = 100,
) -> list[dict[str, Any]]:
    """Search GitHub issues/PRs and return all matching items."""
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        params = f"q={urllib.parse.quote(query)}&per_page={per_page}&page={page}"
        url = f"https://api.github.com/search/issues?{params}"
        result = _github_api_request(url, token=token)
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected GitHub search response type: {type(result)}")
        page_items = result.get("items", [])
        items.extend(page_items)
        if len(page_items) < per_page:
            break
        page += 1
    return items


def _github_fetch_paginated(
    url: str,
    *,
    token: str | None = None,
    per_page: int = 100,
) -> list[dict[str, Any]]:
    """Fetch a paginated GitHub REST endpoint and return all items."""
    items: list[dict[str, Any]] = []
    page_url = url
    while page_url:
        separator = "&" if "?" in page_url else "?"
        paginated_url = f"{page_url}{separator}per_page={per_page}"
        result = _github_api_request(paginated_url, token=token)
        if isinstance(result, list):
            items.extend(result)
            # GitHub paginates via Link header; detect next by length for simplicity
            if len(result) < per_page:
                break
            page_url = f"{url}{separator}per_page={per_page}&page={len(items) // per_page + 1}"
        else:
            break
    return items


def _github_pr_commits(pr_number: int, *, token: str | None = None) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/pulls/{pr_number}/commits"
    return _github_fetch_paginated(url, token=token)


def _github_pr_comments(pr_number: int, *, token: str | None = None) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues/{pr_number}/comments"
    return _github_fetch_paginated(url, token=token)


def _github_pr_review_comments(pr_number: int, *, token: str | None = None) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/pulls/{pr_number}/comments"
    return _github_fetch_paginated(url, token=token)


def _collect_github_prs(
    start: datetime.datetime,
    end: datetime.datetime,
    *,
    token: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Collect closed, active open, and inactive open PRs for the week."""
    start_date = _iso_date(start.date())
    end_date = _iso_date(end.date())

    closed_query = (
        f"repo:{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME} is:pr is:closed "
        f"closed:{start_date}..{end_date}"
    )
    active_open_query = (
        f"repo:{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME} is:pr is:open "
        f"updated:{start_date}..{end_date}"
    )
    inactive_open_query = (
        f"repo:{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME} is:pr is:open updated:<{start_date}"
    )

    closed_prs_raw = _github_search_issues(closed_query, token=token)
    active_prs_raw = _github_search_issues(active_open_query, token=token)
    inactive_prs_raw = _github_search_issues(inactive_open_query, token=token)

    closed_prs: list[dict[str, Any]] = []
    for item in closed_prs_raw:
        pr_number = item["number"]
        commits = _github_pr_commits(pr_number, token=token)
        comments = _github_pr_comments(pr_number, token=token)
        review_comments = _github_pr_review_comments(pr_number, token=token)
        closed_prs.append(
            {
                "number": pr_number,
                "title": item["title"],
                "author": item["user"]["login"] if item.get("user") else None,
                "url": item["html_url"],
                "state_reason": item.get("state_reason"),
                "merged_at": item.get("merged_at"),
                "closed_at": item.get("closed_at"),
                "body": item.get("body") or "",
                "commits": [
                    {
                        "sha": c.get("sha", "")[:7],
                        "message": (c.get("commit", {}).get("message") or "").split("\n")[0],
                        "url": c.get("html_url", ""),
                    }
                    for c in commits
                ],
                "comments": [
                    {"author": c["user"]["login"], "body": c.get("body", "")[:200]}
                    for c in comments
                    if c.get("user")
                ],
                "review_comments": [
                    {"author": c["user"]["login"], "body": c.get("body", "")[:200]}
                    for c in review_comments
                    if c.get("user")
                ],
            }
        )

    active_prs: list[dict[str, Any]] = []
    for item in active_prs_raw:
        pr_number = item["number"]
        commits = _github_pr_commits(pr_number, token=token)
        comments = _github_pr_comments(pr_number, token=token)
        review_comments = _github_pr_review_comments(pr_number, token=token)
        active_prs.append(
            {
                "number": pr_number,
                "title": item["title"],
                "author": item["user"]["login"] if item.get("user") else None,
                "url": item["html_url"],
                "updated_at": item.get("updated_at"),
                "commits": [
                    {
                        "sha": c.get("sha", "")[:7],
                        "message": (c.get("commit", {}).get("message") or "").split("\n")[0],
                        "url": c.get("html_url", ""),
                    }
                    for c in commits
                ],
                "comments": [
                    {"author": c["user"]["login"], "body": c.get("body", "")[:200]}
                    for c in comments
                    if c.get("user")
                ],
                "review_comments": [
                    {"author": c["user"]["login"], "body": c.get("body", "")[:200]}
                    for c in review_comments
                    if c.get("user")
                ],
            }
        )

    inactive_prs: list[dict[str, Any]] = [
        {
            "number": item["number"],
            "title": item["title"],
            "author": item["user"]["login"] if item.get("user") else None,
            "url": item["html_url"],
            "updated_at": item.get("updated_at"),
        }
        for item in inactive_prs_raw
    ]

    return {
        "closed_prs": closed_prs,
        "active_prs": active_prs,
        "inactive_prs": inactive_prs,
    }


def _collect_github_issues(
    start: datetime.datetime,
    end: datetime.datetime,
    *,
    token: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Collect opened and closed issues for the week."""
    start_date = _iso_date(start.date())
    end_date = _iso_date(end.date())

    opened_query = (
        f"repo:{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME} is:issue created:{start_date}..{end_date}"
    )
    closed_query = (
        f"repo:{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME} is:issue is:closed "
        f"closed:{start_date}..{end_date}"
    )

    opened_raw = _github_search_issues(opened_query, token=token)
    closed_raw = _github_search_issues(closed_query, token=token)

    opened_issues = [
        {
            "number": item["number"],
            "title": item["title"],
            "author": item["user"]["login"] if item.get("user") else None,
            "url": item["html_url"],
            "created_at": item.get("created_at"),
            "body_snippet": (item.get("body") or "")[:400],
        }
        for item in opened_raw
    ]

    closed_issues = [
        {
            "number": item["number"],
            "title": item["title"],
            "author": item["user"]["login"] if item.get("user") else None,
            "url": item["html_url"],
            "closed_at": item.get("closed_at"),
            "state_reason": item.get("state_reason"),
        }
        for item in closed_raw
    ]

    return {"opened_issues": opened_issues, "closed_issues": closed_issues}


def _gitlab_api_request(url: str) -> dict[str, Any] | list[Any]:
    """Make a GitLab API request and return the parsed JSON response."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "icon4py-weekly-slack-summary"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitLab API request failed ({e.code}): {e.reason}\n{error_body}") from e
    return json.loads(body.decode("utf-8"))


def _collect_gitlab_ci(
    start: datetime.datetime,
    end: datetime.datetime,
) -> dict[str, Any]:
    """Collect the latest GitLab weekly CI pipeline status within the window."""
    project_path = urllib.parse.quote(
        f"cscs-ci/ci-testing/webhook-ci/mirrors/{GITLAB_PROJECT_ID}/2255149825504677",
        safe="",
    )
    url = f"{GITLAB_API_BASE}/projects/{project_path}/pipelines?order_by=id&sort=desc&per_page=20"
    pipelines = _gitlab_api_request(url)
    if not isinstance(pipelines, list):
        raise RuntimeError(f"Unexpected GitLab pipelines response type: {type(pipelines)}")

    latest: dict[str, Any] | None = None
    for pipeline in pipelines:
        created_str = pipeline.get("created_at")
        if not created_str:
            continue
        created = datetime.datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        if start <= created <= end:
            latest = pipeline
            break

    if latest is None:
        return {
            "status": "no_recent_pipeline",
            "url": GITLAB_PIPELINE_URL_TEMPLATE,
            "message": "No weekly pipeline found in the last ~30 hours.",
            "failed_jobs": [],
            "running_jobs": [],
        }

    pipeline_id = latest["id"]
    status = latest.get("status", "unknown")
    pipeline_web_url = latest.get("web_url", GITLAB_PIPELINE_URL_TEMPLATE)

    failed_jobs: list[dict[str, Any]] = []
    running_jobs: list[dict[str, Any]] = []

    if status in ("failed", "running"):
        jobs_url = (
            f"{GITLAB_API_BASE}/projects/{project_path}/pipelines/{pipeline_id}/jobs?per_page=100"
        )
        jobs = _gitlab_api_request(jobs_url)
        if not isinstance(jobs, list):
            raise RuntimeError(f"Unexpected GitLab jobs response type: {type(jobs)}")
        for job in jobs:
            job_status = job.get("status")
            job_info = {
                "id": job.get("id"),
                "name": job.get("name"),
                "stage": job.get("stage"),
                "status": job_status,
                "url": job.get("web_url"),
                "failure_reason": job.get("failure_reason"),
            }
            if job_status == "failed":
                failed_jobs.append(job_info)
            elif job_status == "running":
                running_jobs.append(job_info)

    return {
        "status": status,
        "url": pipeline_web_url,
        "pipeline_id": pipeline_id,
        "created_at": latest.get("created_at"),
        "message": None,
        "failed_jobs": failed_jobs,
        "running_jobs": running_jobs,
    }


def _format_pr_line(pr: dict[str, Any], *, merged_status: bool = False) -> str:
    if merged_status:
        merged = "merged" if pr.get("merged_at") else "closed"
        return f"- [{pr['title']}]({pr['url']}) by {pr['author']} ({merged})"
    return f"- [{pr['title']}]({pr['url']}) by {pr['author']}"


def _format_closed_pr_lines(pr: dict[str, Any]) -> list[str]:
    lines = [_format_pr_line(pr, merged_status=True)]
    if pr.get("commits"):
        lines.append("  Commits:")
        lines.extend(f"  - [{commit['sha']}] {commit['message']}" for commit in pr["commits"])
    all_comments = pr.get("comments", []) + pr.get("review_comments", [])
    if all_comments:
        lines.append(f"  Recent comments ({len(all_comments)}):")
        for comment in all_comments:
            body = comment.get("body", "")
            snippet = body[:200].replace("\n", " ")
            lines.append(f"    - {comment.get('author', 'unknown')}: {snippet}")
    return lines


def _format_active_pr_lines(pr: dict[str, Any]) -> list[str]:
    lines = [_format_pr_line(pr)]
    if pr.get("updated_at"):
        lines.append(f"  Last updated: {pr['updated_at']}")
    if pr.get("commits"):
        lines.append("  Recent commits:")
        lines.extend(f"  - [{commit['sha']}] {commit['message']}" for commit in pr["commits"])
    all_comments = pr.get("comments", []) + pr.get("review_comments", [])
    if all_comments:
        lines.append(f"  Recent comments ({len(all_comments)}):")
        for comment in all_comments:
            body = comment.get("body", "")
            snippet = body[:200].replace("\n", " ")
            lines.append(f"    - {comment.get('author', 'unknown')}: {snippet}")
    return lines


def _format_inactive_pr_lines(pr: dict[str, Any]) -> list[str]:
    return [f"- [{pr['title']}]({pr['url']}) by {pr['author']} (updated {pr['updated_at']})"]


def _format_issue_lines(issue: dict[str, Any], *, closed: bool = False) -> list[str]:
    if closed:
        return [
            f"- [{issue['title']}]({issue['url']}) by {issue['author']} "
            f"(reason: {issue.get('state_reason') or 'n/a'})"
        ]
    return [f"- [{issue['title']}]({issue['url']}) by {issue['author']}"]


def _format_gitlab_ci_lines(gitlab_ci: dict[str, Any]) -> list[str]:
    lines = [
        f"- Status: **{gitlab_ci['status']}**",
        f"- URL: {gitlab_ci['url']}",
    ]
    if gitlab_ci.get("message"):
        lines.append(f"- Message: {gitlab_ci['message']}")
    if gitlab_ci.get("failed_jobs"):
        lines.append(f"- Failed jobs ({len(gitlab_ci['failed_jobs'])}):")
        lines.extend(
            f"  - [{job['name']}]({job['url']}): {job.get('failure_reason') or 'failed'}"
            for job in gitlab_ci["failed_jobs"]
        )
    elif gitlab_ci.get("status") == "failed":
        lines.append(
            "- The pipeline status is failed, but no individual failed jobs were retrieved. "
            "See the pipeline link for details."
        )
    if gitlab_ci.get("running_jobs"):
        lines.append(f"- Running jobs ({len(gitlab_ci['running_jobs'])}):")
        lines.extend(f"  - [{job['name']}]({job['url']})" for job in gitlab_ci["running_jobs"])
    return lines


def _format_context_markdown(
    week_start: datetime.datetime,
    week_end: datetime.datetime,
    github_prs: dict[str, list[dict[str, Any]]],
    github_issues: dict[str, list[dict[str, Any]]],
    gitlab_ci: dict[str, Any],
) -> str:
    """Format the collected data as Markdown context for OpenCode."""
    lines: list[str] = [
        "# Weekly Activity Context",
        "",
        f"**Repository:** {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}",
        f"**Week:** {week_start.date().isoformat()} to {week_end.date().isoformat()} UTC",
        "",
        "## Pull Requests",
        "",
    ]

    lines.append(f"### Closed PRs ({len(github_prs['closed_prs'])})")
    for pr in github_prs["closed_prs"]:
        lines.extend(_format_closed_pr_lines(pr))

    lines.extend(["", f"### Active Open PRs ({len(github_prs['active_prs'])})", ""])
    for pr in github_prs["active_prs"]:
        lines.extend(_format_active_pr_lines(pr))

    lines.extend(["", f"### Inactive Open PRs ({len(github_prs['inactive_prs'])})", ""])
    for pr in github_prs["inactive_prs"]:
        lines.extend(_format_inactive_pr_lines(pr))

    lines.extend(
        ["", "## Issues", "", f"### Opened Issues ({len(github_issues['opened_issues'])})", ""]
    )
    for issue in github_issues["opened_issues"]:
        lines.extend(_format_issue_lines(issue))

    lines.extend(["", f"### Closed Issues ({len(github_issues['closed_issues'])})", ""])
    for issue in github_issues["closed_issues"]:
        lines.extend(_format_issue_lines(issue, closed=True))

    lines.extend(["", "## GitLab Weekly CI", ""])
    lines.extend(_format_gitlab_ci_lines(gitlab_ci))

    return "\n".join(lines) + "\n"


def _write_context_files(
    output_dir: pathlib.Path,
    context: dict[str, Any],
    markdown: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write JSON context and Markdown context files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "weekly_slack_summary_context.json"
    markdown_path = output_dir / "weekly_slack_summary_context.md"
    json_path.write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def _run_opencode(
    instructions_path: pathlib.Path,
    context_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Invoke OpenCode in batch mode to generate the final Markdown summary."""
    if not shutil.which("opencode"):
        raise RuntimeError(
            "The 'opencode' executable was not found in PATH. "
            "Install it with 'npm install -g opencode-ai' or ensure it is on PATH."
        )

    cmd = [
        "opencode",
        "run",
        "--quiet",
        "--file",
        str(instructions_path),
        "--file",
        str(context_path),
        "Generate the weekly Slack summary following the attached instructions and context.",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    output_path.write_text(result.stdout, encoding="utf-8")


def _post_to_slack(
    webhook_url: str,
    channel: str | None,
    markdown: str,
) -> None:
    """Post the Markdown summary to Slack via an incoming webhook."""
    payload: dict[str, Any] = {"text": markdown}
    if channel:
        payload["channel"] = channel
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            _ = response.read()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Slack webhook request failed ({e.code}): {e.reason}\n{error_body}"
        ) from e


def _github_token() -> str | None:
    """Return a GitHub API token from the environment, if available."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _collect_all(
    now: datetime.datetime | None = None,
    *,
    token: str | None = None,
) -> dict[str, Any]:
    """Collect all static inputs for the weekly summary."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    week_start, week_end = _previous_week_bounds(now)
    # Look for the latest weekly GitLab CI pipeline in the ~30 hours before now.
    gitlab_start = now - datetime.timedelta(hours=30)
    gitlab_end = now

    github_prs = _collect_github_prs(week_start, week_end, token=token)
    github_issues = _collect_github_issues(week_start, week_end, token=token)
    gitlab_ci = _collect_gitlab_ci(gitlab_start, gitlab_end)

    return {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "repository": f"{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}",
        "github_prs": github_prs,
        "github_issues": github_issues,
        "gitlab_ci": gitlab_ci,
    }


def _sample_context(now: datetime.datetime | None = None) -> dict[str, Any]:
    """Return synthetic context for offline/demo runs."""
    week_start, week_end = _previous_week_bounds(now)
    return {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "repository": f"{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}",
        "github_prs": {
            "closed_prs": [
                {
                    "number": 42,
                    "title": "Fix boundary handling",
                    "author": "alice",
                    "url": f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/pull/42",
                    "state_reason": "merged",
                    "merged_at": "2024-07-03T10:00:00Z",
                    "closed_at": "2024-07-03T10:00:00Z",
                    "body": "This PR fixes boundary handling.",
                    "commits": [
                        {
                            "sha": "abc1234",
                            "message": "Fix boundary handling",
                            "url": f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/commit/abc1234",
                        }
                    ],
                    "comments": [],
                    "review_comments": [],
                }
            ],
            "active_prs": [
                {
                    "number": 43,
                    "title": "Add new stencil",
                    "author": "bob",
                    "url": f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/pull/43",
                    "updated_at": "2024-07-05T12:00:00Z",
                    "commits": [
                        {
                            "sha": "def5678",
                            "message": "Add new stencil",
                            "url": f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/commit/def5678",
                        }
                    ],
                    "comments": [],
                    "review_comments": [],
                }
            ],
            "inactive_prs": [
                {
                    "number": 44,
                    "title": "Refactor old module",
                    "author": "carol",
                    "url": f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/pull/44",
                    "updated_at": "2024-06-01T12:00:00Z",
                }
            ],
        },
        "github_issues": {
            "opened_issues": [
                {
                    "number": 101,
                    "title": "Bug report",
                    "author": "dave",
                    "url": f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues/101",
                    "created_at": "2024-07-04T09:00:00Z",
                    "body_snippet": "Something is broken.",
                }
            ],
            "closed_issues": [
                {
                    "number": 100,
                    "title": "Resolved issue",
                    "author": "eve",
                    "url": f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues/100",
                    "closed_at": "2024-07-02T15:00:00Z",
                    "state_reason": "completed",
                }
            ],
        },
        "gitlab_ci": {
            "status": "success",
            "url": GITLAB_PIPELINE_URL_TEMPLATE,
            "pipeline_id": 123,
            "created_at": "2024-07-06T01:00:00Z",
            "message": None,
            "failed_jobs": [],
            "running_jobs": [],
        },
    }


@cli.command(name="generate")
def generate_cmd(
    *,
    output_dir: Annotated[
        pathlib.Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory where context and generated summary files are written.",
        ),
    ] = pathlib.Path("weekly_slack_summary_output"),
    dummy_input_data: Annotated[
        bool,
        typer.Option(
            "--dummy-input-data",
            help="Use synthetic fixture data instead of live GitHub/GitLab API calls.",
        ),
    ] = False,
    dummy_summarization: Annotated[
        bool,
        typer.Option(
            "--dummy-summarization",
            help="Use the deterministic direct Markdown formatter instead of invoking OpenCode.",
        ),
    ] = False,
    dummy_output: Annotated[
        bool,
        typer.Option(
            "--dummy-output",
            help="Skip posting to Slack and print the final Markdown summary to stdout.",
        ),
    ] = False,
    github_token: Annotated[
        str | None,
        typer.Option("--github-token", help="GitHub API token (defaults to GITHUB_TOKEN env var)."),
    ] = None,
    slack_webhook_url: Annotated[
        str | None,
        typer.Option("--slack-webhook-url", help="Slack incoming webhook URL."),
    ] = None,
    slack_channel: Annotated[
        str | None,
        typer.Option("--slack-channel", help="Optional Slack channel override."),
    ] = None,
) -> None:
    """Collect activity data and produce a weekly Slack summary."""
    output_dir = output_dir.resolve()

    if dummy_input_data:
        typer.echo("Using dummy input data...")
        context = _sample_context()
    else:
        token = github_token or _github_token()
        typer.echo("Collecting GitHub activity and GitLab CI status...")
        context = _collect_all(token=token)
    markdown_context = _format_context_markdown(
        datetime.datetime.fromisoformat(context["week_start"]),
        datetime.datetime.fromisoformat(context["week_end"]),
        context["github_prs"],
        context["github_issues"],
        context["gitlab_ci"],
    )
    json_path, context_md_path = _write_context_files(output_dir, context, markdown_context)
    typer.echo(f"Wrote context: {json_path}")
    typer.echo(f"Wrote context markdown: {context_md_path}")

    summary_path = output_dir / "weekly_slack_summary.md"
    if dummy_summarization:
        summary_path.write_text(markdown_context, encoding="utf-8")
        typer.echo(f"Wrote summary (dummy summarization): {summary_path}")
    else:
        instructions = INSTRUCTIONS_FILE
        if not instructions.exists():
            raise RuntimeError(f"OpenCode instructions file not found: {instructions}")
        typer.echo("Running OpenCode to generate polished summary...")
        _run_opencode(instructions, context_md_path, summary_path)
        typer.echo(f"Wrote summary: {summary_path}")

    if dummy_output:
        summary_text = summary_path.read_text(encoding="utf-8")
        typer.echo(summary_text)
        typer.echo("Dummy output mode: skipping Slack post.")
        raise typer.Exit(code=0)

    webhook = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        typer.echo(
            "Error: no Slack webhook URL provided; summary saved but not posted.",
            err=True,
        )
        raise typer.Exit(code=1)

    channel = slack_channel or os.environ.get("SLACK_CHANNEL")
    summary_text = summary_path.read_text(encoding="utf-8")
    _post_to_slack(webhook, channel, summary_text)
    typer.echo("Summary posted to Slack.")


@cli.command(name="version")
def version_cmd() -> None:
    """Print the script version."""
    typer.echo("weekly_slack_summary 0.1.0")


if __name__ == "__main__":
    sys.exit(cli())
