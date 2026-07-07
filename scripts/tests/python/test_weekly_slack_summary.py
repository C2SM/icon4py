# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the weekly Slack summary helper script."""

from __future__ import annotations

import datetime
import json
import pathlib
import urllib.parse
from typing import Any
from unittest import mock

import pytest
import typer
import weekly_slack_summary


def test_previous_week_bounds_returns_monday_to_sunday():
    # 2024-07-08 is a Monday
    monday = datetime.datetime(2024, 7, 8, 12, 0, tzinfo=datetime.timezone.utc)
    start, end = weekly_slack_summary._previous_week_bounds(monday)
    assert start == datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)
    assert end == datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.timezone.utc)


def test_previous_week_bounds_sunday_boundary():
    # 2024-07-14 is a Sunday; the most recently completed week is still
    # 2024-07-01 to 2024-07-07 because the current week has not finished.
    sunday = datetime.datetime(2024, 7, 14, 23, 0, tzinfo=datetime.timezone.utc)
    start, end = weekly_slack_summary._previous_week_bounds(sunday)
    assert start == datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)
    assert end == datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.timezone.utc)


@pytest.fixture
def sample_closed_pr():
    return {
        "number": 42,
        "title": "Fix boundary handling",
        "html_url": "https://github.com/C2SM/icon4py/pull/42",
        "user": {"login": "alice"},
        "state_reason": "merged",
        "merged_at": "2024-07-03T10:00:00Z",
        "closed_at": "2024-07-03T10:00:00Z",
        "body": "This PR fixes boundary handling.",
    }


@pytest.fixture
def sample_open_pr():
    return {
        "number": 43,
        "title": "Add new stencil",
        "html_url": "https://github.com/C2SM/icon4py/pull/43",
        "user": {"login": "bob"},
        "updated_at": "2024-07-05T12:00:00Z",
    }


@pytest.fixture
def sample_inactive_pr():
    return {
        "number": 44,
        "title": "Refactor old module",
        "html_url": "https://github.com/C2SM/icon4py/pull/44",
        "user": {"login": "carol"},
        "updated_at": "2024-06-01T12:00:00Z",
    }


@pytest.fixture
def sample_issue_opened():
    return {
        "number": 101,
        "title": "Bug report",
        "html_url": "https://github.com/C2SM/icon4py/issues/101",
        "user": {"login": "dave"},
        "created_at": "2024-07-04T09:00:00Z",
        "body": "Something is broken.",
    }


@pytest.fixture
def sample_issue_closed():
    return {
        "number": 100,
        "title": "Resolved issue",
        "html_url": "https://github.com/C2SM/icon4py/issues/100",
        "user": {"login": "eve"},
        "closed_at": "2024-07-02T15:00:00Z",
        "state_reason": "completed",
    }


class TestCollectGitHubPRs:
    def test_collects_closed_and_open_prs(
        self,
        sample_closed_pr,
        sample_open_pr,
        sample_inactive_pr,
    ):
        def fake_request(url: str, **kwargs):
            decoded = urllib.parse.unquote(url)
            if "/search/issues" in decoded:
                if "is:pr is:closed" in decoded:
                    return {"items": [sample_closed_pr]}
                if (
                    "is:pr is:open" in decoded
                    and "updated:" in decoded
                    and "updated:<" not in decoded
                ):
                    return {"items": [sample_open_pr]}
                if "updated:<" in decoded:
                    return {"items": [sample_inactive_pr]}
            if "/commits" in decoded:
                return [
                    {
                        "sha": "abc1234",
                        "commit": {"message": "Fix boundary handling\n\nDetails"},
                        "html_url": "https://github.com/C2SM/icon4py/commit/abc1234",
                    }
                ]
            if "/comments" in decoded or "/review_comments" in decoded:
                return []
            raise AssertionError(f"Unexpected URL: {url}")

        with mock.patch.object(
            weekly_slack_summary, "_github_api_request", side_effect=fake_request
        ):
            start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)
            end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.timezone.utc)
            result = weekly_slack_summary._collect_github_prs(start, end, token="fake")

        assert len(result["closed_prs"]) == 1
        assert result["closed_prs"][0]["number"] == 42
        assert result["closed_prs"][0]["merged_at"] is not None
        assert len(result["active_prs"]) == 1
        assert result["active_prs"][0]["number"] == 43
        assert len(result["inactive_prs"]) == 1
        assert result["inactive_prs"][0]["number"] == 44


class TestCollectGitHubIssues:
    def test_collects_opened_and_closed_issues(
        self,
        sample_issue_opened,
        sample_issue_closed,
    ):
        def fake_request(url: str, **kwargs):
            decoded = urllib.parse.unquote(url)
            assert "/search/issues" in decoded
            if "is:issue " in decoded and "created:" in decoded:
                return {"items": [sample_issue_opened]}
            if "is:issue is:closed" in decoded:
                return {"items": [sample_issue_closed]}
            raise AssertionError(f"Unexpected URL: {url}")

        with mock.patch.object(
            weekly_slack_summary, "_github_api_request", side_effect=fake_request
        ):
            start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)
            end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.timezone.utc)
            result = weekly_slack_summary._collect_github_issues(start, end, token="fake")

        assert len(result["opened_issues"]) == 1
        assert result["opened_issues"][0]["number"] == 101
        assert len(result["closed_issues"]) == 1
        assert result["closed_issues"][0]["state_reason"] == "completed"


class TestCollectGitLabCI:
    def test_no_recent_pipeline(self):
        def fake_request(url: str):
            assert "/pipelines" in url
            return []

        with mock.patch.object(
            weekly_slack_summary, "_gitlab_api_request", side_effect=fake_request
        ):
            start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)
            end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.timezone.utc)
            result = weekly_slack_summary._collect_gitlab_ci(start, end)

        assert result["status"] == "no_recent_pipeline"
        assert result["failed_jobs"] == []
        assert result["running_jobs"] == []

    def test_failed_pipeline_lists_failed_jobs(self):
        def fake_request(url: str):
            if "/jobs" in url:
                return [
                    {
                        "id": 1000,
                        "name": "test-dycore",
                        "stage": "test",
                        "status": "failed",
                        "web_url": "https://gitlab.com/jobs/1000",
                        "failure_reason": "test_failure",
                    },
                    {
                        "id": 1001,
                        "name": "build",
                        "stage": "build",
                        "status": "success",
                        "web_url": "https://gitlab.com/jobs/1001",
                        "failure_reason": None,
                    },
                ]
            if "/pipelines" in url:
                return [
                    {
                        "id": 123,
                        "status": "failed",
                        "created_at": "2024-07-06T02:00:00Z",
                        "web_url": "https://gitlab.com/pipelines/123",
                    }
                ]
            raise AssertionError(f"Unexpected URL: {url}")

        with mock.patch.object(
            weekly_slack_summary, "_gitlab_api_request", side_effect=fake_request
        ):
            start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)
            end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.timezone.utc)
            result = weekly_slack_summary._collect_gitlab_ci(start, end)

        assert result["status"] == "failed"
        assert len(result["failed_jobs"]) == 1
        assert result["failed_jobs"][0]["name"] == "test-dycore"

    def test_running_pipeline_lists_running_jobs(self):
        def fake_request(url: str):
            if "/jobs" in url:
                return [
                    {
                        "id": 2000,
                        "name": "test-diffusion",
                        "stage": "test",
                        "status": "running",
                        "web_url": "https://gitlab.com/jobs/2000",
                        "failure_reason": None,
                    }
                ]
            if "/pipelines" in url:
                return [
                    {
                        "id": 124,
                        "status": "running",
                        "created_at": "2024-07-06T02:00:00Z",
                        "web_url": "https://gitlab.com/pipelines/124",
                    }
                ]
            raise AssertionError(f"Unexpected URL: {url}")

        with mock.patch.object(
            weekly_slack_summary, "_gitlab_api_request", side_effect=fake_request
        ):
            start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)
            end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.timezone.utc)
            result = weekly_slack_summary._collect_gitlab_ci(start, end)

        assert result["status"] == "running"
        assert len(result["running_jobs"]) == 1
        assert result["running_jobs"][0]["name"] == "test-diffusion"


class TestFormatContextMarkdown:
    def test_contains_all_sections(self):
        start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)
        end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.timezone.utc)
        github_prs = {
            "closed_prs": [
                {
                    "number": 1,
                    "title": "Closed PR",
                    "author": "alice",
                    "url": "https://github.com/C2SM/icon4py/pull/1",
                    "merged_at": "2024-07-03T10:00:00Z",
                    "commits": [],
                    "comments": [],
                    "review_comments": [],
                }
            ],
            "active_prs": [
                {
                    "number": 2,
                    "title": "Active PR",
                    "author": "bob",
                    "url": "https://github.com/C2SM/icon4py/pull/2",
                    "updated_at": "2024-07-05T12:00:00Z",
                }
            ],
            "inactive_prs": [
                {
                    "number": 3,
                    "title": "Inactive PR",
                    "author": "carol",
                    "url": "https://github.com/C2SM/icon4py/pull/3",
                    "updated_at": "2024-06-01T12:00:00Z",
                }
            ],
        }
        github_issues = {
            "opened_issues": [
                {
                    "number": 10,
                    "title": "New issue",
                    "author": "dave",
                    "url": "https://github.com/C2SM/icon4py/issues/10",
                }
            ],
            "closed_issues": [
                {
                    "number": 11,
                    "title": "Fixed issue",
                    "author": "eve",
                    "url": "https://github.com/C2SM/icon4py/issues/11",
                    "state_reason": "completed",
                }
            ],
        }
        gitlab_ci = {
            "status": "success",
            "url": "https://gitlab.com/pipelines/123",
            "message": None,
            "failed_jobs": [],
            "running_jobs": [],
        }

        markdown = weekly_slack_summary._format_context_markdown(
            start, end, github_prs, github_issues, gitlab_ci
        )

        assert "Closed PRs (1)" in markdown
        assert "Active Open PRs (1)" in markdown
        assert "Inactive Open PRs (1)" in markdown
        assert "Opened Issues (1)" in markdown
        assert "Closed Issues (1)" in markdown
        assert "Status: **success**" in markdown


class TestRunOpenCode:
    def test_fails_when_opencode_missing(self, monkeypatch):
        monkeypatch.setattr(weekly_slack_summary.shutil, "which", lambda _name: None)
        with pytest.raises(RuntimeError, match="opencode"):
            weekly_slack_summary._run_opencode(
                pathlib.Path("instructions.md"),
                pathlib.Path("context.md"),
                pathlib.Path("out.md"),
            )


class TestGenerateCommand:
    def test_dry_run_writes_files(self, tmp_path, monkeypatch):
        def fake_collect_all(*args, **kwargs):
            return {
                "week_start": "2024-07-01T00:00:00+00:00",
                "week_end": "2024-07-07T23:59:59+00:00",
                "repository": "C2SM/icon4py",
                "github_prs": {
                    "closed_prs": [],
                    "active_prs": [],
                    "inactive_prs": [],
                },
                "github_issues": {"opened_issues": [], "closed_issues": []},
                "gitlab_ci": {
                    "status": "no_recent_pipeline",
                    "url": "https://gitlab.com/pipelines",
                    "message": "none",
                    "failed_jobs": [],
                    "running_jobs": [],
                },
            }

        monkeypatch.setattr(weekly_slack_summary, "_collect_all", fake_collect_all)

        with pytest.raises(typer.Exit) as exc_info:
            weekly_slack_summary.generate(
                output_dir=tmp_path,
                dry_run=True,
                skip_opencode=True,
            )
        assert exc_info.value.exit_code == 0

        context_json = tmp_path / "weekly_slack_summary_context.json"
        context_md = tmp_path / "weekly_slack_summary_context.md"
        summary_md = tmp_path / "weekly_slack_summary.md"
        assert context_json.exists()
        assert context_md.exists()
        assert summary_md.exists()

        data = json.loads(context_json.read_text())
        assert data["repository"] == "C2SM/icon4py"

    def test_offline_dry_run_writes_files(self, tmp_path):
        with pytest.raises(typer.Exit) as exc_info:
            weekly_slack_summary.generate(
                output_dir=tmp_path,
                dry_run=True,
                skip_opencode=True,
                offline=True,
            )
        assert exc_info.value.exit_code == 0

        summary_md = tmp_path / "weekly_slack_summary.md"
        assert summary_md.exists()
        markdown = summary_md.read_text()
        assert "Closed PRs" in markdown
        assert "Active Open PRs" in markdown
        assert "Inactive Open PRs" in markdown
        assert "Opened Issues" in markdown
        assert "Closed Issues" in markdown
        assert "GitLab Weekly CI" in markdown
