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

import generate_weekly_summary_opencode_config
import weekly_slack_summary


def test_previous_week_bounds_returns_monday_to_sunday():
    # 2024-07-08 is a Monday
    monday = datetime.datetime(2024, 7, 8, 12, 0, tzinfo=datetime.UTC)
    start, end = weekly_slack_summary._previous_week_bounds(monday)
    assert start == datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.UTC)
    assert end == datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.UTC)


def test_previous_week_bounds_sunday_boundary():
    # 2024-07-14 is a Sunday; the most recently completed week is still
    # 2024-07-01 to 2024-07-07 because the current week has not finished.
    sunday = datetime.datetime(2024, 7, 14, 23, 0, tzinfo=datetime.UTC)
    start, end = weekly_slack_summary._previous_week_bounds(sunday)
    assert start == datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.UTC)
    assert end == datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.UTC)


def test_current_week_bounds_from_thursday():
    # 2024-07-11 is a Thursday; the current week is Mon 2024-07-08 to
    # Sun 2024-07-14.
    thursday = datetime.datetime(2024, 7, 11, 12, 0, tzinfo=datetime.UTC)
    start, end = weekly_slack_summary._current_week_bounds(thursday)
    assert start == datetime.datetime(2024, 7, 8, 0, 0, tzinfo=datetime.UTC)
    assert end == datetime.datetime(2024, 7, 14, 23, 59, 59, tzinfo=datetime.UTC)


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
            if "/pulls/42" in decoded:
                return {
                    "number": 42,
                    "merged": True,
                    "merged_at": sample_closed_pr["merged_at"],
                    "merged_by": {"login": "alice"},
                }
            if "/pulls/44" in decoded:
                return {
                    "number": 44,
                    "comments": 2,
                    "review_comments": 1,
                    "draft": False,
                    "mergeable_state": "clean",
                }
            raise AssertionError(f"Unexpected URL: {url}")

        with mock.patch.object(
            weekly_slack_summary, "_github_api_request", side_effect=fake_request
        ):
            start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.UTC)
            end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.UTC)
            result = weekly_slack_summary._collect_github_prs(start, end, token="fake")

        assert len(result["merged_prs"]) == 1
        assert result["merged_prs"][0]["number"] == 42
        assert result["merged_prs"][0]["merged_at"] is not None
        assert len(result["closed_without_merge_prs"]) == 0
        assert len(result["active_prs"]) == 1
        assert result["active_prs"][0]["number"] == 43
        assert len(result["active_prs"][0]["commits"]) == 1
        assert result["active_prs"][0]["commits"][0]["sha"] == "abc1234"
        assert len(result["inactive_prs"]) == 1
        assert result["inactive_prs"][0]["number"] == 44
        assert result["inactive_prs"][0]["comments_count"] == 2
        assert result["inactive_prs"][0]["review_comments_count"] == 1
        assert len(result["inactive_pr_highlights"]) == 1
        assert result["inactive_pr_highlights"][0]["reasons"] == [
            "newest inactive",
            "oldest inactive",
            "most active inactive",
        ]


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
            start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.UTC)
            end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.UTC)
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
            start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.UTC)
            end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.UTC)
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
            if "/bridges" in url:
                return []
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

        with (
            mock.patch.object(
                weekly_slack_summary, "_gitlab_api_request", side_effect=fake_request
            ),
            mock.patch.object(
                weekly_slack_summary, "_gitlab_job_log", return_value="ERROR: test-dycore failed"
            ),
        ):
            start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.UTC)
            end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.UTC)
            result = weekly_slack_summary._collect_gitlab_ci(start, end)

        assert result["status"] == "failed"
        assert len(result["failed_jobs"]) == 1
        assert result["failed_jobs"][0]["name"] == "test-dycore"
        assert result["failed_jobs"][0]["log_snippet"] == "ERROR: test-dycore failed"

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
            if "/bridges" in url:
                return []
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
            start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.UTC)
            end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.UTC)
            result = weekly_slack_summary._collect_gitlab_ci(start, end)

        assert result["status"] == "running"
        assert len(result["running_jobs"]) == 1
        assert result["running_jobs"][0]["name"] == "test-diffusion"


class TestFormatContextMarkdown:
    def test_contains_all_sections(self):
        start = datetime.datetime(2024, 7, 1, 0, 0, tzinfo=datetime.UTC)
        end = datetime.datetime(2024, 7, 7, 23, 59, 59, tzinfo=datetime.UTC)
        ci_start = datetime.datetime(2024, 7, 8, 0, 0, tzinfo=datetime.UTC)
        github_prs = {
            "merged_prs": [
                {
                    "number": 1,
                    "title": "Merged PR",
                    "author": "alice",
                    "url": "https://github.com/C2SM/icon4py/pull/1",
                    "merged_at": "2024-07-03T10:00:00Z",
                    "commits": [],
                    "comments": [],
                    "review_comments": [],
                }
            ],
            "closed_without_merge_prs": [],
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
                    "comments_count": 0,
                    "review_comments_count": 0,
                    "draft": False,
                    "mergeable_state": "clean",
                }
            ],
            "inactive_pr_highlights": [
                {
                    "number": 3,
                    "title": "Inactive PR",
                    "author": "carol",
                    "url": "https://github.com/C2SM/icon4py/pull/3",
                    "updated_at": "2024-06-01T12:00:00Z",
                    "reasons": ["newest inactive", "oldest inactive"],
                    "draft": False,
                    "mergeable_state": "clean",
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
        gitlab_nightly_ci = {
            "status": "no_recent_pipeline",
            "url": "https://gitlab.com/pipelines/nightly",
            "message": "No nightly benchmarking pipeline found in the last 24 hours.",
            "failed_jobs": [],
            "running_jobs": [],
        }

        markdown = weekly_slack_summary._format_context_markdown(
            start,
            end,
            github_prs,
            github_issues,
            gitlab_ci,
            gitlab_nightly_ci=gitlab_nightly_ci,
            ci_week_start=ci_start,
        )

        assert "Merged PRs (1)" in markdown
        assert "Active Open PRs (1)" in markdown
        assert "Inactive Open PRs (1 total)" in markdown
        assert "Selection of up to 9 highlights" in markdown
        assert "(newest inactive & oldest inactive)" in markdown
        assert "Opened Issues (1)" in markdown
        assert "Closed Issues (1)" in markdown
        assert "CI window:** 2024-07-08 (Mon) to end of week UTC" in markdown
        assert "Status: **success**" in markdown


class TestInactivePRHighlights:
    def test_selects_newest_oldest_most_active_and_deduplicates(self):
        prs = [
            {
                "number": 1,
                "title": "PR one",
                "author": "a1",
                "url": "https://github.com/C2SM/icon4py/pull/1",
                "updated_at": "2024-07-10T00:00:00Z",
                "comments_count": 1,
                "review_comments_count": 0,
                "draft": False,
                "mergeable_state": "clean",
            },
            {
                "number": 2,
                "title": "PR two",
                "author": "a2",
                "url": "https://github.com/C2SM/icon4py/pull/2",
                "updated_at": "2024-07-05T00:00:00Z",
                "comments_count": 10,
                "review_comments_count": 0,
                "draft": False,
                "mergeable_state": "clean",
            },
            {
                "number": 3,
                "title": "PR three",
                "author": "a3",
                "url": "https://github.com/C2SM/icon4py/pull/3",
                "updated_at": "2024-07-01T00:00:00Z",
                "comments_count": 0,
                "review_comments_count": 0,
                "draft": False,
                "mergeable_state": "clean",
            },
            {
                "number": 4,
                "title": "PR four",
                "author": "a4",
                "url": "https://github.com/C2SM/icon4py/pull/4",
                "updated_at": "2024-07-08T00:00:00Z",
                "comments_count": 5,
                "review_comments_count": 0,
                "draft": False,
                "mergeable_state": "clean",
            },
            {
                "number": 5,
                "title": "PR five",
                "author": "a5",
                "url": "https://github.com/C2SM/icon4py/pull/5",
                "updated_at": "2024-06-15T00:00:00Z",
                "comments_count": 0,
                "review_comments_count": 0,
                "draft": False,
                "mergeable_state": "clean",
            },
            {
                "number": 6,
                "title": "PR six",
                "author": "a6",
                "url": "https://github.com/C2SM/icon4py/pull/6",
                "updated_at": "2024-06-20T00:00:00Z",
                "comments_count": 3,
                "review_comments_count": 0,
                "draft": False,
                "mergeable_state": "clean",
            },
        ]
        highlights = weekly_slack_summary._select_inactive_pr_highlights(prs)

        assert len(highlights) == 6
        assert [h["number"] for h in highlights] == [1, 4, 2, 5, 6, 3]
        assert highlights[0]["reasons"] == ["newest inactive"]
        assert set(highlights[1]["reasons"]) == {"newest inactive", "most active inactive"}
        assert set(highlights[2]["reasons"]) == {
            "newest inactive",
            "most active inactive",
        }
        assert highlights[3]["reasons"] == ["oldest inactive"]
        assert set(highlights[4]["reasons"]) == {"oldest inactive", "most active inactive"}
        assert highlights[5]["reasons"] == ["oldest inactive"]

    def test_empty_inactive_prs_returns_empty_highlights(self):
        assert weekly_slack_summary._select_inactive_pr_highlights([]) == []


class TestRunOpenCode:
    def test_fails_when_opencode_missing(self, monkeypatch):
        monkeypatch.setattr(weekly_slack_summary.shutil, "which", lambda _name: None)
        with pytest.raises(RuntimeError, match="opencode"):
            weekly_slack_summary._run_opencode(
                pathlib.Path("instructions.md"),
                pathlib.Path("context.md"),
                pathlib.Path("out.md"),
            )

    def test_runs_opencode_and_writes_stdout(self, tmp_path, monkeypatch):
        instructions = tmp_path / "instructions.md"
        context = tmp_path / "context.md"
        output = tmp_path / "out.md"
        instructions.write_text("instructions", encoding="utf-8")
        context.write_text("context", encoding="utf-8")

        monkeypatch.setattr(weekly_slack_summary.shutil, "which", lambda _name: "/usr/bin/opencode")

        def fake_run(cmd, *, capture_output, check, text):
            assert cmd == [
                "opencode",
                "run",
                "Generate the weekly Slack summary following the attached instructions "
                f"and context. Write the final summary to: {output}",
                "--dir",
                str(output.parent),
                "--file",
                str(instructions),
                "--file",
                str(context),
            ]
            assert capture_output is True
            assert check is True
            assert text is True
            output.write_text("polished summary", encoding="utf-8")
            return mock.Mock(stdout="polished summary", stderr="")

        monkeypatch.setattr(weekly_slack_summary.subprocess, "run", fake_run)

        weekly_slack_summary._run_opencode(instructions, context, output)
        assert output.read_text(encoding="utf-8") == "polished summary"


class TestGenerateCommand:
    @pytest.fixture
    def output_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            weekly_slack_summary.tempfile,
            "mkdtemp",
            lambda *args, **kwargs: str(tmp_path),
        )
        return tmp_path

    def test_dummy_output_writes_files(self, output_dir, monkeypatch):
        def fake_collect_all(*args, **kwargs):
            return {
                "week_start": "2024-07-01T00:00:00+00:00",
                "week_end": "2024-07-07T23:59:59+00:00",
                "ci_week_start": "2024-07-08T00:00:00+00:00",
                "repository": "C2SM/icon4py",
                "github_prs": {
                    "merged_prs": [],
                    "closed_without_merge_prs": [],
                    "active_prs": [],
                    "inactive_prs": [],
                    "inactive_pr_highlights": [],
                },
                "github_issues": {"opened_issues": [], "closed_issues": []},
                "gitlab_ci": {
                    "status": "no_recent_pipeline",
                    "url": "https://gitlab.com/pipelines",
                    "message": "none",
                    "failed_jobs": [],
                    "running_jobs": [],
                },
                "gitlab_nightly_ci": {
                    "status": "no_recent_pipeline",
                    "url": "https://gitlab.com/pipelines/nightly",
                    "message": "none",
                    "failed_jobs": [],
                    "running_jobs": [],
                },
            }

        monkeypatch.setattr(weekly_slack_summary, "_collect_all", fake_collect_all)

        with pytest.raises(typer.Exit) as exc_info:
            weekly_slack_summary.generate_cmd(
                dummy_input_data=False,
                dummy_summarization=True,
                dummy_output=True,
            )
        assert exc_info.value.exit_code == 0

        context_json = output_dir / "weekly_slack_summary_context.json"
        context_md = output_dir / "weekly_slack_summary_context.md"
        summary_md = output_dir / "weekly_slack_summary.md"
        assert context_json.exists()
        assert context_md.exists()
        assert summary_md.exists()

        data = json.loads(context_json.read_text())
        assert data["repository"] == "C2SM/icon4py"

    def test_dummy_input_data_and_output_writes_files(self, output_dir):
        with pytest.raises(typer.Exit) as exc_info:
            weekly_slack_summary.generate_cmd(
                dummy_input_data=True,
                dummy_summarization=True,
                dummy_output=True,
            )
        assert exc_info.value.exit_code == 0

        context_json = output_dir / "weekly_slack_summary_context.json"
        context_md = output_dir / "weekly_slack_summary_context.md"
        summary_md = output_dir / "weekly_slack_summary.md"
        assert context_json.exists()
        assert context_md.exists()
        assert summary_md.exists()

        data = json.loads(context_json.read_text())
        assert data["repository"] == "C2SM/icon4py"
        assert len(data["github_prs"]["merged_prs"]) == 1
        assert len(data["github_prs"]["active_prs"]) == 1
        assert len(data["github_prs"]["inactive_prs"]) == 1
        assert len(data["github_issues"]["opened_issues"]) == 1
        assert len(data["github_issues"]["closed_issues"]) == 1
        assert data["gitlab_ci"]["status"] == "success"

        markdown = summary_md.read_text()
        assert "Merged PRs" in markdown
        assert "Active Open PRs" in markdown
        assert "Inactive Open PRs" in markdown
        assert "Opened Issues" in markdown
        assert "Closed Issues" in markdown
        assert "GitLab Weekly CI" in markdown

    def test_dummy_output_prints_summary(self, output_dir, capsys):
        with pytest.raises(typer.Exit) as exc_info:
            weekly_slack_summary.generate_cmd(
                dummy_input_data=True,
                dummy_summarization=True,
                dummy_output=True,
            )
        assert exc_info.value.exit_code == 0

        summary_md = output_dir / "weekly_slack_summary.md"
        assert summary_md.exists()

        captured = capsys.readouterr()
        assert summary_md.read_text(encoding="utf-8") in captured.out

    def test_default_mode_calls_opencode_and_posts(self, output_dir, monkeypatch):
        def fake_collect_all(*args, **kwargs):
            return {
                "week_start": "2024-07-01T00:00:00+00:00",
                "week_end": "2024-07-07T23:59:59+00:00",
                "ci_week_start": "2024-07-08T00:00:00+00:00",
                "repository": "C2SM/icon4py",
                "github_prs": {
                    "merged_prs": [],
                    "closed_without_merge_prs": [],
                    "active_prs": [],
                    "inactive_prs": [],
                    "inactive_pr_highlights": [],
                },
                "github_issues": {"opened_issues": [], "closed_issues": []},
                "gitlab_ci": {
                    "status": "no_recent_pipeline",
                    "url": "https://gitlab.com/pipelines",
                    "message": "none",
                    "failed_jobs": [],
                    "running_jobs": [],
                },
                "gitlab_nightly_ci": {
                    "status": "no_recent_pipeline",
                    "url": "https://gitlab.com/pipelines/nightly",
                    "message": "none",
                    "failed_jobs": [],
                    "running_jobs": [],
                },
            }

        monkeypatch.setattr(weekly_slack_summary, "_collect_all", fake_collect_all)

        opencode_calls: list[pathlib.Path] = []

        def fake_run_opencode(instructions, context, output):
            opencode_calls.append(output)
            output.write_text("polished summary", encoding="utf-8")

        monkeypatch.setattr(weekly_slack_summary, "_run_opencode", fake_run_opencode)

        slack_calls: list[tuple[str, str]] = []

        def fake_post_to_slack(webhook_url, markdown):
            slack_calls.append((webhook_url, markdown))

        monkeypatch.setattr(weekly_slack_summary, "_post_to_slack", fake_post_to_slack)

        weekly_slack_summary.generate_cmd(
            slack_webhook_url="https://hooks.slack.com/test",
        )

        summary_md = output_dir / "weekly_slack_summary.md"
        assert summary_md.exists()
        assert summary_md.read_text(encoding="utf-8") == "polished summary"
        assert opencode_calls == [summary_md]
        assert slack_calls == [("https://hooks.slack.com/test", "polished summary")]

    def test_over_limit_summary_posts_with_warning_and_oversized_file(
        self, output_dir, monkeypatch, capsys
    ):
        def fake_collect_all(*args, **kwargs):
            return {
                "week_start": "2024-07-01T00:00:00+00:00",
                "week_end": "2024-07-07T23:59:59+00:00",
                "ci_week_start": "2024-07-08T00:00:00+00:00",
                "repository": "C2SM/icon4py",
                "github_prs": {
                    "merged_prs": [],
                    "closed_without_merge_prs": [],
                    "active_prs": [],
                    "inactive_prs": [],
                    "inactive_pr_highlights": [],
                },
                "github_issues": {"opened_issues": [], "closed_issues": []},
                "gitlab_ci": {
                    "status": "no_recent_pipeline",
                    "url": "https://gitlab.com/pipelines",
                    "message": "none",
                    "failed_jobs": [],
                    "running_jobs": [],
                },
                "gitlab_nightly_ci": {
                    "status": "no_recent_pipeline",
                    "url": "https://gitlab.com/pipelines/nightly",
                    "message": "none",
                    "failed_jobs": [],
                    "running_jobs": [],
                },
            }

        monkeypatch.setattr(weekly_slack_summary, "_collect_all", fake_collect_all)

        long_summary = "x" * 3801

        def fake_run_opencode(instructions, context, output):
            output.write_text(long_summary, encoding="utf-8")

        monkeypatch.setattr(weekly_slack_summary, "_run_opencode", fake_run_opencode)

        slack_calls: list[tuple[str, str]] = []

        def fake_post_to_slack(webhook_url, markdown):
            slack_calls.append((webhook_url, markdown))

        monkeypatch.setattr(weekly_slack_summary, "_post_to_slack", fake_post_to_slack)

        weekly_slack_summary.generate_cmd(
            slack_webhook_url="https://hooks.slack.com/test",
        )

        summary_md = output_dir / "weekly_slack_summary.md"
        assert summary_md.exists()
        assert summary_md.read_text(encoding="utf-8") == long_summary
        assert slack_calls == [("https://hooks.slack.com/test", long_summary)]

        oversized_md = output_dir / "weekly_slack_summary_oversized.md"
        assert oversized_md.exists()
        assert oversized_md.read_text(encoding="utf-8") == long_summary

        captured = capsys.readouterr()
        assert "Warning: generated summary exceeds recommended limits" in captured.err

    def test_missing_webhook_exits_nonzero(self, output_dir, monkeypatch):
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

        def fake_collect_all(*args, **kwargs):
            return {
                "week_start": "2024-07-01T00:00:00+00:00",
                "week_end": "2024-07-07T23:59:59+00:00",
                "ci_week_start": "2024-07-08T00:00:00+00:00",
                "repository": "C2SM/icon4py",
                "github_prs": {
                    "merged_prs": [],
                    "closed_without_merge_prs": [],
                    "active_prs": [],
                    "inactive_prs": [],
                    "inactive_pr_highlights": [],
                },
                "github_issues": {"opened_issues": [], "closed_issues": []},
                "gitlab_ci": {
                    "status": "no_recent_pipeline",
                    "url": "https://gitlab.com/pipelines",
                    "message": "none",
                    "failed_jobs": [],
                    "running_jobs": [],
                },
                "gitlab_nightly_ci": {
                    "status": "no_recent_pipeline",
                    "url": "https://gitlab.com/pipelines/nightly",
                    "message": "none",
                    "failed_jobs": [],
                    "running_jobs": [],
                },
            }

        monkeypatch.setattr(weekly_slack_summary, "_collect_all", fake_collect_all)

        with pytest.raises(typer.Exit) as exc_info:
            weekly_slack_summary.generate_cmd(
                dummy_input_data=False,
                dummy_summarization=True,
                dummy_output=False,
            )
        assert exc_info.value.exit_code == 1

        summary_md = output_dir / "weekly_slack_summary.md"
        assert summary_md.exists()


class TestVerifySummaryLength:
    def test_under_limit_does_not_warn_or_write_oversized_file(self, tmp_path, capsys):
        weekly_slack_summary._verify_summary_length("short summary", tmp_path)
        captured = capsys.readouterr()
        assert captured.err == ""
        assert not (tmp_path / "weekly_slack_summary_oversized.md").exists()

    def test_over_limit_warns_and_writes_oversized_file(self, tmp_path, capsys):
        long_summary = "x" * 3801
        weekly_slack_summary._verify_summary_length(long_summary, tmp_path)
        captured = capsys.readouterr()
        assert "Warning: generated summary exceeds recommended limits" in captured.err
        assert "(3801 characters, 1 lines" in captured.err
        oversized_path = tmp_path / "weekly_slack_summary_oversized.md"
        assert oversized_path.exists()
        assert oversized_path.read_text(encoding="utf-8") == long_summary


class TestGenerateOpenCodeConfig:
    def test_writes_config_with_model_and_base_url(self, tmp_path):
        output = tmp_path / "opencode-config.json"
        generate_weekly_summary_opencode_config.generate_cmd(
            output_path=output,
            base_url="https://api.inference.cscs.ch/v1",
            model_id="google/gemma-4-31B-it",
        )
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["model"] == "cscs-inference/google/gemma-4-31B-it"
        assert (
            data["provider"]["cscs-inference"]["options"]["baseURL"]
            == "https://api.inference.cscs.ch/v1"
        )
        assert (
            data["provider"]["cscs-inference"]["models"]["google/gemma-4-31B-it"]["name"]
            == "google/gemma-4-31B-it"
        )
        assert (
            data["provider"]["cscs-inference"]["options"]["apiKey"]
            == "{env:CSCS_INFERENCE_API_KEY}"
        )
