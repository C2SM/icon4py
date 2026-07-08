# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the GitLab CI child pipeline generator."""

from __future__ import annotations

import generate_ci_pipeline as gcp
import pytest
import yaml


class _SubprocessResult:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_parse_collected_count_basic():
    assert gcp._parse_collected_count("collected 42 items") == 42


def test_parse_collected_count_with_selected():
    output = "collected 42 items / 5 deselected / 37 selected"
    assert gcp._parse_collected_count(output) == 37


def test_parse_collected_count_zero():
    assert gcp._parse_collected_count("collected 0 items") == 0


def test_parse_collected_count_no_match():
    assert gcp._parse_collected_count("no summary here") is None


def test_parse_collected_count_uses_last_match():
    output = "collected 1 item\ncollected 42 items / 5 deselected / 37 selected"
    assert gcp._parse_collected_count(output) == 37


def test_collection_env():
    env = gcp._collection_env()
    assert env["ICON4PY_NOX_USE_ACTIVE_VENV"] == "1"
    assert env["ICON4PY_ENABLE_GRID_DOWNLOAD"] == "false"
    assert env["ICON4PY_ENABLE_TESTDATA_DOWNLOAD"] == "false"
    assert env["GT4PY_BUILD_CACHE_LIFETIME"] == "persistent"


def test_run_nox_collection_constructs_command(monkeypatch):
    captured: dict = {}

    def mock_run(cmd, *, capture_output, text, timeout, env, check):
        captured["cmd"] = cmd
        captured["env"] = env
        return _SubprocessResult(0, stdout="collected 3 items\n")

    monkeypatch.setattr("subprocess.run", mock_run)

    env = {"ICON4PY_NOX_USE_ACTIVE_VENV": "1"}
    count = gcp._run_nox_collection(
        "test_model-3.13(basic, common)",
        ["--collect-only", "-n0", "--backend=dace_cpu", "--level=unit"],
        env,
        300,
    )
    assert count == 3
    assert captured["cmd"] == [
        "nox",
        "-s",
        "test_model-3.13(basic, common)",
        "--",
        "--collect-only",
        "-n0",
        "--backend=dace_cpu",
        "--level=unit",
    ]
    assert captured["env"]["ICON4PY_NOX_USE_ACTIVE_VENV"] == "1"


def test_run_nox_collection_returns_none_on_nonzero_exit(monkeypatch):
    def mock_run(cmd, **kwargs):
        return _SubprocessResult(1)

    monkeypatch.setattr("subprocess.run", mock_run)
    assert gcp._run_nox_collection("session", ["--collect-only"], {}, 300) is None


def test_run_nox_collection_returns_none_on_timeout(monkeypatch):
    def mock_run(cmd, **kwargs):
        raise TimeoutError

    monkeypatch.setattr("subprocess.run", mock_run)
    assert gcp._run_nox_collection("session", ["--collect-only"], {}, 300) is None


def test_run_nox_collection_returns_none_on_unparseable_output(monkeypatch):
    def mock_run(cmd, **kwargs):
        return _SubprocessResult(0, stdout="no summary")

    monkeypatch.setattr("subprocess.run", mock_run)
    assert gcp._run_nox_collection("session", ["--collect-only"], {}, 300) is None


def test_filter_removes_zero_test_cells(monkeypatch):
    """Only the matrix cell that reports tests survives collection filtering."""

    def mock_run(session: str, args: list[str], env: dict, timeout: float) -> int | None:
        if (
            "common" in session
            and "basic" in session
            and "--level=unit" in args
            and "--backend=dace_cpu" in args
        ):
            return 1
        return 0

    monkeypatch.setattr("generate_ci_pipeline._run_nox_collection", mock_run)

    output = gcp._generate_child_pipeline(
        sessions="model",
        model_subpackages="common",
        model_subsets="basic",
        backends="dace_cpu,embedded",
        levels="unit,integration",
    )
    pipeline = yaml.safe_load(output)
    matrix = pipeline["test_model_basic_aarch64"]["parallel"]["matrix"]
    assert matrix == [{"MODEL_SUBPACKAGE": "common", "BACKEND": "dace_cpu", "LEVEL": "unit"}]


def test_filter_keeps_cells_on_collection_failure(monkeypatch):
    """Collection failures are treated conservatively: keep the matrix entry."""

    def mock_run(session: str, args: list[str], env: dict, timeout: float) -> int | None:
        return None

    monkeypatch.setattr("generate_ci_pipeline._run_nox_collection", mock_run)

    output = gcp._generate_child_pipeline(
        sessions="model",
        model_subpackages="common",
        model_subsets="basic",
        backends="dace_cpu",
        levels="unit",
    )
    pipeline = yaml.safe_load(output)
    matrix = pipeline["test_model_basic_aarch64"]["parallel"]["matrix"]
    assert matrix == [{"MODEL_SUBPACKAGE": "common", "BACKEND": "dace_cpu", "LEVEL": "unit"}]


def test_skip_collection_env_var(monkeypatch):
    """ICON4PY_CI_SKIP_COLLECTION bypasses collection and emits all cells."""
    monkeypatch.setenv("ICON4PY_CI_SKIP_COLLECTION", "1")

    output = gcp._generate_child_pipeline(
        sessions="model",
        model_subpackages="common",
        model_subsets="basic",
        backends="dace_cpu,embedded",
        levels="unit,integration",
    )
    pipeline = yaml.safe_load(output)
    matrix = pipeline["test_model_basic_aarch64"]["parallel"]["matrix"]
    assert len(matrix) == 4


def test_tools_cells_use_correct_session(monkeypatch):
    """Tools selections generate the expected nox session name."""
    captured: dict = {}

    def mock_run(session: str, args: list[str], env: dict, timeout: float) -> int | None:
        captured["session"] = session
        captured["args"] = args
        return 1

    monkeypatch.setattr("generate_ci_pipeline._run_nox_collection", mock_run)

    gcp._generate_child_pipeline(
        sessions="tools",
        tools_subsets="unittest",
    )
    assert captured["session"] == "test_tools_and_bindings(unittest)"
    assert "--datatest-skip" in captured["args"]


def test_model_mpi_cells_use_correct_session(monkeypatch):
    """MPI cells use the test_model_mpi nox session."""
    captured: dict = {}

    def mock_run(session: str, args: list[str], env: dict, timeout: float) -> int | None:
        captured["session"] = session
        return 1

    monkeypatch.setattr("generate_ci_pipeline._run_nox_collection", mock_run)

    gcp._generate_child_pipeline(
        sessions="model_mpi",
        model_mpi_subpackages="common",
        model_mpi_subsets="basic",
        backends="dace_cpu",
        levels="unit",
    )
    assert captured["session"] == "test_model_mpi(basic, common)"
