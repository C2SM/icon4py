# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the GitLab CI child pipeline generator."""

from __future__ import annotations

from concurrent.futures import Future

import generate_ci_pipeline as gcp
import pytest
import yaml


def _make_cell(name: str) -> gcp._MatrixCell:
    return gcp._MatrixCell(
        job_name=name,
        extends=".test",
        variables={},
        matrix={"NAME": name},
        session=f"test({name})",
        pytest_args=[],
    )


def _future_with_result(value: int | BaseException) -> Future:
    future: Future = Future()
    if isinstance(value, BaseException):
        future.set_exception(value)
    else:
        future.set_result(value)
    return future


def _fake_executor_factory(futures: list[Future]):
    class _FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, *args, **kwargs):
            return futures.pop(0)

    return _FakeExecutor


def _as_completed_timeout_immediately(futures, timeout=None):
    raise TimeoutError


def _as_completed_first_then_timeout(futures, timeout=None):
    yield from list(futures.keys())[:1]
    raise TimeoutError


class _SubprocessResult:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_parse_collected_count_basic():
    assert gcp._parse_collected_count("collected 42 items") == 42


def test_parse_collected_count_singular():
    assert gcp._parse_collected_count("collected 1 item") == 1


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


def test_total_timeout_drops_done_futures_with_zero_count(monkeypatch):
    """A done future reporting zero tests is dropped after total timeout."""
    cell = _make_cell("zero")
    futures = [_future_with_result(0)]
    monkeypatch.setattr(gcp, "ThreadPoolExecutor", _fake_executor_factory(futures))
    monkeypatch.setattr(gcp, "as_completed", _as_completed_timeout_immediately)

    assert gcp._collect_cells([cell]) == []


def test_total_timeout_keeps_done_futures_with_positive_count(monkeypatch):
    """Done futures not yet yielded with positive counts are kept on timeout."""
    cell_zero = _make_cell("zero")
    cell_positive = _make_cell("positive")
    futures = [_future_with_result(0), _future_with_result(5)]
    monkeypatch.setattr(gcp, "ThreadPoolExecutor", _fake_executor_factory(futures))
    monkeypatch.setattr(gcp, "as_completed", _as_completed_first_then_timeout)

    kept = gcp._collect_cells([cell_zero, cell_positive])
    assert kept == [cell_positive]


def test_total_timeout_keeps_done_futures_with_exception(monkeypatch):
    """Done futures that raised are kept conservatively after total timeout."""
    cell = _make_cell("failed")
    futures = [_future_with_result(RuntimeError("boom"))]
    monkeypatch.setattr(gcp, "ThreadPoolExecutor", _fake_executor_factory(futures))
    monkeypatch.setattr(gcp, "as_completed", _as_completed_timeout_immediately)

    assert gcp._collect_cells([cell]) == [cell]


def test_total_timeout_keeps_pending_futures(monkeypatch):
    """Pending futures are cancelled and their cells kept after total timeout."""
    cell = _make_cell("pending")
    pending = Future()
    futures = [pending]
    monkeypatch.setattr(gcp, "ThreadPoolExecutor", _fake_executor_factory(futures))
    monkeypatch.setattr(gcp, "as_completed", _as_completed_timeout_immediately)

    assert gcp._collect_cells([cell]) == [cell]
    assert pending.cancelled()


def test_total_timeout_no_duplicate_done_futures_with_positive_count(monkeypatch):
    """A done future yielded before timeout is kept exactly once."""
    cell_positive = _make_cell("positive")
    cell_other = _make_cell("other")
    futures = [_future_with_result(5), _future_with_result(0)]
    monkeypatch.setattr(gcp, "ThreadPoolExecutor", _fake_executor_factory(futures))
    monkeypatch.setattr(gcp, "as_completed", _as_completed_first_then_timeout)

    kept = gcp._collect_cells([cell_positive, cell_other])
    assert kept == [cell_positive]
