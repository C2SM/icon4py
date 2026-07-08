# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the GitLab CI child pipeline generator."""

from __future__ import annotations

import subprocess
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


def _future_with_result(value: bool | BaseException) -> Future:
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


class _SubprocessResult:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_collection_env():
    env = gcp._collection_env()
    assert env == {"ICON4PY_NOX_USE_ACTIVE_VENV": "1"}


def test_run_nox_collection_constructs_command(monkeypatch):
    captured: dict = {}

    def mock_run(cmd, *, capture_output, text, timeout, env, check):
        captured["cmd"] = cmd
        captured["env"] = env
        return _SubprocessResult(0)

    monkeypatch.setattr("subprocess.run", mock_run)

    env = {"ICON4PY_NOX_USE_ACTIVE_VENV": "1"}
    keep = gcp._run_nox_collection(
        "test_model(basic, common)",
        ["--collect-only", "-n0", "-p", "no:tach", "--backend=dace_cpu", "--level=unit"],
        env,
        300,
    )
    assert keep is True
    assert captured["cmd"] == [
        "nox",
        "-s",
        "test_model(basic, common)",
        "--",
        "--collect-only",
        "-n0",
        "-p",
        "no:tach",
        "--backend=dace_cpu",
        "--level=unit",
    ]
    assert captured["env"]["ICON4PY_NOX_USE_ACTIVE_VENV"] == "1"


def test_run_nox_collection_false_on_exit_1(monkeypatch):
    def mock_run(cmd, **kwargs):
        return _SubprocessResult(1)

    monkeypatch.setattr("subprocess.run", mock_run)
    assert gcp._run_nox_collection("session", ["--collect-only"], {}, 300) is False


def test_run_nox_collection_raises_on_nonzero_exit(monkeypatch):
    def mock_run(cmd, **kwargs):
        return _SubprocessResult(2, stdout="stdout text", stderr="stderr text")

    monkeypatch.setattr("subprocess.run", mock_run)
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        gcp._run_nox_collection("session", ["--collect-only"], {}, 300)
    assert exc_info.value.returncode == 2


def test_run_nox_collection_propagates_timeout(monkeypatch):
    def mock_run(cmd, **kwargs):
        raise TimeoutError

    monkeypatch.setattr("subprocess.run", mock_run)
    with pytest.raises(TimeoutError):
        gcp._run_nox_collection("session", ["--collect-only"], {}, 300)


def test_filter_removes_zero_test_cells(monkeypatch):
    """Only the matrix cell that reports tests survives collection filtering."""

    def mock_run(session: str, args: list[str], env: dict, timeout: float) -> bool:
        return (
            "common" in session
            and "basic" in session
            and "--level=unit" in args
            and "--backend=dace_cpu" in args
        )

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


def test_filter_raises_on_collection_failure(monkeypatch):
    """Collection failures abort pipeline generation."""

    def mock_run(session: str, args: list[str], env: dict, timeout: float) -> bool:
        raise RuntimeError("collection failed")

    monkeypatch.setattr("generate_ci_pipeline._run_nox_collection", mock_run)

    with pytest.raises(RuntimeError, match="collection failed"):
        gcp._generate_child_pipeline(
            sessions="model",
            model_subpackages="common",
            model_subsets="basic",
            backends="dace_cpu",
            levels="unit",
        )


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

    def mock_run(session: str, args: list[str], env: dict, timeout: float) -> bool:
        captured["session"] = session
        captured["args"] = args
        return True

    monkeypatch.setattr("generate_ci_pipeline._run_nox_collection", mock_run)

    gcp._generate_child_pipeline(
        sessions="tools",
        tools_subsets="unittest",
    )
    assert captured["session"] == "test_tools_and_bindings(unittest)"
    assert captured["args"][2:4] == ["-p", "no:tach"]


def test_model_mpi_cells_use_correct_session(monkeypatch):
    """MPI cells use the test_model_mpi nox session."""
    captured: dict = {}

    def mock_run(session: str, args: list[str], env: dict, timeout: float) -> bool:
        captured["session"] = session
        captured["args"] = args
        return True

    monkeypatch.setattr("generate_ci_pipeline._run_nox_collection", mock_run)

    gcp._generate_child_pipeline(
        sessions="model_mpi",
        model_mpi_subpackages="common",
        model_mpi_subsets="basic",
        backends="dace_cpu",
        levels="unit",
    )
    assert captured["session"] == "test_model_mpi(basic, common)"
    assert captured["args"][2:4] == ["-p", "no:tach"]


def test_collect_cells_keeps_true_and_drops_false(monkeypatch):
    """Cells returning True are kept; cells returning False are dropped."""
    keep_cell = _make_cell("keep")
    drop_cell = _make_cell("drop")
    futures = [_future_with_result(True), _future_with_result(False)]
    monkeypatch.setattr(gcp, "ThreadPoolExecutor", _fake_executor_factory(futures))
    monkeypatch.setattr(gcp, "as_completed", lambda futures, timeout=None: futures.keys())

    kept = gcp._collect_cells([keep_cell, drop_cell])
    assert kept == [keep_cell]


def test_collect_cells_raises_on_future_exception(monkeypatch):
    """An exception from any future aborts collection."""
    cell = _make_cell("failing")
    futures = [_future_with_result(RuntimeError("boom"))]
    monkeypatch.setattr(gcp, "ThreadPoolExecutor", _fake_executor_factory(futures))
    monkeypatch.setattr(gcp, "as_completed", lambda futures, timeout=None: futures.keys())

    with pytest.raises(RuntimeError, match="boom"):
        gcp._collect_cells([cell])


def test_total_timeout_cancels_pending_futures_and_raises(monkeypatch):
    """A total timeout cancels pending futures and re-raises the error."""
    cell = _make_cell("pending")
    pending = Future()
    futures = [pending]
    monkeypatch.setattr(gcp, "ThreadPoolExecutor", _fake_executor_factory(futures))
    monkeypatch.setattr(gcp, "as_completed", _as_completed_timeout_immediately)

    with pytest.raises(TimeoutError):
        gcp._collect_cells([cell])
    assert pending.cancelled()
