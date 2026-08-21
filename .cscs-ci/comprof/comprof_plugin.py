# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest plugin logging test boundaries for the compile profiler.

Streams new JSONL log lines to stdout after each test (prefix ``COMPROF|``) so
data survives a SLURM timeout, and reprints the complete log between markers at
session end (controller process only, after draining outstanding compilations).
"""

import os
import pathlib

from compcore import log_event


_ACTIVE = os.environ.get("ICON4PY_COMPROF") == "1"
_LOG_PATH = pathlib.Path(os.environ.get("ICON4PY_COMPROF_LOG", "/tmp/comprof.jsonl"))
_state = {"pos": 0}


def pytest_sessionstart(session):
    log_event(event="session_start")


def pytest_runtest_logstart(nodeid, location):
    log_event(event="test_start", nodeid=nodeid)


def pytest_runtest_logfinish(nodeid, location):
    log_event(event="test_end", nodeid=nodeid)
    _flush_incremental()


def _flush_incremental():
    """Print newly appended complete JSONL lines with the COMPROF| prefix."""
    if not _ACTIVE or os.environ.get("PYTEST_XDIST_WORKER"):
        return
    try:
        with _LOG_PATH.open(encoding="utf-8") as f:
            f.seek(_state["pos"])
            data = f.read()
    except OSError:
        return
    last_nl = data.rfind("\n")
    if last_nl < 0:
        return
    complete = data[: last_nl + 1]
    _state["pos"] += last_nl + 1
    for line in complete.splitlines():
        print(f"COMPROF|{line}")


def pytest_sessionfinish(session, exitstatus):
    log_event(event="session_end", exitstatus=str(exitstatus))
    if not _ACTIVE:
        return
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    try:
        import gt4py.next as gtx  # noqa: PLC0415 [import-outside-top-level]

        gtx.wait_for_compilation()
    except Exception as exc:
        print(f"[COMPROF] wait_for_compilation failed: {exc!r}")
    log_event(event="drained")
    _flush_incremental()
    try:
        payload = _LOG_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        payload = f"<failed to read {_LOG_PATH}: {exc}>"
    print("\n=== COMPROF_JSONL_BEGIN ===")
    print(payload)
    print("=== COMPROF_JSONL_END ===")
