# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the serialization run driver."""

from __future__ import annotations

import pytest
from run_serialization import run_command


def test_run_command_reports_what_the_command_printed():
    # 'CalledProcessError' names the exit status but not the captured output, which is
    # where sbatch says things like "the view 'debug' does not exist".
    with pytest.raises(RuntimeError, match="the view 'debug' does not exist"):
        run_command(
            ["sh", "-c", "echo \"sbatch: error: the view 'debug' does not exist\" >&2; exit 1"]
        )


def test_run_command_reports_a_silent_failure():
    with pytest.raises(RuntimeError, match="printed nothing"):
        run_command(["sh", "-c", "exit 3"])


def test_run_command_returns_the_result_when_not_checking():
    result = run_command(["sh", "-c", "exit 3"], check=False)

    assert result.returncode == 3
