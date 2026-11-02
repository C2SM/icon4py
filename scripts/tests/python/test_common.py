# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# mypy: ignore-errors
# Repository-infrastructure tooling that predates the type-checking rollout
# and lives outside the type-checked packages.

"""Tests for py._util shared helpers."""

from __future__ import annotations

import pytest
import typer
from helpers.common import run_or_fail


def test_run_or_fail_success():
    result = run_or_fail(["echo", "hello"])
    assert result.returncode == 0


def test_run_or_fail_propagates_failure():
    with pytest.raises(typer.Exit) as exc_info:
        run_or_fail(["false"])
    assert exc_info.value.exit_code != 0
