# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for py._util shared helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from scripts.py._common import repo_root, run_or_fail


def test_repo_root_is_parent_of_scripts():
    root = repo_root()
    assert (root / "scripts").is_dir()


def test_run_or_fail_success():
    result = run_or_fail(["echo", "hello"])
    assert result.returncode == 0


def test_run_or_fail_propagates_failure():
    with pytest.raises(SystemExit) as exc_info:
        run_or_fail(["false"])
    assert exc_info.value.code != 0
