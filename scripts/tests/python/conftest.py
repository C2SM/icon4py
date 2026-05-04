# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for dev-scripts tests."""

from __future__ import annotations

import pathlib
import sys

import pytest


_scripts_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_scripts_dir / "python"))


@pytest.fixture()
def scripts_dir() -> pathlib.Path:
    """Return the ``scripts/`` directory."""
    return _scripts_dir


@pytest.fixture()
def repo_root(scripts_dir: pathlib.Path) -> pathlib.Path:
    """Return the repository root."""
    return scripts_dir.parent
