# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
import random
import shutil
from collections.abc import Generator

import pytest

from icon4py.model.testing.datatest_fixtures import (
    decomposition_info,
)
from icon4py.model.testing.helpers import connectivities_as_numpy


# ruff: noqa: F405
# Make sure custom icon4py pytest hooks are loaded
try:
    import sys

    _ = sys.modules["icon4py.model.testing.pytest_config"]
except KeyError:
    from icon4py.model.testing.pytest_config import *  # noqa: F403

__all__ = [
    # local:
    "random_name",
    "tmp_io_tests_path",
    # imported fixtures:
    "backend",
    "grid",
    "decomposition_info",
    "experiment__DELETE",
    "connectivities_as_numpy",
]


@pytest.fixture
def random_name() -> str:
    return "test" + str(random.randint(0, 100000))


@pytest.fixture
def tmp_io_tests_path(tmp_path) -> Generator[pathlib.Path]:
    base_path = (tmp_path / "io_tests").resolve()
    base_path.mkdir(exist_ok=True, parents=True, mode=0o777)

    yield base_path

    shutil.rmtree(base_path)
