# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import random

import pytest

from icon4py.model.testing.datatest_fixtures import (
    decomposition_info,
    experiment,
)
from icon4py.model.testing.helpers import backend, grid
from icon4py.model.testing.pytest_config import *  # noqa: F401

__all__ = [
    # local:
    "random_name",
    "test_path",
    # imported fixtures:
    "backend",
    "grid",
    "decomposition_info",
    "experiment"
]


@pytest.fixture
def random_name() -> str:
    return "test" + str(random.randint(0, 100000))


@pytest.fixture
def test_path(tmp_path):
    base_path = tmp_path.joinpath("io_tests")
    base_path.mkdir(exist_ok=True, parents=True, mode=0o777)
    yield base_path
    _delete_recursive(base_path)


def _delete_recursive(p: pathlib.Path) -> None:
    for child in p.iterdir():
        if child.is_file():
            child.unlink()
        else:
            _delete_recursive(child)
    p.rmdir()
