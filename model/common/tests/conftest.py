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

from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
    decomposition_info,
    experiment,
)
from icon4py.model.common.test_utils.grid_utils import grid  # noqa: F401 # fixtures
from icon4py.model.common.test_utils.helpers import backend  # noqa: F401 # fixtures


@pytest.fixture
def random_name():
    return "test" + str(random.randint(0, 100000))


def delete_recursive(p: pathlib.Path):
    for child in p.iterdir():
        if child.is_file():
            child.unlink()
        else:
            delete_recursive(child)
    p.rmdir()


@pytest.fixture
def test_path(tmp_path):
    base_path = tmp_path.joinpath("io_tests")
    base_path.mkdir(exist_ok=True, parents=True, mode=0o777)
    yield base_path
    delete_recursive(base_path)
