# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging

import numpy as np
import pytest
from gt4py import next as gtx
from model.common.tests.grid_tests.test_grid_manager import _run_grid_manager
from model.common.tests.grid_tests.test_icon import from_file

import icon4py.model
import icon4py.model.common.grid.base
from icon4py.model.common.grid import grid_manager as gm, simple
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils


@pytest.mark.parametrize("grid_file", (dt_utils.REGIONAL_EXPERIMENT, dt_utils.R02B04_GLOBAL))
def test_has_skip_values(grid_file):
    grid = from_file(grid_file)
    assert grid.has_skip_values()


@pytest.mark.parametrize("grid_file", (dt_utils.REGIONAL_EXPERIMENT, dt_utils.R02B04_GLOBAL))
def test_replace_skip_values(grid_file, caplog, backend):
    caplog.set_level(logging.DEBUG)
    xp = data_alloc.import_array_ns(backend)
    clear_skip_values = functools.partial(
        icon4py.model.common.grid.base.replace_skip_values, array_ns=xp
    )

    grid = _run_grid_manager(grid_file, backend=None).grid
    horizontal_connectivities = (
        c for c in grid.offset_providers.values() if isinstance(c, gtx.Connectivity)
    )
    for connectivity in horizontal_connectivities:
        clear_skip_values(connectivity)
        assert not np.any(connectivity.asnumpy() == gm.GridFile.INVALID_INDEX).item()


def test_replace_skip_values_validate_disconnected_grids():
    grid = simple.SimpleGrid()
    connectivity = grid.get_offset_provider("V2E")
    connectivity.ndarray[2, :] = gm.GridFile.INVALID_INDEX

    with pytest.raises(AssertionError) as error:
        icon4py.model.common.grid.base.replace_skip_values(connectivity)
        assert "disconnected" in error.value
