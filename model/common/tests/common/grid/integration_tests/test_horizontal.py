# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from gt4py import next as gtx

from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.testing import datatest_utils as dt_utils
from .. import utils
from ...fixtures import *  # noqa: F401, F403


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_map_domain_bounds_start_index(experiment, dim, grid_savepoint):
    grid_savepoint.start_index(dim)
    start_index_array = grid_savepoint.start_index(dim)
    _map_and_assert_array(dim, start_index_array)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_map_domain_bounds_end_index(experiment, dim, grid_savepoint):
    end_index_array = grid_savepoint.end_index(dim)
    _map_and_assert_array(dim, end_index_array)


def _map_and_assert_array(dim, index_array):
    index_map = h_grid.map_icon_domain_bounds(dim, index_array)
    for d, index in index_map.items():
        icon_index = h_grid._map_to_icon_index(d.dim, d.zone)
        assert index == index_array[icon_index]
        assert isinstance(index, gtx.int32)
