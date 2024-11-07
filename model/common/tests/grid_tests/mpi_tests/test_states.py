# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import icon4py.model.common.constants as constants
from icon4py.model.common.grid import icon, states as grid_states


@pytest.mark.parametrize(
    "grid_root,grid_level,expected",
    [
        (2, 4, 24907282236.708576),
        (4, 9, 6080879.45232143),
    ],
)
def test_mean_cell_area_calculation(grid_root, grid_level, expected):
    params = icon.GlobalGridParams(grid_root, grid_level)
    assert expected == grid_states.compute_mean_cell_area_for_sphere(
        constants.EARTH_RADIUS, params.num_cells
    )
