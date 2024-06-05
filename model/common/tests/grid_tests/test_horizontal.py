# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from icon4py.model.common import constants
from icon4py.model.common.grid.horizontal import CellParams
from icon4py.model.common.grid.icon import GlobalGridParams


@pytest.mark.parametrize(
    "grid_root,grid_level,expected",
    [
        (2, 4, 24907282236.708576),
        (4, 9, 6080879.45232143),
    ],
)
def test_mean_cell_area_calculation(grid_root, grid_level, expected):
    params = GlobalGridParams(grid_root, grid_level)
    assert expected == CellParams._compute_mean_cell_area(constants.EARTH_RADIUS, params.num_cells)
