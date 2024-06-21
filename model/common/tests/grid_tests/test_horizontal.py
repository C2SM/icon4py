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

from icon4py.model.common.grid import icon


@pytest.mark.parametrize(
    "grid_root,grid_level,expected",
    [
        (2, 4, 24907282236.708576),
        (4, 9, 6080879.45232143),
    ],
)
def test_mean_cell_area_calculation_on_sphere(grid_root, grid_level, expected):
    icosahedron = icon.Icosahedron(grid_root, grid_level)
    assert expected == icosahedron.mean_cell_area


@pytest.mark.parametrize(
    "edge_length, expected_area",
    [
        (33333.3333333333, 481125224.324688),
        (291.545189504373, 36805.4723705445),
        (3125.0, 4228639.6669162),
    ],
)
def test_mean_cell_area_calculation_on_torus(edge_length, expected_area):
    torus = icon.Torus(edge_length, 1000)
    assert expected_area == pytest.approx(torus.mean_cell_area, 1e-12)
