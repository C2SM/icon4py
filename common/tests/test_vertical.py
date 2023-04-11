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

import math

import numpy as np
import pytest

from icon4py.grid.vertical import VerticalGridConfig, VerticalModelParams
from icon4py.testutils.fixtures import (  # noqa F401
    data_provider,
    grid_savepoint,
    setup_icon_data,
)


@pytest.mark.parametrize(
    "max_h,damping,delta",
    [(60000, 34000, 612), (12000, 10000, 100), (109050, 45000, 123)],
)
def test_nrdmax_calculation(max_h, damping, delta):
    vct_a = np.arange(0, max_h, delta)
    vct_a = vct_a[::-1]
    vertical_params = VerticalModelParams(
        config=VerticalGridConfig(num_lev=10),
        rayleigh_damping_height=damping,
        vct_a=vct_a,
    )
    assert (
        vertical_params.index_of_damping_layer
        == vct_a.shape[0] - math.ceil(damping / delta) - 1
    )


@pytest.mark.datatest
def test_nrdmax_calculation_from_icon_input(icon_grid, grid_savepoint):  # noqa: F811
    a = grid_savepoint.vct_a()
    damping_height = 12500
    vertical_params = VerticalModelParams(
        VerticalGridConfig(num_lev=10), rayleigh_damping_height=damping_height, vct_a=a
    )
    assert 9 == vertical_params.index_of_damping_layer
    a_array = np.asarray(a)
    assert a_array[9] > damping_height
    assert a_array[10] < damping_height
