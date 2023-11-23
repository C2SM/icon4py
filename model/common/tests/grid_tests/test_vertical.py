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

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common.dimension import KDim
from icon4py.model.common.grid.vertical import VerticalModelParams


@pytest.mark.parametrize(
    "max_h,damping,delta",
    [(60000, 34000, 612), (12000, 10000, 100), (109050, 45000, 123)],
)
def test_nrdmax_calculation(max_h, damping, delta, grid_savepoint):
    vct_a = np.arange(0, max_h, delta)
    vct_a_field = gtx.as_field((KDim,), data=vct_a[::-1])
    vertical_params = VerticalModelParams(
        rayleigh_damping_height=damping,
        vct_a=vct_a_field,
        nflat_gradp=grid_savepoint.nflat_gradp,
        nflatlev=grid_savepoint.nflatlev(),
    )
    assert vertical_params.index_of_damping_layer == vct_a.shape[0] - math.ceil(damping / delta) - 1


@pytest.mark.datatest
def test_nrdmax_calculation_from_icon_input(grid_savepoint, damping_height):
    a = grid_savepoint.vct_a()
    nrdmax = grid_savepoint.nrdmax()
    vertical_params = VerticalModelParams(
        rayleigh_damping_height=damping_height,
        vct_a=a,
        nflat_gradp=grid_savepoint.nflat_gradp,
        nflatlev=grid_savepoint.nflatlev(),
    )
    assert nrdmax == vertical_params.index_of_damping_layer
    a_array = a.asnumpy()
    assert a_array[nrdmax] > damping_height
    assert a_array[nrdmax + 1] < damping_height


@pytest.mark.datatest
def test_grid_size(grid_savepoint):
    assert 65 == grid_savepoint.num(KDim)
