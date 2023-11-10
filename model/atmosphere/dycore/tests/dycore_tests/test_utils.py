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

import numpy as np

from icon4py.model.atmosphere.dycore.state_utils.utils import (
    _calculate_bdy_divdamp,
    _scal_divdamp_NEW,
)
from icon4py.model.common import constants
from icon4py.model.common.dimension import KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field


def scal_divdamp_numpy_for_order_24(a: np.array, factor: float, mean_cell_area: float):
    a = np.maximum(0.0, a - 0.25 * factor)
    return -a * mean_cell_area**2


def test_scal_divdamp_order_24():
    divdamp_fac_o2 = 3.0
    divdamp_order = 24
    mean_cell_area = 1000.0
    grid = SimpleGrid()
    enh_divdamp_fac = random_field(grid, KDim)
    out = random_field(grid, KDim)

    _scal_divdamp_NEW(
        enh_divdamp_fac=enh_divdamp_fac,
        divdamp_fac_o2=divdamp_fac_o2,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        out=out,
        offset_provider={},
    )

    ref = scal_divdamp_numpy_for_order_24(enh_divdamp_fac, divdamp_fac_o2, mean_cell_area)
    assert np.allclose(np.asarray(ref), np.asarray(out))


def test_scal_divdamp_order_invalid():
    divdamp_fac_o2 = 4.2
    divdamp_order = 3
    mean_cell_area = 1000.0
    grid = SimpleGrid()
    enh_divdamp_fac = random_field(grid, KDim)
    out = random_field(grid, KDim)

    _scal_divdamp_NEW(
        enh_divdamp_fac=enh_divdamp_fac,
        divdamp_fac_o2=divdamp_fac_o2,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        out=out,
        offset_provider={},
    )
    assert np.allclose(np.asarray(-enh_divdamp_fac * mean_cell_area**2), np.asarray(out))


def test_bdy_divdamp():
    grid = SimpleGrid()
    scal_divdamp = random_field(grid, KDim)
    out = zero_field(grid, KDim)
    coeff = 0.3
    _calculate_bdy_divdamp(scal_divdamp, coeff, constants.dbl_eps, out=out, offset_provider={})

    def bdy_divdamp_numpy(coeff: float, field: np.array):
        return 0.75 / (coeff + constants.dbl_eps) * np.abs(field)

    assert np.allclose(out, bdy_divdamp_numpy(coeff, np.asarray(scal_divdamp)))
