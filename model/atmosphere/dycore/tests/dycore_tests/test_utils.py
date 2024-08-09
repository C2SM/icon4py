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

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.dycore.state_utils import utils as solve_nh_utils
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import simple as simple_grid
from icon4py.model.common.settings import backend
from icon4py.model.common.test_utils import helpers


def scal_divdamp_for_order_24_numpy(a: np.array, factor: float, mean_cell_area: float):
    a = np.maximum(0.0, a - 0.25 * factor)
    return -a * mean_cell_area**2


def bdy_divdamp_numpy(coeff: float, field: np.array):
    return 0.75 / (coeff + constants.DBL_EPS) * np.abs(field)


def test_caclulate_scal_divdamp_order_24():
    divdamp_fac_o2 = 3.0
    divdamp_order = 24
    mean_cell_area = 1000.0
    grid = simple_grid.SimpleGrid()
    enh_divdamp_fac = helpers.random_field(grid, dims.KDim)
    out = helpers.random_field(grid, dims.KDim)

    solve_nh_utils._calculate_scal_divdamp.with_backend(backend)(
        enh_divdamp_fac=enh_divdamp_fac,
        divdamp_fac_o2=divdamp_fac_o2,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        out=out,
        offset_provider={},
    )

    ref = scal_divdamp_for_order_24_numpy(enh_divdamp_fac.asnumpy(), divdamp_fac_o2, mean_cell_area)
    assert helpers.dallclose(ref, out.asnumpy())


def test_calculate_scal_divdamp_any_order():
    divdamp_fac_o2 = 4.2
    divdamp_order = 3
    mean_cell_area = 1000.0
    grid = simple_grid.SimpleGrid()
    enh_divdamp_fac = helpers.random_field(grid, dims.KDim)
    out = helpers.random_field(grid, dims.KDim)

    solve_nh_utils._calculate_scal_divdamp.with_backend(backend)(
        enh_divdamp_fac=enh_divdamp_fac,
        divdamp_fac_o2=divdamp_fac_o2,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        out=out,
        offset_provider={},
    )
    enhanced_factor = -enh_divdamp_fac.asnumpy() * mean_cell_area**2
    assert helpers.dallclose(enhanced_factor, out.asnumpy())


def test_calculate_bdy_divdamp():
    grid = simple_grid.SimpleGrid()
    scal_divdamp = helpers.random_field(grid, dims.KDim)
    out = helpers.zero_field(grid, dims.KDim)
    coeff = 0.3
    solve_nh_utils._calculate_bdy_divdamp.with_backend(backend)(
        scal_divdamp, coeff, constants.DBL_EPS, out=out, offset_provider={}
    )
    assert helpers.dallclose(out.asnumpy(), bdy_divdamp_numpy(coeff, scal_divdamp.asnumpy()))


def test_calculate_divdamp_fields():
    grid = simple_grid.SimpleGrid()
    divdamp_field = helpers.random_field(grid, dims.KDim)
    scal_divdamp = helpers.zero_field(grid, dims.KDim)
    boundary_divdamp = helpers.zero_field(grid, dims.KDim)
    divdamp_order = gtx.int32(24)
    mean_cell_area = 1000.0
    divdamp_fac_o2 = 0.7
    nudge_max_coeff = 0.3

    scaled_ref = scal_divdamp_for_order_24_numpy(
        np.asarray(divdamp_field), divdamp_fac_o2, mean_cell_area
    )

    boundary_ref = bdy_divdamp_numpy(nudge_max_coeff, scaled_ref)

    solve_nh_utils._calculate_divdamp_fields.with_backend(backend)(
        divdamp_field,
        divdamp_order,
        mean_cell_area,
        divdamp_fac_o2,
        nudge_max_coeff,
        constants.DBL_EPS,
        out=(scal_divdamp, boundary_divdamp),
        offset_provider={},
    )
    helpers.dallclose(scal_divdamp.asnumpy(), scaled_ref)
    helpers.dallclose(boundary_divdamp.asnumpy(), boundary_ref)
