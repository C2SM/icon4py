# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.dycore import dycore_utils
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import simple as simple_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers


def scal_divdamp_for_order_24_numpy(a: np.array, factor: float, mean_cell_area: float):
    a = np.maximum(0.0, a - 0.25 * factor)
    return -a * mean_cell_area**2


def bdy_divdamp_numpy(coeff: float, field: np.array):
    return 0.75 / (coeff + constants.DBL_EPS) * np.abs(field)


def test_calculate_scal_divdamp_order_24(backend):
    divdamp_fac_o2 = 3.0
    divdamp_order = 24
    mean_cell_area = 1000.0
    grid = simple_grid.SimpleGrid()
    enh_divdamp_fac = data_alloc.random_field(grid, dims.KDim, backend=backend)
    out = data_alloc.random_field(grid, dims.KDim, backend=backend)

    dycore_utils._calculate_scal_divdamp.with_backend(backend)(
        enh_divdamp_fac=enh_divdamp_fac,
        divdamp_fac_o2=divdamp_fac_o2,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        out=out,
        offset_provider={},
    )

    ref = scal_divdamp_for_order_24_numpy(enh_divdamp_fac.asnumpy(), divdamp_fac_o2, mean_cell_area)
    assert helpers.dallclose(ref, out.asnumpy())


def test_calculate_scal_divdamp_any_order(backend):
    divdamp_fac_o2 = 4.2
    divdamp_order = 3
    mean_cell_area = 1000.0
    grid = simple_grid.SimpleGrid()
    enh_divdamp_fac = data_alloc.random_field(grid, dims.KDim, backend=backend)
    out = data_alloc.random_field(grid, dims.KDim, backend=backend)

    dycore_utils._calculate_scal_divdamp.with_backend(backend)(
        enh_divdamp_fac=enh_divdamp_fac,
        divdamp_fac_o2=divdamp_fac_o2,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        out=out,
        offset_provider={},
    )
    enhanced_factor = -enh_divdamp_fac.asnumpy() * mean_cell_area**2
    assert helpers.dallclose(enhanced_factor, out.asnumpy())


def test_calculate_bdy_divdamp(backend):
    grid = simple_grid.SimpleGrid()
    scal_divdamp = data_alloc.random_field(grid, dims.KDim, backend=backend)
    out = data_alloc.zero_field(grid, dims.KDim, backend=backend)
    coeff = 0.3
    dycore_utils._calculate_bdy_divdamp.with_backend(backend)(
        scal_divdamp, coeff, constants.DBL_EPS, out=out, offset_provider={}
    )
    assert helpers.dallclose(out.asnumpy(), bdy_divdamp_numpy(coeff, scal_divdamp.asnumpy()))


def test_calculate_divdamp_fields(backend):
    grid = simple_grid.SimpleGrid()
    divdamp_field = data_alloc.random_field(grid, dims.KDim, backend=backend)
    scal_divdamp = data_alloc.zero_field(grid, dims.KDim, backend=backend)
    boundary_divdamp = data_alloc.zero_field(grid, dims.KDim, backend=backend)
    divdamp_order = gtx.int32(24)
    mean_cell_area = 1000.0
    divdamp_fac_o2 = 0.7
    nudge_max_coeff = 0.3

    scaled_ref = scal_divdamp_for_order_24_numpy(
        np.asarray(divdamp_field), divdamp_fac_o2, mean_cell_area
    )

    boundary_ref = bdy_divdamp_numpy(nudge_max_coeff, scaled_ref)

    dycore_utils._calculate_divdamp_fields.with_backend(backend)(
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
