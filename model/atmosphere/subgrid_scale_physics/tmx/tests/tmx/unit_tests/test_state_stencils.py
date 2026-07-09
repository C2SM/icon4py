# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import state_stencils
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import test_utils


def _run(program, grid, **fields):
    program.with_grid_type(gtx.GridType.UNSTRUCTURED)(
        **fields,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(grid.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(grid.num_levels),
        offset_provider={},
    )


def test_compute_air_mass_is_rho_dz():
    grid = simple.simple_grid()
    rho = data_alloc.constant_field(grid, 1.2, dims.CellDim, dims.KDim)
    dz = data_alloc.constant_field(grid, 250.0, dims.CellDim, dims.KDim)
    air_mass = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
    _run(state_stencils.compute_air_mass, grid, rho=rho, ddqz_z_full=dz, air_mass=air_mass)
    test_utils.assert_dallclose(air_mass.asnumpy(), 1.2 * 250.0)


def test_compute_cv_air_matches_fortran_formula():
    grid = simple.simple_grid()
    q = dict(qv=1e-3, qc=2e-4, qi=1e-4, qr=5e-5, qs=3e-5, qg=1e-5)
    fields = {
        k: data_alloc.constant_field(grid, val, dims.CellDim, dims.KDim) for k, val in q.items()
    }
    air_mass = data_alloc.constant_field(grid, 300.0, dims.CellDim, dims.KDim)
    cv_air = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
    _run(state_stencils.compute_cv_air, grid, **fields, air_mass=air_mass, cv_air=cv_air)

    qtot = sum(q.values())
    cv = (
        constants.CVD * (1.0 - qtot)
        + constants.CVV * q["qv"]
        + constants.CPL * (q["qc"] + q["qr"])
        + constants.SPECIFIC_HEAT_CAPACITY_ICE * (q["qi"] + q["qs"] + q["qg"])
    )
    test_utils.assert_dallclose(cv_air.asnumpy(), cv * 300.0)
