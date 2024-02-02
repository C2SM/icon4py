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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import constants
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.metrics.metric_fields import compute_z_mc
from icon4py.model.common.metrics.reference_atmosphere import compute_reference_atmosphere
from icon4py.model.common.test_utils.helpers import dallclose, zero_field
from icon4py.model.common.type_alias import wpfloat


@pytest.mark.datatest
def test_compute_reference_atmsophere_fields(grid_savepoint, metrics_savepoint):
    grid: IconGrid = grid_savepoint.construct_icon_grid()
    exner_ref_mc_ref = metrics_savepoint.exner_ref_mc()
    rho_ref_mc_ref = metrics_savepoint.rho_ref_mc()
    theta_ref_mc_ref = metrics_savepoint.theta_ref_mc()
    z_ifc = metrics_savepoint.z_ifc()

    exner_ref_mc = zero_field(grid, CellDim, KDim, dtype=wpfloat)
    rho_ref_mc = zero_field(grid, CellDim, KDim, dtype=wpfloat)
    theta_ref_mc = zero_field(grid, CellDim, KDim, dtype=wpfloat)
    z_mc = zero_field(grid, CellDim, KDim, dtype=wpfloat)
    start = int32(0)
    horizontal_end = grid.num_cells
    vertical_end = grid.num_levels
    compute_z_mc(
        z_ifc=z_ifc,
        z_mc=z_mc,
        horizontal_start=start,
        horizontal_end=horizontal_end,
        vertical_start=start,
        vertical_end=vertical_end,
        offset_provider={"Koff": grid.get_offset_provider("Koff")},
    )

    compute_reference_atmosphere(
        z_height=z_mc,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEAL_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants._H_SCAL_BG,
        del_t_bg=constants.DELTA_TEMPERATURE,
        exner_ref_mc=exner_ref_mc,
        rho_ref_mc=rho_ref_mc,
        theta_ref_mc=theta_ref_mc,
        horizontal_start=start,
        horizontal_end=horizontal_end,
        vertical_start=start,
        vertical_end=vertical_end,
        offset_provider={},
    )

    assert dallclose(rho_ref_mc.asnumpy(), rho_ref_mc_ref.asnumpy())
    assert dallclose(theta_ref_mc.asnumpy(), theta_ref_mc_ref.asnumpy())
    assert dallclose(exner_ref_mc.asnumpy(), exner_ref_mc_ref.asnumpy())


def test_compute_reference_atmsophere_on_half_level_mass_points(grid_savepoint, metrics_savepoint):
    grid: IconGrid = grid_savepoint.construct_icon_grid()
    theta_ref_ic_ref = metrics_savepoint.theta_ref_ic()
    z_ifc = metrics_savepoint.z_ifc()

    exner_ref_ic = zero_field(grid, CellDim, KDim, extend={KDim: 1}, dtype=wpfloat)
    rho_ref_ic = zero_field(grid, CellDim, KDim, extend={KDim: 1}, dtype=wpfloat)
    theta_ref_ic = zero_field(grid, CellDim, KDim, extend={KDim: 1}, dtype=wpfloat)
    start = int32(0)
    horizontal_end = grid.num_cells
    vertical_end = grid.num_levels + 1

    compute_reference_atmosphere(
        z_height=z_ifc,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEAL_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants._H_SCAL_BG,
        del_t_bg=constants.DELTA_TEMPERATURE,
        exner_ref_mc=exner_ref_ic,
        rho_ref_mc=rho_ref_ic,
        theta_ref_mc=theta_ref_ic,
        horizontal_start=start,
        horizontal_end=horizontal_end,
        vertical_start=start,
        vertical_end=vertical_end,
        offset_provider={},
    )

    assert dallclose(theta_ref_ic.asnumpy(), theta_ref_ic_ref.asnumpy())
