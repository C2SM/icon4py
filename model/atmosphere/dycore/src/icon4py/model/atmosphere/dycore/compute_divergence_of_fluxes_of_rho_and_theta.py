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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_divergence_of_fluxes_of_rho_and_theta(
    geofac_div: Field[[CEDim], wpfloat],
    mass_fl_e: Field[[EdgeDim, KDim], wpfloat],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], wpfloat],
) -> tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], vpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_41."""
    z_flxdiv_mass_wp = neighbor_sum(geofac_div(C2CE) * mass_fl_e(C2E), axis=C2EDim)
    z_flxdiv_theta_wp = neighbor_sum(geofac_div(C2CE) * z_theta_v_fl_e(C2E), axis=C2EDim)
    return astype((z_flxdiv_mass_wp, z_flxdiv_theta_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_divergence_of_fluxes_of_rho_and_theta(
    geofac_div: Field[[CEDim], wpfloat],
    mass_fl_e: Field[[EdgeDim, KDim], wpfloat],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], wpfloat],
    z_flxdiv_mass: Field[[CellDim, KDim], vpfloat],
    z_flxdiv_theta: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_divergence_of_fluxes_of_rho_and_theta(
        geofac_div,
        mass_fl_e,
        z_theta_v_fl_e,
        out=(z_flxdiv_mass, z_flxdiv_theta),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
