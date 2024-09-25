# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, C2EDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_divergence_of_fluxes_of_rho_and_theta(
    geofac_div: Field[[dims.CEDim], wpfloat],
    mass_fl_e: fa.EdgeKField[wpfloat],
    z_theta_v_fl_e: fa.EdgeKField[wpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_41."""
    z_flxdiv_mass_wp = neighbor_sum(geofac_div(C2CE) * mass_fl_e(C2E), axis=C2EDim)
    z_flxdiv_theta_wp = neighbor_sum(geofac_div(C2CE) * z_theta_v_fl_e(C2E), axis=C2EDim)
    return astype((z_flxdiv_mass_wp, z_flxdiv_theta_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_divergence_of_fluxes_of_rho_and_theta(
    geofac_div: Field[[dims.CEDim], wpfloat],
    mass_fl_e: fa.EdgeKField[wpfloat],
    z_theta_v_fl_e: fa.EdgeKField[wpfloat],
    z_flxdiv_mass: fa.CellKField[vpfloat],
    z_flxdiv_theta: fa.CellKField[vpfloat],
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
