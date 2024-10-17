# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_mass_flux(
    z_rho_e: fa.EdgeKField[wpfloat],
    z_vn_avg: fa.EdgeKField[wpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_32."""
    mass_fl_e_wp = z_rho_e * z_vn_avg * astype(ddqz_z_full_e, wpfloat)
    z_theta_v_fl_e_wp = mass_fl_e_wp * z_theta_v_e
    return mass_fl_e_wp, z_theta_v_fl_e_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_mass_flux(
    z_rho_e: fa.EdgeKField[wpfloat],
    z_vn_avg: fa.EdgeKField[wpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    mass_fl_e: fa.EdgeKField[wpfloat],
    z_theta_v_fl_e: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_mass_flux(
        z_rho_e,
        z_vn_avg,
        ddqz_z_full_e,
        z_theta_v_e,
        out=(mass_fl_e, z_theta_v_fl_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
