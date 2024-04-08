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
from gt4py.next.ffront.fbuiltins import Field, astype, int32

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.model_backend import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_mass_flux(
    z_rho_e: Field[[EdgeDim, KDim], wpfloat],
    z_vn_avg: Field[[EdgeDim, KDim], wpfloat],
    ddqz_z_full_e: Field[[EdgeDim, KDim], vpfloat],
    z_theta_v_e: Field[[EdgeDim, KDim], wpfloat],
) -> tuple[Field[[EdgeDim, KDim], wpfloat], Field[[EdgeDim, KDim], wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_32."""
    mass_fl_e_wp = z_rho_e * z_vn_avg * astype(ddqz_z_full_e, wpfloat)
    z_theta_v_fl_e_wp = mass_fl_e_wp * z_theta_v_e
    return mass_fl_e_wp, z_theta_v_fl_e_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_mass_flux(
    z_rho_e: Field[[EdgeDim, KDim], wpfloat],
    z_vn_avg: Field[[EdgeDim, KDim], wpfloat],
    ddqz_z_full_e: Field[[EdgeDim, KDim], vpfloat],
    z_theta_v_e: Field[[EdgeDim, KDim], wpfloat],
    mass_fl_e: Field[[EdgeDim, KDim], wpfloat],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_mass_flux(
        z_rho_e,
        z_vn_avg,
        ddqz_z_full_e,
        z_theta_v_e,
        out=(mass_fl_e, z_theta_v_fl_e),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
