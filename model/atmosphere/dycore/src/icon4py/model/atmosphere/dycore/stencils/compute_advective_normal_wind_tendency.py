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

from icon4py.model.common.dimension import (
    E2C,
    E2EC,
    E2V,
    CellDim,
    E2CDim,
    E2VDim,
    ECDim,
    EdgeDim,
    KDim,
    Koff,
    VertexDim,
)
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_advective_normal_wind_tendency(
    z_kin_hor_e: Field[[EdgeDim, KDim], vpfloat],
    coeff_gradekin: Field[[ECDim], vpfloat],
    z_ekinh: Field[[CellDim, KDim], vpfloat],
    zeta: Field[[VertexDim, KDim], vpfloat],
    vt: Field[[EdgeDim, KDim], vpfloat],
    f_e: Field[[EdgeDim], wpfloat],
    c_lin_e: Field[[EdgeDim, E2CDim], wpfloat],
    z_w_con_c_full: Field[[CellDim, KDim], vpfloat],
    vn_ie: Field[[EdgeDim, KDim], vpfloat],
    ddqz_z_full_e: Field[[EdgeDim, KDim], vpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_19."""
    vt_wp, z_w_con_c_full_wp, ddqz_z_full_e_wp = astype(
        (vt, z_w_con_c_full, ddqz_z_full_e), wpfloat
    )

    ddt_vn_apc_wp = -(
        astype(
            z_kin_hor_e * (coeff_gradekin(E2EC[0]) - coeff_gradekin(E2EC[1]))
            + coeff_gradekin(E2EC[1]) * z_ekinh(E2C[1])
            - coeff_gradekin(E2EC[0]) * z_ekinh(E2C[0]),
            wpfloat,
        )
        + vt_wp * (f_e + astype(vpfloat("0.5") * neighbor_sum(zeta(E2V), axis=E2VDim), wpfloat))
        + neighbor_sum(c_lin_e * z_w_con_c_full_wp(E2C), axis=E2CDim)
        * astype((vn_ie - vn_ie(Koff[1])), wpfloat)
        / ddqz_z_full_e_wp
    )

    return astype(ddt_vn_apc_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_advective_normal_wind_tendency(
    z_kin_hor_e: Field[[EdgeDim, KDim], vpfloat],
    coeff_gradekin: Field[[ECDim], vpfloat],
    z_ekinh: Field[[CellDim, KDim], vpfloat],
    zeta: Field[[VertexDim, KDim], vpfloat],
    vt: Field[[EdgeDim, KDim], vpfloat],
    f_e: Field[[EdgeDim], wpfloat],
    c_lin_e: Field[[EdgeDim, E2CDim], wpfloat],
    z_w_con_c_full: Field[[CellDim, KDim], vpfloat],
    vn_ie: Field[[EdgeDim, KDim], vpfloat],
    ddqz_z_full_e: Field[[EdgeDim, KDim], vpfloat],
    ddt_vn_apc: Field[[EdgeDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_advective_normal_wind_tendency(
        z_kin_hor_e,
        coeff_gradekin,
        z_ekinh,
        zeta,
        vt,
        f_e,
        c_lin_e,
        z_w_con_c_full,
        vn_ie,
        ddqz_z_full_e,
        out=ddt_vn_apc,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
