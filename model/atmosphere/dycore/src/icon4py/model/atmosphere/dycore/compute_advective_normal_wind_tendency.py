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
from gt4py.next.ffront.fbuiltins import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import (
    E2C,
    E2EC,
    E2V,
    E2CDim,
    E2VDim,
    Koff,
)
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_advective_normal_wind_tendency(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.ECDim], vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    zeta: fa.VertexKField[vpfloat],
    vt: fa.EdgeKField[vpfloat],
    f_e: fa.EdgeField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, E2CDim], wpfloat],
    z_w_con_c_full: fa.CellKField[vpfloat],
    vn_ie: fa.EdgeKField[vpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
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
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.ECDim], vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    zeta: fa.VertexKField[vpfloat],
    vt: fa.EdgeKField[vpfloat],
    f_e: fa.EdgeField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, E2CDim], wpfloat],
    z_w_con_c_full: fa.CellKField[vpfloat],
    vn_ie: fa.EdgeKField[vpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    ddt_vn_apc: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
