# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import constants, dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


dycore_consts: Final = constants.PhysicsConstants()


@gtx.field_operator
def _add_temporal_tendencies_to_vn(
    vn_nnow: fa.EdgeKField[wpfloat],
    ddt_vn_apc_ntl1: fa.EdgeKField[vpfloat],
    ddt_vn_phy: fa.EdgeKField[vpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    dtime: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_24."""
    z_gradh_exner_wp = astype(z_gradh_exner, wpfloat)

    vn_nnew_wp = vn_nnow + dtime * (
        astype(ddt_vn_apc_ntl1, wpfloat)
        - dycore_consts.cpd * z_theta_v_e * z_gradh_exner_wp
        + astype(ddt_vn_phy, wpfloat)
    )
    return vn_nnew_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def add_temporal_tendencies_to_vn(
    vn_nnow: fa.EdgeKField[wpfloat],
    ddt_vn_apc_ntl1: fa.EdgeKField[vpfloat],
    ddt_vn_phy: fa.EdgeKField[vpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    vn_nnew: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _add_temporal_tendencies_to_vn(
        vn_nnow,
        ddt_vn_apc_ntl1,
        ddt_vn_phy,
        z_theta_v_e,
        z_gradh_exner,
        dtime,
        out=vn_nnew,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
