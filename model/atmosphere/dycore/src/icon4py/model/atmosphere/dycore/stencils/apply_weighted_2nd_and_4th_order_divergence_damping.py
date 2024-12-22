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
from gt4py.next.ffront.fbuiltins import astype, broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_weighted_2nd_and_4th_order_divergence_damping(
    scal_divdamp: fa.KField[wpfloat],
    bdy_divdamp: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """Formelry known as _mo_solve_nonhydro_stencil_27."""
    z_graddiv2_vn_wp = astype(z_graddiv2_vn, wpfloat)

    scal_divdamp = broadcast(scal_divdamp, (dims.EdgeDim, dims.KDim))
    bdy_divdamp = broadcast(bdy_divdamp, (dims.EdgeDim, dims.KDim))
    vn_wp = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * z_graddiv2_vn_wp
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED)
def apply_weighted_2nd_and_4th_order_divergence_damping(
    scal_divdamp: fa.KField[wpfloat],
    bdy_divdamp: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_weighted_2nd_and_4th_order_divergence_damping(
        scal_divdamp,
        bdy_divdamp,
        nudgecoeff_e,
        z_graddiv2_vn,
        vn,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
