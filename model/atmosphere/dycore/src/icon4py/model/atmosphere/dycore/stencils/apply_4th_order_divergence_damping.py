# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, broadcast

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_4th_order_divergence_damping(
    scal_divdamp: fa.KField[wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """Formelry known as _mo_solve_nonhydro_4th_order_divdamp."""
    z_graddiv2_vn_wp = astype(z_graddiv2_vn, wpfloat)
    scal_divdamp = broadcast(scal_divdamp, (EdgeDim, KDim))
    vn_wp = vn + (scal_divdamp * z_graddiv2_vn_wp)
    return vn_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_4th_order_divergence_damping(
    scal_divdamp: fa.KField[wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_4th_order_divergence_damping(
        scal_divdamp,
        z_graddiv2_vn,
        vn,
        out=vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
