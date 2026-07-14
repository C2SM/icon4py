# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_nabla4(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[vpfloat]:
    nabv_tang_vp = astype(wpfloat("0.0"), vpfloat)  # BISECT exp 6: zero E2C2V
    nabv_norm_vp = astype(wpfloat("0.0"), vpfloat)  # BISECT exp 6: zero E2C2V
    nabv_tang_wp, nabv_norm_wp = astype((nabv_tang_vp, nabv_norm_vp), wpfloat)
    z_nabla4_e2_wp = wpfloat("4.0") * (
        (nabv_norm_wp - wpfloat("2.0") * z_nabla2_e) * (inv_vert_vert_length * inv_vert_vert_length)
        + (nabv_tang_wp - wpfloat("2.0") * z_nabla2_e)
        * (inv_primal_edge_length * inv_primal_edge_length)
    )
    return astype(z_nabla4_e2_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_nabla4(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    z_nabla4_e2: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _calculate_nabla4(
        u_vert=u_vert,
        v_vert=v_vert,
        primal_normal_vert_v1=primal_normal_vert_v1,
        primal_normal_vert_v2=primal_normal_vert_v2,
        z_nabla2_e=z_nabla2_e,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_primal_edge_length=inv_primal_edge_length,
        out=z_nabla4_e2,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
