# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_global_to_vn import (
    _apply_nabla2_and_nabla4_global_to_vn,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_to_vn import (
    _apply_nabla2_and_nabla4_to_vn,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_vn_in_lateral_boundary import (
    _apply_nabla2_to_vn_in_lateral_boundary,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla4 import _calculate_nabla4
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_diffusion_to_vn(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.ECVDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.ECVDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    edge: fa.EdgeField[gtx.int32],
    nudgezone_diff: vpfloat,
    fac_bdydiff_v: wpfloat,
    start_2nd_nudge_line_idx_e: gtx.int32,
    limited_area: bool,
) -> fa.EdgeKField[wpfloat]:
    z_nabla4_e2 = _calculate_nabla4(
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
    )

    # TODO: Use if-else statement instead
    vn = (
        where(
            start_2nd_nudge_line_idx_e <= edge,
            _apply_nabla2_and_nabla4_to_vn(
                area_edge,
                kh_smag_e,
                z_nabla2_e,
                z_nabla4_e2,
                diff_multfac_vn,
                nudgecoeff_e,
                vn,
                nudgezone_diff,
            ),
            _apply_nabla2_to_vn_in_lateral_boundary(z_nabla2_e, area_edge, vn, fac_bdydiff_v),
        )
        if limited_area
        else where(
            start_2nd_nudge_line_idx_e <= edge,
            _apply_nabla2_and_nabla4_global_to_vn(
                area_edge,
                kh_smag_e,
                z_nabla2_e,
                z_nabla4_e2,
                diff_multfac_vn,
                vn,
            ),
            vn,
        )
    )

    return vn


@program(grid_type=GridType.UNSTRUCTURED)
def apply_diffusion_to_vn(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.ECVDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.ECVDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    edge: fa.EdgeField[gtx.int32],
    nudgezone_diff: vpfloat,
    fac_bdydiff_v: wpfloat,
    start_2nd_nudge_line_idx_e: gtx.int32,
    limited_area: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_diffusion_to_vn(
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        area_edge,
        kh_smag_e,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        edge,
        nudgezone_diff,
        fac_bdydiff_v,
        start_2nd_nudge_line_idx_e,
        limited_area,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
