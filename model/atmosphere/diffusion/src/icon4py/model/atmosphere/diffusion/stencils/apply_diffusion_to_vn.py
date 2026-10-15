# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, broadcast, maximum
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2V
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
    u_vert_wp, v_vert_wp = astype((u_vert, v_vert), wpfloat)

    nabv_tang_vp = astype(
        (
            u_vert_wp(E2C2V[0]) * primal_normal_vert_v1[dims.E2C2VDim(0)]
            + v_vert_wp(E2C2V[0]) * primal_normal_vert_v2[dims.E2C2VDim(0)]
            + u_vert_wp(E2C2V[1]) * primal_normal_vert_v1[dims.E2C2VDim(1)]
            + v_vert_wp(E2C2V[1]) * primal_normal_vert_v2[dims.E2C2VDim(1)]
        ),
        vpfloat,
    )

    nabv_norm_vp = astype(
        (
            u_vert_wp(E2C2V[2]) * primal_normal_vert_v1[dims.E2C2VDim(2)]
            + v_vert_wp(E2C2V[2]) * primal_normal_vert_v2[dims.E2C2VDim(2)]
            + u_vert_wp(E2C2V[3]) * primal_normal_vert_v1[dims.E2C2VDim(3)]
            + v_vert_wp(E2C2V[3]) * primal_normal_vert_v2[dims.E2C2VDim(3)]
        ),
        vpfloat,
    )
    nabv_tang_wp, nabv_norm_wp = astype((nabv_tang_vp, nabv_norm_vp), wpfloat)
    z_nabla4_e2_wp = wpfloat("4.0") * (
        (nabv_norm_wp - wpfloat("2.0") * z_nabla2_e) * (inv_vert_vert_length * inv_vert_vert_length)
        + (nabv_tang_wp - wpfloat("2.0") * z_nabla2_e)
        * (inv_primal_edge_length * inv_primal_edge_length)
    )
    return astype(z_nabla4_e2_wp, vpfloat)


@gtx.field_operator
def _apply_nabla2_and_nabla4_to_vn(
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    z_nabla4_e2: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    nudgezone_diff: vpfloat,
) -> fa.EdgeKField[wpfloat]:
    kh_smag_e_wp, z_nabla4_e2_wp, nudgezone_diff_wp = astype(
        (kh_smag_e, z_nabla4_e2, nudgezone_diff), wpfloat
    )
    area_edge_broadcast = broadcast(area_edge, (dims.EdgeDim, dims.KDim))

    vn_wp = vn + area_edge * (
        maximum(nudgezone_diff_wp * nudgecoeff_e, kh_smag_e_wp) * z_nabla2_e
        - area_edge_broadcast * diff_multfac_vn * z_nabla4_e2_wp
    )
    return vn_wp


@gtx.field_operator
def _apply_nabla2_and_nabla4_global_to_vn(
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    z_nabla4_e2: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    kh_smag_e_wp, z_nabla4_e2_wp = astype((kh_smag_e, z_nabla4_e2), wpfloat)

    vn_wp = vn + area_edge * (
        kh_smag_e_wp * z_nabla2_e - diff_multfac_vn * z_nabla4_e2_wp * area_edge
    )
    return vn_wp


@gtx.field_operator
def _apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: fa.EdgeKField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    fac_bdydiff_v: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    vn_wp = vn + (area_edge * fac_bdydiff_v * z_nabla2_e)
    return vn_wp


@gtx.field_operator
def _apply_diffusion_to_vn(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
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

    # TODO(): Use if-else statement instead
    vn = (
        concat_where(
            dims.EdgeDim >= start_2nd_nudge_line_idx_e,
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
        else concat_where(
            dims.EdgeDim >= start_2nd_nudge_line_idx_e,
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
