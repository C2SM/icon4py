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
from gt4py.next.ffront.fbuiltins import astype, maximum, minimum, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2V, E2C2VDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    smag_limit: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    smag_offset: vpfloat,
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[wpfloat],
]:
    diff_multfac_smag_wp, u_vert_wp, v_vert_wp, smag_offset_wp = astype(
        (diff_multfac_smag, u_vert, v_vert, smag_offset), wpfloat
    )

    tmp = u_vert_wp(E2C2V) * dual_normal_vert_x + v_vert_wp(E2C2V) * dual_normal_vert_y

    dvt_tang_wp = -tmp[E2C2VDim(0)] + tmp[E2C2VDim(1)]
    dvt_norm_vp = astype((-tmp[E2C2VDim(2)]) + tmp[E2C2VDim(3)], vpfloat)

    tmp2 = astype(
        u_vert_wp(E2C2V) * primal_normal_vert_x + v_vert_wp(E2C2V) * primal_normal_vert_y, vpfloat
    )
    kh_smag_1_vp = -tmp2[E2C2VDim(0)] + tmp2[E2C2VDim(1)]

    dvt_tang_vp = astype(dvt_tang_wp * tangent_orientation, vpfloat)

    kh_smag_1_wp, dvt_norm_wp = astype((kh_smag_1_vp, dvt_norm_vp), wpfloat)
    kh_smag_1_wp = (kh_smag_1_wp * tangent_orientation * inv_primal_edge_length) + (
        dvt_norm_wp * inv_vert_vert_length
    )

    kh_smag_1_wp = kh_smag_1_wp * kh_smag_1_wp

    kh_smag_2_vp = -tmp2[E2C2VDim(2)] + tmp2[E2C2VDim(3)]

    kh_smag_2_wp = astype(kh_smag_2_vp, wpfloat)
    kh_smag_2_wp = (kh_smag_2_wp * inv_vert_vert_length) - (
        astype(dvt_tang_vp, wpfloat) * inv_primal_edge_length
    )

    kh_smag_2_wp = kh_smag_2_wp * kh_smag_2_wp

    kh_smag_e_wp = diff_multfac_smag_wp * sqrt(kh_smag_2_wp + kh_smag_1_wp)
    z_nabla2_e_wp = (
        astype(tmp2[E2C2VDim(0)] + tmp2[E2C2VDim(1)], wpfloat) - wpfloat("2.0") * vn
    ) * (inv_primal_edge_length * inv_primal_edge_length) + (
        astype(tmp2[E2C2VDim(2)] + tmp2[E2C2VDim(3)], wpfloat) - wpfloat("2.0") * vn
    ) * (inv_vert_vert_length * inv_vert_vert_length)

    z_nabla2_e_wp = wpfloat("4.0") * z_nabla2_e_wp

    kh_smag_ec_wp = kh_smag_e_wp
    kh_smag_e_vp = maximum(vpfloat("0.0"), astype(kh_smag_e_wp - smag_offset_wp, vpfloat))
    kh_smag_e_vp = minimum(kh_smag_e_vp, smag_limit)

    return kh_smag_e_vp, astype(kh_smag_ec_wp, vpfloat), z_nabla2_e_wp


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    smag_limit: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    kh_smag_ec: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    smag_offset: vpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_nabla2_and_smag_coefficients_for_vn(
        diff_multfac_smag,
        tangent_orientation,
        inv_primal_edge_length,
        inv_vert_vert_length,
        u_vert,
        v_vert,
        primal_normal_vert_x,
        primal_normal_vert_y,
        dual_normal_vert_x,
        dual_normal_vert_y,
        vn,
        smag_limit,
        smag_offset,
        out=(kh_smag_e, kh_smag_ec, z_nabla2_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
