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
from gt4py.next.ffront.fbuiltins import Field, astype, int32, maximum, minimum, sqrt
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import E2C2V, E2ECV, ECVDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: Field[[KDim], vpfloat],
    tangent_orientation: fa.EwpField,
    inv_primal_edge_length: fa.EwpField,
    inv_vert_vert_length: fa.EwpField,
    u_vert: Field[[VertexDim, KDim], vpfloat],
    v_vert: Field[[VertexDim, KDim], vpfloat],
    primal_normal_vert_x: Field[[ECVDim], wpfloat],
    primal_normal_vert_y: Field[[ECVDim], wpfloat],
    dual_normal_vert_x: Field[[ECVDim], wpfloat],
    dual_normal_vert_y: Field[[ECVDim], wpfloat],
    vn: fa.EKwpField,
    smag_limit: Field[[KDim], vpfloat],
    smag_offset: vpfloat,
) -> tuple[
    Field[[EdgeDim, KDim], vpfloat],
    Field[[EdgeDim, KDim], vpfloat],
    fa.EKwpField,
]:
    diff_multfac_smag_wp, u_vert_wp, v_vert_wp, smag_offset_wp = astype(
        (diff_multfac_smag, u_vert, v_vert, smag_offset), wpfloat
    )

    dvt_tang_wp = (
        -(
            u_vert_wp(E2C2V[0]) * dual_normal_vert_x(E2ECV[0])
            + v_vert_wp(E2C2V[0]) * dual_normal_vert_y(E2ECV[0])
        )
    ) + (
        u_vert_wp(E2C2V[1]) * dual_normal_vert_x(E2ECV[1])
        + v_vert_wp(E2C2V[1]) * dual_normal_vert_y(E2ECV[1])
    )

    dvt_norm_vp = astype(
        (
            -(
                u_vert_wp(E2C2V[2]) * dual_normal_vert_x(E2ECV[2])
                + v_vert_wp(E2C2V[2]) * dual_normal_vert_y(E2ECV[2])
            )
        )
        + (
            u_vert_wp(E2C2V[3]) * dual_normal_vert_x(E2ECV[3])
            + v_vert_wp(E2C2V[3]) * dual_normal_vert_y(E2ECV[3])
        ),
        vpfloat,
    )

    kh_smag_1_vp = (
        -astype(
            u_vert_wp(E2C2V[0]) * primal_normal_vert_x(E2ECV[0])
            + v_vert_wp(E2C2V[0]) * primal_normal_vert_y(E2ECV[0]),
            vpfloat,
        )
    ) + astype(
        u_vert_wp(E2C2V[1]) * primal_normal_vert_x(E2ECV[1])
        + v_vert_wp(E2C2V[1]) * primal_normal_vert_y(E2ECV[1]),
        vpfloat,
    )

    dvt_tang_vp = astype(dvt_tang_wp * tangent_orientation, vpfloat)

    kh_smag_1_wp, dvt_norm_wp = astype((kh_smag_1_vp, dvt_norm_vp), wpfloat)
    kh_smag_1_wp = (kh_smag_1_wp * tangent_orientation * inv_primal_edge_length) + (
        dvt_norm_wp * inv_vert_vert_length
    )

    kh_smag_1_wp = kh_smag_1_wp * kh_smag_1_wp

    kh_smag_2_vp = (
        -astype(
            u_vert_wp(E2C2V[2]) * primal_normal_vert_x(E2ECV[2])
            + v_vert_wp(E2C2V[2]) * primal_normal_vert_y(E2ECV[2]),
            vpfloat,
        )
    ) + astype(
        u_vert_wp(E2C2V[3]) * primal_normal_vert_x(E2ECV[3])
        + v_vert_wp(E2C2V[3]) * primal_normal_vert_y(E2ECV[3]),
        vpfloat,
    )

    kh_smag_2_wp = astype(kh_smag_2_vp, wpfloat)
    kh_smag_2_wp = (kh_smag_2_wp * inv_vert_vert_length) - (
        astype(dvt_tang_vp, wpfloat) * inv_primal_edge_length
    )

    kh_smag_2_wp = kh_smag_2_wp * kh_smag_2_wp

    kh_smag_e_wp = diff_multfac_smag_wp * sqrt(kh_smag_2_wp + kh_smag_1_wp)
    z_nabla2_e_wp = (
        astype(
            astype(
                u_vert_wp(E2C2V[0]) * primal_normal_vert_x(E2ECV[0])
                + v_vert_wp(E2C2V[0]) * primal_normal_vert_y(E2ECV[0]),
                vpfloat,
            )
            + astype(
                u_vert_wp(E2C2V[1]) * primal_normal_vert_x(E2ECV[1])
                + v_vert_wp(E2C2V[1]) * primal_normal_vert_y(E2ECV[1]),
                vpfloat,
            ),
            wpfloat,
        )
        - wpfloat("2.0") * vn
    ) * (inv_primal_edge_length * inv_primal_edge_length)
    z_nabla2_e_wp = z_nabla2_e_wp + (
        astype(
            astype(
                u_vert_wp(E2C2V[2]) * primal_normal_vert_x(E2ECV[2])
                + v_vert_wp(E2C2V[2]) * primal_normal_vert_y(E2ECV[2]),
                vpfloat,
            )
            + astype(
                u_vert_wp(E2C2V[3]) * primal_normal_vert_x(E2ECV[3])
                + v_vert_wp(E2C2V[3]) * primal_normal_vert_y(E2ECV[3]),
                vpfloat,
            ),
            wpfloat,
        )
        - wpfloat("2.0") * vn
    ) * (inv_vert_vert_length * inv_vert_vert_length)

    z_nabla2_e_wp = wpfloat("4.0") * z_nabla2_e_wp

    kh_smag_ec_wp = kh_smag_e_wp
    kh_smag_e_vp = maximum(vpfloat("0.0"), astype(kh_smag_e_wp - smag_offset_wp, vpfloat))
    kh_smag_e_vp = minimum(kh_smag_e_vp, smag_limit)

    return kh_smag_e_vp, astype(kh_smag_ec_wp, vpfloat), z_nabla2_e_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: Field[[KDim], vpfloat],
    tangent_orientation: fa.EwpField,
    inv_primal_edge_length: fa.EwpField,
    inv_vert_vert_length: fa.EwpField,
    u_vert: Field[[VertexDim, KDim], vpfloat],
    v_vert: Field[[VertexDim, KDim], vpfloat],
    primal_normal_vert_x: Field[[ECVDim], wpfloat],
    primal_normal_vert_y: Field[[ECVDim], wpfloat],
    dual_normal_vert_x: Field[[ECVDim], wpfloat],
    dual_normal_vert_y: Field[[ECVDim], wpfloat],
    vn: fa.EKwpField,
    smag_limit: Field[[KDim], vpfloat],
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    kh_smag_ec: Field[[EdgeDim, KDim], vpfloat],
    z_nabla2_e: fa.EKwpField,
    smag_offset: vpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
