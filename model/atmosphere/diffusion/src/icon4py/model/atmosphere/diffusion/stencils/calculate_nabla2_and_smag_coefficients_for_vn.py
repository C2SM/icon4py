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
from gt4py.next.ffront.fbuiltins import Field, int32, maximum, minimum, sqrt

from icon4py.model.common.dimension import E2C2V, E2ECV, ECVDim, EdgeDim, KDim, VertexDim


@field_operator
def _calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: Field[[KDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    inv_vert_vert_length: Field[[EdgeDim], float],
    u_vert: Field[[VertexDim, KDim], float],
    v_vert: Field[[VertexDim, KDim], float],
    primal_normal_vert_x: Field[[ECVDim], float],
    primal_normal_vert_y: Field[[ECVDim], float],
    dual_normal_vert_x: Field[[ECVDim], float],
    dual_normal_vert_y: Field[[ECVDim], float],
    vn: Field[[EdgeDim, KDim], float],
    smag_limit: Field[[KDim], float],
    smag_offset: float,
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    dvt_tang = (
        -(
            u_vert(E2C2V[0]) * dual_normal_vert_x(E2ECV[0])
            + v_vert(E2C2V[0]) * dual_normal_vert_y(E2ECV[0])
        )
    ) + (
        u_vert(E2C2V[1]) * dual_normal_vert_x(E2ECV[1])
        + v_vert(E2C2V[1]) * dual_normal_vert_y(E2ECV[1])
    )

    dvt_norm = (
        -(
            u_vert(E2C2V[2]) * dual_normal_vert_x(E2ECV[2])
            + v_vert(E2C2V[2]) * dual_normal_vert_y(E2ECV[2])
        )
    ) + (
        u_vert(E2C2V[3]) * dual_normal_vert_x(E2ECV[3])
        + v_vert(E2C2V[3]) * dual_normal_vert_y(E2ECV[3])
    )

    kh_smag_1 = (
        -(
            u_vert(E2C2V[0]) * primal_normal_vert_x(E2ECV[0])
            + v_vert(E2C2V[0]) * primal_normal_vert_y(E2ECV[0])
        )
    ) + (
        u_vert(E2C2V[1]) * primal_normal_vert_x(E2ECV[1])
        + v_vert(E2C2V[1]) * primal_normal_vert_y(E2ECV[1])
    )

    dvt_tang = dvt_tang * tangent_orientation

    kh_smag_1 = (kh_smag_1 * tangent_orientation * inv_primal_edge_length) + (
        dvt_norm * inv_vert_vert_length
    )

    kh_smag_1 = kh_smag_1 * kh_smag_1

    kh_smag_2 = (
        -(
            u_vert(E2C2V[2]) * primal_normal_vert_x(E2ECV[2])
            + v_vert(E2C2V[2]) * primal_normal_vert_y(E2ECV[2])
        )
    ) + (
        u_vert(E2C2V[3]) * primal_normal_vert_x(E2ECV[3])
        + v_vert(E2C2V[3]) * primal_normal_vert_y(E2ECV[3])
    )

    kh_smag_2 = (kh_smag_2 * inv_vert_vert_length) - (dvt_tang * inv_primal_edge_length)

    kh_smag_2 = kh_smag_2 * kh_smag_2

    kh_smag_e = diff_multfac_smag * sqrt(kh_smag_2 + kh_smag_1)
    z_nabla2_e = (
        (
            (
                u_vert(E2C2V[0]) * primal_normal_vert_x(E2ECV[0])
                + v_vert(E2C2V[0]) * primal_normal_vert_y(E2ECV[0])
            )
            + (
                u_vert(E2C2V[1]) * primal_normal_vert_x(E2ECV[1])
                + v_vert(E2C2V[1]) * primal_normal_vert_y(E2ECV[1])
            )
        )
        - 2.0 * vn
    ) * (inv_primal_edge_length**2)
    z_nabla2_e = z_nabla2_e + (
        (
            (
                u_vert(E2C2V[2]) * primal_normal_vert_x(E2ECV[2])
                + v_vert(E2C2V[2]) * primal_normal_vert_y(E2ECV[2])
            )
            + (
                u_vert(E2C2V[3]) * primal_normal_vert_x(E2ECV[3])
                + v_vert(E2C2V[3]) * primal_normal_vert_y(E2ECV[3])
            )
        )
        - 2.0 * vn
    ) * (inv_vert_vert_length**2)

    z_nabla2_e = 4.0 * z_nabla2_e

    kh_smag_ec = kh_smag_e
    kh_smag_e = maximum(0.0, kh_smag_e - smag_offset)
    kh_smag_e = minimum(kh_smag_e, smag_limit)

    return kh_smag_e, kh_smag_ec, z_nabla2_e


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: Field[[KDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    inv_vert_vert_length: Field[[EdgeDim], float],
    u_vert: Field[[VertexDim, KDim], float],
    v_vert: Field[[VertexDim, KDim], float],
    primal_normal_vert_x: Field[[ECVDim], float],
    primal_normal_vert_y: Field[[ECVDim], float],
    dual_normal_vert_x: Field[[ECVDim], float],
    dual_normal_vert_y: Field[[ECVDim], float],
    vn: Field[[EdgeDim, KDim], float],
    smag_limit: Field[[KDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    kh_smag_ec: Field[[EdgeDim, KDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    smag_offset: float,
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
