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
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import E2C2V, E2ECV, ECVDim, EdgeDim, KDim, VertexDim


@field_operator
def _calculate_nabla4(
    u_vert: Field[[VertexDim, KDim], float],
    v_vert: Field[[VertexDim, KDim], float],
    primal_normal_vert_v1: Field[[ECVDim], float],
    primal_normal_vert_v2: Field[[ECVDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    inv_vert_vert_length: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
) -> Field[[EdgeDim, KDim], float]:
    nabv_tang = u_vert(E2C2V[0]) * primal_normal_vert_v1(E2ECV[0])
        + v_vert(E2C2V[0]) * primal_normal_vert_v2(E2ECV[0])
        + u_vert(E2C2V[1]) * primal_normal_vert_v1(E2ECV[1])
        + v_vert(E2C2V[1]) * primal_normal_vert_v2(E2ECV[1])

    nabv_norm = u_vert(E2C2V[2]) * primal_normal_vert_v1(E2ECV[2])
        + v_vert(E2C2V[2]) * primal_normal_vert_v2(E2ECV[2])
        + u_vert(E2C2V[3]) * primal_normal_vert_v1(E2ECV[3])
        + v_vert(E2C2V[3]) * primal_normal_vert_v2(E2ECV[3])

    z_nabla4_e2 = 4.0 * (
        (nabv_norm - 2.0 * z_nabla2_e) * (inv_vert_vert_length*inv_vert_vert_length)
        + (nabv_tang - 2.0 * z_nabla2_e) * (inv_primal_edge_length*inv_primal_edge_length)
    )
    return z_nabla4_e2


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_nabla4(
    u_vert: Field[[VertexDim, KDim], float],
    v_vert: Field[[VertexDim, KDim], float],
    primal_normal_vert_v1: Field[[ECVDim], float],
    primal_normal_vert_v2: Field[[ECVDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    inv_vert_vert_length: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    z_nabla4_e2: Field[[EdgeDim, KDim], float],
):
    _calculate_nabla4(
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        out=z_nabla4_e2,
    )
