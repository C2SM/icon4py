# icon4pY - icon INSPIRED CODE IN pYTHON AND gt4Py
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

import numpy as np
from gt4py.next import where, neighbor_sum
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import C2E, E2C, V2E, E2V, V2C, C2EDim, CellDim, EdgeDim, V2EDim, V2CDim, VertexDim, KDim, Koff

@field_operator
def grad_fd_norm(
    psi_c: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
) -> Field[[EdgeDim, KDim], float]:
    grad_norm_psi_e = (psi_c(E2C[1]) - psi_c(E2C[0])) * inv_dual_edge_length
    return grad_norm_psi_e

@field_operator
def grad_fd_tang(
    psi_v: Field[[VertexDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
) -> Field[[EdgeDim, KDim], float]:
    grad_tang_psi_e = tangent_orientation * (psi_v(E2V[1]) - psi_v(E2V[0])) * inv_primal_edge_length
    return grad_tang_psi_e

@field_operator
def compute_ddxn_z_half_e(
    z_ifc: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
) -> Field[[EdgeDim, KDim], float]:
    ddxn_z_half_e = grad_fd_norm(z_ifc, inv_dual_edge_length)
    return ddxn_z_half_e

@field_operator
def compute_ddxt_z_half_e(
    z_ifv: Field[[VertexDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
) -> Field[[EdgeDim, KDim], float]:
    ddxt_z_half_e = grad_fd_tang(z_ifv, inv_primal_edge_length, tangent_orientation)
    return ddxt_z_half_e

@field_operator
def compute_ddxnt_z_full(
    z_ddxnt_z_half_e: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    ddxnt_z_full = 0.5 * (z_ddxnt_z_half_e + z_ddxnt_z_half_e(Koff[1]))
    return ddxnt_z_full

@field_operator
def compute_cells2verts_scalar(
    p_cell_in: Field[[CellDim, KDim], float],
    c_int: Field[[VertexDim, V2CDim], float],
) -> Field[[VertexDim, KDim], float]:
    p_vert_out = neighbor_sum(c_int * p_cell_in(V2C), axis=V2CDim)
    return p_vert_out

def compute_cells_aw_verts(
    dual_area: np.array,
    edge_vert_length: np.array,
    edge_cell_length: np.array,
    owner_mask: np.array,
    e2c: np.array,
    v2c: np.array,
    v2e: np.array,
    e2v: np.array,
    second_boundary_layer_start_index: np.int32,
    second_boundary_layer_end_index: np.int32,
) -> np.array:
    llb = second_boundary_layer_start_index
    cells_aw_verts = np.zeros([second_boundary_layer_end_index, 6])
    index = np.zeros([second_boundary_layer_end_index, 6])
    for j in range (6):
        index[:, j] = np.arange(second_boundary_layer_end_index)
    idx_ve = np.where(index == e2v[v2e, 0], 0, 1)

    for i in range(2):
        for j in range (6):
            for k in range (6):
                cells_aw_verts[llb:, k] = np.where(np.logical_and(owner_mask[:], e2c[v2e[llb:, j], i] == v2c[llb:, k]), cells_aw_verts[llb:, k] + 0.5 / dual_area[llb:] * edge_vert_length[v2e[llb:, j], idx_ve[llb:, j]] * edge_cell_length[v2e[llb:, j], i], cells_aw_verts[llb:, k])
    return cells_aw_verts
