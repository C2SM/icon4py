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
from gt4py.next import where
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import C2E, V2E, C2EDim, CellDim, EdgeDim, V2EDim, VertexDim

def grad_fd_norm(
    psi_c: np.array,
    inv_dual_edge_length: np.array,
    e2c: np.array,
    second_boundary_layer_start_index: np.int32,
    nlev,
) -> np.array:
    llb = second_boundary_layer_start_index
    grad_norm_psi_e = np.zeros([len(e2c[:, 0]), nlev])
    for i in range(nlev):
        grad_norm_psi_e[llb:, i] = (psi_c[e2c[llb:, 1], i] - psi_c[e2c[llb:, 0], i]) * inv_dual_edge_length[llb:]
    return grad_norm_psi_e

def grad_fd_tang(
    psi_v: np.array,
    inv_primal_edge_length: np.array,
    tangent_orientation: np.array,
    e2v: np.array,
    third_boundary_layer_start_index: np.int32,
    nlev,
) -> np.array:
    llb = third_boundary_layer_start_index
    grad_tang_psi_e = np.zeros([e2v.shape[0], nlev])
    for i in range(nlev):
        grad_tang_psi_e[llb:, i] = tangent_orientation[llb:] * (psi_v[e2v[llb:, 1], i] - psi_v[e2v[llb:, 0], i]) * inv_primal_edge_length[llb:]
    return grad_tang_psi_e

def compute_ddxn_z_half_e(
    z_ifc: np.array,
    inv_dual_edge_length: np.array,
    e2c: np.array,
    second_boundary_layer_start_index: np.int32,
) -> np.array:
    nlev = z_ifc.shape[1]
    ddxn_z_half_e = grad_fd_norm(z_ifc, inv_dual_edge_length, e2c, second_boundary_layer_start_index, nlev)
    return ddxn_z_half_e

def compute_ddxt_z_half_e(
    z_ifv: np.array,
    inv_primal_edge_length: np.array,
    tangent_orientation: np.array,
    e2v: np.array,
    third_boundary_layer_start_index: np.int32,
) -> np.array:
    nlev = z_ifv.shape[1]
    ddxt_z_half_e = grad_fd_tang(z_ifv, inv_primal_edge_length, tangent_orientation, e2v, third_boundary_layer_start_index, nlev)
    return ddxt_z_half_e

def compute_ddxnt_z_full(
    z_ddxnt_z_half_e: np.array,
) -> np.array:
    ddxnt_z_full = 0.5 * (z_ddxnt_z_half_e[:, :z_ddxnt_z_half_e.shape[1]-1] + z_ddxnt_z_half_e[:, 1:])
    return ddxnt_z_full

def compute_cells2verts_scalar(
    p_cell_in: np.array,
    c_int: np.array,
    v2c: np.array,
    second_boundary_layer_start_index: np.int32,
) -> np.array:
    llb = second_boundary_layer_start_index
    kdim = p_cell_in.shape[1]
    p_vert_out = np.zeros([c_int.shape[0], kdim])
    for j in range(kdim):
        for i in range(6):
            p_vert_out[llb:, j] = p_vert_out[llb:, j] + c_int[llb:, i] * p_cell_in[v2c[llb:, i], j]
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
