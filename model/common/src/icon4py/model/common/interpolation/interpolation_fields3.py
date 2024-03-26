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
    second_boundary_layer_end_index: np.int32,
    nlev,
) -> np.array:
    llb = second_boundary_layer_start_index
    grad_norm_psi_e = np.zeros([second_boundary_layer_end_index, nlev])
    for i in range(nlev):
        grad_norm_psi_e[llb:, i] = (psi_c[e2c[llb:, 1], i] - psi_c[e2c[llb:, 0], i]) * inv_dual_edge_length[llb:]
    return grad_norm_psi_e

def grad_fd_tang(
    psi_c: np.array,
    inv_primal_edge_length: np.array,
    e2v: np.array,
    third_boundary_layer_start_index: np.int32,
    second_boundary_layer_end_index: np.int32,
    nlev,
) -> np.array:
    llb = third_boundary_layer_start_index
    grad_tang_psi_e = np.zeros([second_boundary_layer_end_index, nlev])
    for i in range(nlev):
        grad_norm_psi_e[llb:, i] = (psi_v[e2v[llb:, 1], i] - psi_v[e2v[llb:, 0], i]) * inv_primal_edge_length[llb:]
    return grad_tang_psi_e

def compute_ddxn_z_half_e(
    z_ifc: np.array,
    inv_dual_edge_length: np.array,
    e2c: np.array,
    second_boundary_layer_start_index: np.int32,
    second_boundary_layer_end_index: np.int32,
) -> np.array:
    nlev = z_ifc.shape[1]
    ddxn_z_half_e = grad_fd_norm(z_ifc, inv_dual_edge_length, e2c, second_boundary_layer_start_index, second_boundary_layer_end_index, nlev)
    return ddxn_z_half_e

def compute_ddxt_z_half_e(
    z_ifv: np.array,
    inv_primal_edge_length: np.array,
    e2v: np.array,
    third_boundary_layer_start_index: np.int32,
    second_boundary_layer_end_index: np.int32,
) -> np.array:
    nlev = z_ifv.shape[1]
    return ddxt_z_half_e

def compute_ddxnt_z_full(
    z_ddxnt_z_half_e: np.array,
) -> np.array:
    ddxnt_z_full = 0.5 * (z_ddxnt_z_half_e[:, :z_ddxnt_z_half_e.shape[1]-1] + z_ddxnt_z_half_e[:, 1:])
    return ddxnt_z_full

def cells2verts_scalar(
    c_int: np.array,
    p_cell_in: np.array,
    v2c: np.array,
) -> np.array:
#    iz_ifc: np.array,
#    cells_aw_verts: np.array,
    p_vert_out = np.zeros()
    for i in range(6):
        p_vert_out[:, i] = p_vert_out[:, i] + p_cell_in[v2c[:, 0]]
#           c_int(jv,1,jb) * p_cell_in(iidx(jv,jb,1),jk,iblk(jv,jb,1)) + &
#           c_int(jv,2,jb) * p_cell_in(iidx(jv,jb,2),jk,iblk(jv,jb,2)) + &
#           c_int(jv,3,jb) * p_cell_in(iidx(jv,jb,3),jk,iblk(jv,jb,3)) + &
#           c_int(jv,4,jb) * p_cell_in(iidx(jv,jb,4),jk,iblk(jv,jb,4)) + &
#           c_int(jv,5,jb) * p_cell_in(iidx(jv,jb,5),jk,iblk(jv,jb,5)) + &
#           c_int(jv,6,jb) * p_cell_in(iidx(jv,jb,6),jk,iblk(jv,jb,6))
    return p_vert_out

def compute_cells_aw_verts(
    dual_area: np.array,
    edge_vert_length: np.array,
    edge_cell_length: np.array,
    e2c: np.array,
    v2c: np.array,
    v2e: np.array,
    e2v: np.array,
    second_boundary_layer_end_index: np.int32,
) -> np.array:
    cells_aw_verts = np.zeros([second_boundary_layer_end_index, 6])
    index = np.zeros([second_boundary_layer_end_index, 6])
    for j in range (6):
        index[:, j] = np.arange(second_boundary_layer_end_index)
    idx_ve = np.where(index == e2v[v2e, 0], 0, 1)

    for i in range(2):
        for j in range (6):
            for k in range (6):
                cells_aw_verts[:, k] = np.where(e2c[v2e[:, j], i] == v2c[:, k], cells_aw_verts[:, k] + 0.5 / dual_area[:] * edge_vert_length[v2e[:, j], idx_ve[:, j]] * edge_cell_length[v2e[:, j], i], cells_aw_verts[:, k])
    return cells_aw_verts
