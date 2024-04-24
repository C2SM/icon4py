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
from gt4py.next import Field, field_operator, int32, program

from icon4py.model.common.dimension import (
    E2V,
    CellDim,
    EdgeDim,
    KDim,
    Koff,
    VertexDim,
)
from icon4py.model.common.math.helpers import grad_fd_norm


@field_operator
def _grad_fd_tang(
    psi_v: Field[[VertexDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
) -> Field[[EdgeDim, KDim], float]:
    grad_tang_psi_e = tangent_orientation * (psi_v(E2V[1]) - psi_v(E2V[0])) * inv_primal_edge_length
    return grad_tang_psi_e


@program
def compute_ddxn_z_half_e(
    z_ifc: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    ddxn_z_half_e: Field[[EdgeDim, KDim], float],
    horizontal_lower: int32,
    horizontal_upper: int32,
    vertical_lower: int32,
    vertical_upper: int32,
):
    grad_fd_norm(
        z_ifc,
        inv_dual_edge_length,
        out=ddxn_z_half_e,
        domain={
            EdgeDim: (horizontal_lower, horizontal_upper),
            KDim: (vertical_lower, vertical_upper),
        },
    )


@program
def compute_ddxt_z_half_e(
    z_ifv: Field[[VertexDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    ddxt_z_half_e: Field[[EdgeDim, KDim], float],
    horizontal_lower: int32,
    horizontal_upper: int32,
    vertical_lower: int32,
    vertical_upper: int32,
):
    _grad_fd_tang(
        z_ifv,
        inv_primal_edge_length,
        tangent_orientation,
        out=ddxt_z_half_e,
        domain={
            EdgeDim: (horizontal_lower, horizontal_upper),
            KDim: (vertical_lower, vertical_upper),
        },
    )


@field_operator
def _compute_ddxnt_z_full(
    z_ddxnt_z_half_e: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    ddxnt_z_full = 0.5 * (z_ddxnt_z_half_e + z_ddxnt_z_half_e(Koff[1]))
    return ddxnt_z_full


def compute_cells_aw_verts(
    dual_area: np.array,
    edge_vert_length: np.array,
    edge_cell_length: np.array,
    owner_mask: np.array,
    e2c: np.array,
    v2c: np.array,
    v2e: np.array,
    e2v: np.array,
    horizontal_start: np.int32,
    horizontal_end: np.int32,
) -> np.array:
    llb = horizontal_start
    cells_aw_verts = np.zeros([horizontal_end, 6])
    index = np.repeat(np.arange(horizontal_end, dtype=float), 6).reshape(horizontal_end, 6)
    idx_ve = np.where(index == e2v[v2e, 0], 0, 1)

    for i in range(2):
        for j in range(6):
            for k in range(6):
                cells_aw_verts[llb:, k] = np.where(
                    np.logical_and(owner_mask[:], e2c[v2e[llb:, j], i] == v2c[llb:, k]),
                    cells_aw_verts[llb:, k]
                    + 0.5
                    / dual_area[llb:]
                    * edge_vert_length[v2e[llb:, j], idx_ve[llb:, j]]
                    * edge_cell_length[v2e[llb:, j], i],
                    cells_aw_verts[llb:, k],
                )
    return cells_aw_verts
