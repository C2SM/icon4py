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
from gt4py.next import Field, program

from icon4py.model.common.dimension import (
    EdgeDim,
    KDim,
)
from icon4py.model.common.math.helpers import average_edge_kdim_level_up


def compute_c_lin_e(
    edge_cell_length: np.array,
    inv_dual_edge_length: np.array,
    owner_mask: np.array,
    second_boundary_layer_start_index: np.int32,
) -> np.array:
    """
    Compute E2C average inverse distance.

    Args:
        edge_cell_length: numpy array, representing a Field[[EdgeDim, E2CDim], float]
        inv_dual_edge_length: inverse dual edge length, numpy array representing a Field[[EdgeDim], float]
        owner_mask: numpy array, representing a Field[[EdgeDim], bool]boolean field, True for all edges owned by this compute node
        second_boundary_layer_start_index: start index of the 2nd boundary line: c_lin_e is not calculated for the first boundary layer

    Returns: c_lin_e: numpy array  representing Field[[EdgeDim, E2CDim], float]

    """
    c_lin_e_ = edge_cell_length[:, 1] * inv_dual_edge_length
    c_lin_e = np.transpose(np.vstack((c_lin_e_, (1.0 - c_lin_e_))))
    c_lin_e[0:second_boundary_layer_start_index, :] = 0.0
    mask = np.transpose(np.tile(owner_mask, (2, 1)))
    return np.where(mask, c_lin_e, 0.0)


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


@program
def compute_ddxnt_z_full(
    z_ddxnt_z_half_e: Field[[EdgeDim, KDim], float], ddxn_z_full: Field[[EdgeDim, KDim], float]
):
    average_edge_kdim_level_up(z_ddxnt_z_half_e, out=ddxn_z_full)
