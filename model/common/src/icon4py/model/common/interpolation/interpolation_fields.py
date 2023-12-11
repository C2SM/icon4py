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
from icon4py.model.common.dimension import CellDim, C2EDim, EdgeDim, C2E, CEDim, C2CE
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.fbuiltins import Field

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

#    DO jb = i_startblk, i_endblk
#
#      CALL get_indices_c(ptr_patch, jb, i_startblk, i_endblk, &
#        & i_startidx, i_endidx, rl_start, rl_end)
#
#      DO je = 1, ptr_patch%geometry_info%cell_type
#        DO jc = i_startidx, i_endidx
#
#          IF (je > ptr_patch%cells%num_edges(jc,jb)) CYCLE ! relevant for hexagons
#
#          ile = ptr_patch%cells%edge_idx(jc,jb,je)
#          ibe = ptr_patch%cells%edge_blk(jc,jb,je)
#
#          ptr_int%geofac_div(jc,je,jb) = &
#            & ptr_patch%edges%primal_edge_length(ile,ibe) * &
#            & ptr_patch%cells%edge_orientation(jc,jb,je)  / &
#            & ptr_patch%cells%area(jc,jb)
#
#        ENDDO !cell loop
#      ENDDO

@field_operator
def compute_geofac_div(
    primal_edge_length: Field[[EdgeDim], float],
    edge_orientation: Field[[CellDim, C2EDim], float],
    area: Field[[CellDim], float],
) -> Field[[CellDim, C2EDim], float]:
    """
    Args:
        primal_edge_length:
        edge_orientation:
        area:
    """
    geofac_div_ = primal_edge_length(C2E)*edge_orientation/area
    return geofac_div_

#def compute_geofac_rot(
#    dual_edge_length: np.array,
#    edge_orientation: np.array,
#    area: np.array,
#) -> np.array:
#    geofac_rot_ = dual_edge_length*edge_orientation/dual_area
#    return geofac_rot_
#
#def compute_geofac_n2s(
#    geofac_div: np.array,
#     inv_dual_edge_length: np.array,
#    edges_cell_idx: np.array,
#) -> np.array:
#    if (edges_cell_idx == 1):
#        geofac_n2s_ = geofac_n2s_ - geofac_div / inv_dual_edge_length
#    else:
#        geofac_n2s_ = geofac_n2s_ + geofac_div / inv_dual_edge_length
#    return geofac_n2s_
#
#def compute_geofac_qdiv(
#    primal_edge_length: np.array,
#    quad_orientation: np.array,
#    quad_area: np.array,
#) -> np.array:
#    geofac_qdiv_ = primal_edge_length * quad_orientation / quad_area
#    return geofac_qdiv_
#
#def compute_geofac_grdiv(
#    geofac_div: np.array,
#    inv_dual_edge_length: np.array,
#) -> np.array:
#    geofac_grdiv_ = geofac_grdiv_ - geofac_div * inv_dual_edge_length
#    return geofac_grdiv_
