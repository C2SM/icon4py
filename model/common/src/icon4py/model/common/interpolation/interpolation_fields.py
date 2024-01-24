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
from icon4py.model.common.dimension import CellDim, C2EDim, EdgeDim, C2E, CEDim, C2CE, VertexDim, V2E, V2EDim, E2CDim
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.fbuiltins import Field
from gt4py.next import where, neighbor_sum

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

@field_operator
def compute_geofac_rot(
    dual_edge_length: Field[[EdgeDim], float],
    edge_orientation: Field[[VertexDim, V2EDim], float],
    dual_area: Field[[VertexDim], float],
    owner_mask: Field[[VertexDim], bool],
) -> Field[[VertexDim, V2EDim], float]:
    """
    Args:
        dual_edge_length:
        edge_orientation:
        dual_area:
        owner_mask:
    """
#    geofac_rot_ = dual_edge_length(V2E)*edge_orientation/dual_area
    geofac_rot_ = where(owner_mask, dual_edge_length(V2E)*edge_orientation/dual_area, 0.0)
    return geofac_rot_

#@field_operator
#def compute_geofac_n2s_1(
#    geofac_n2s: Field[[CellDim], float],
#    dual_edge_length: Field[[EdgeDim], float],
#    geofac_div: Field[[CellDim, C2EDim], float],
#) -> Field[[CellDim], float]:
#    """
#    Args:
#        dual_edge_length:
#        edge_orientation:
#        dual_area:
#        owner_mask:
#    """
##    fac = 
##    geofac_n2s_ = where(fac, geofac_n2s - geofac_div/dual_edge_length(C2E), geofac_n2s)
#    geofac_n2s_ = geofac_n2s - neighbor_sum(geofac_div/dual_edge_length(C2E), axis=C2EDim)
#    return geofac_n2s_
#
#def compute_geofac_n2s(
#    geofac_n2s: np.array,
#    dual_edge_length: np.array,
#    geofac_div: np.array,
#    C2E_: np.array,
#    E2C_: np.array,
#    C2E2C_: np.array,
#    lateral_boundary: np.array,
#    grid_savepoint, interpolation_savepoint, icon_grid,
#) -> np.array:
#    """
#    """
#    compute_geofac_n2s_1(
#	geofac_n2s[:, 0],
#	dual_edge_length,
#	geofac_div,
#        out=geofac_n2s[lateral_boundary[0]:, :],
#        offset_provider={"C2E": icon_grid.get_offset_provider("C2E"), "C2EDim": icon_grid.get_offset_provider("C2EDim")}
#    )
#    return geofac_n2s

def compute_geofac_n2s(
    geofac_n2s: np.array,
    dual_edge_length: np.array,
    geofac_div: np.array,
    C2E_: np.array,
    E2C_: np.array,
    C2E2C_: np.array,
    lateral_boundary: np.array,
) -> np.array:
    """
    Args:
    	geofac_n2s: np.array,
    	dual_edge_length: np.array,
    	geofac_div: np.array,
    	C2E_: np.array,
    	E2C_: np.array,
    	C2E2C_: np.array,
    	lateral_boundary: np.array,
    """
    llb = lateral_boundary[0]
    index = np.transpose(np.vstack((np.arange(lateral_boundary[1]), np.arange(lateral_boundary[1]), np.arange(lateral_boundary[1]))))
    fac = E2C_[C2E_, 0] == index
    geofac_n2s[llb:, 0] = (geofac_n2s[llb:, 0]
                       - fac[llb:, 0] * (geofac_div/dual_edge_length[C2E_])[llb:, 0]
                       - fac[llb:, 1] * (geofac_div/dual_edge_length[C2E_])[llb:, 1]
                       - fac[llb:, 2] * (geofac_div/dual_edge_length[C2E_])[llb:, 2])
    fac = E2C_[C2E_, 1] == index
    geofac_n2s[llb:, 0] = (geofac_n2s[llb:, 0]
                       + fac[llb:, 0] * (geofac_div/dual_edge_length[C2E_])[llb:, 0]
                       + fac[llb:, 1] * (geofac_div/dual_edge_length[C2E_])[llb:, 1]
                       + fac[llb:, 2] * (geofac_div/dual_edge_length[C2E_])[llb:, 2])
    fac = E2C_[C2E_, 0] == C2E2C_
    geofac_n2s[llb:, 1:] = geofac_n2s[llb:, 1:] - fac[llb:, :] * (geofac_div/dual_edge_length[C2E_])[llb:, :]
    fac = E2C_[C2E_, 1] == C2E2C_
    geofac_n2s[llb:, 1:] = geofac_n2s[llb:, 1:] + fac[llb:, :] * (geofac_div/dual_edge_length[C2E_])[llb:, :]
    return geofac_n2s

def compute_primal_normal_ec(
    primal_normal_ec: np.array,
    primal_normal_cell_x: np.array,
    primal_normal_cell_y: np.array,
    owner_mask: np.array,
    C2E_: np.array,
    E2C_: np.array,
    lateral_boundary: np.array,
) -> np.array:
    llb = lateral_boundary[0]
    index = np.transpose(np.vstack((np.arange(lateral_boundary[1]), np.arange(lateral_boundary[1]), np.arange(lateral_boundary[1]))))
    for i in range(2):
        fac = E2C_[C2E_, i] == index
        primal_normal_ec[llb:, :, 0] = primal_normal_ec[llb:, :, 0] + np.where(owner_mask, fac[llb:, :] * primal_normal_cell_x[C2E_[llb:], i], 0.0)
        primal_normal_ec[llb:, :, 1] = primal_normal_ec[llb:, :, 1] + np.where(owner_mask, fac[llb:, :] * primal_normal_cell_y[C2E_[llb:], i], 0.0)
    return primal_normal_ec

def compute_geofac_grg(
    geofac_grg: np.array,
    primal_normal_ec: np.array,
    geofac_div: np.array,
    c_lin_e: np.array,
    C2E_: np.array,
    E2C_: np.array,
    C2E2C_: np.array,
    lateral_boundary: np.array,
) -> np.array:
    """
    Args:
        dual_edge_length:
        geofac_div:
    """
    llb = lateral_boundary[0]
    index = np.transpose(np.vstack((np.arange(lateral_boundary[1]), np.arange(lateral_boundary[1]), np.arange(lateral_boundary[1]))))
    for k in range(2):
        fac = E2C_[C2E_, k] == index
        for i in range(2):
            for j in range(3):
                geofac_grg[llb:, 0, i] = geofac_grg[llb:, 0, i] + fac[llb:, j] * (primal_normal_ec[:, :, i]*geofac_div*c_lin_e[C2E_, k])[llb:, j]
    for k in range(2):
        fac = E2C_[C2E_, k] == C2E2C_
        for i in range(2):
            for j in range(3):
                geofac_grg[llb:, 1 + j, i] = geofac_grg[llb:, 1 + j, i] + fac[llb:, j] * (primal_normal_ec[:, :, i]*geofac_div*c_lin_e[C2E_, k])[llb:, j]
    return geofac_grg

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
