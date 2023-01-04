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

from functional.ffront.fbuiltins import Field,  where, sqrt, broadcast, int32
from functional.ffront.decorator import field_operator, program

from icon4py.common.dimension import E2EC, ECDim, EdgeDim, KDim


@field_operator
def _btraj_dreg_stencil_02(
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    edge_cell_length: Field[[ECDim], float],
    p_dt: float,
) -> Field[[EdgeDim, KDim], int32]:

    lvn_pos = where(p_vn >= 0.0, True, False)
    traj_length = sqrt(p_vn**2 + p_vt**2) * p_dt
    e2c_length = where(lvn_pos, edge_cell_length(E2EC[0]), edge_cell_length(E2EC[1]))
    famask = where(traj_length > 1.25*broadcast(e2c_length, (EdgeDim, KDim)), int32(1), int32(0))
    
    return famask


@program
def btraj_dreg_stencil_02(
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    edge_cell_length: Field[[ECDim], float],
    famask: Field[[EdgeDim, KDim], int32],
    p_dt: float,

):
    _btraj_dreg_stencil_02(
        p_vn,
        p_vt,
        edge_cell_length,
        p_dt,
        out=famask)
