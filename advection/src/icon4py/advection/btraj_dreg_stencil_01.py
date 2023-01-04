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

from functional.ffront.fbuiltins import Field, int32, where, broadcast
from functional.ffront.decorator import field_operator, program

from icon4py.common.dimension import EdgeDim, KDim
@field_operator
def _btraj_dreg_stencil_01(
    lcounterclock: bool,
    p_vn: Field[[EdgeDim, KDim], float],
    tangent_orientation: Field[[EdgeDim], float],
) -> Field[[EdgeDim, KDim], bool]:

    tangent_orientation = broadcast(tangent_orientation, (EdgeDim, KDim))
    lvn_sys_pos_true = where(p_vn*tangent_orientation >= 0.0, True, False)
    mask_lcounterclock = broadcast(lcounterclock, (EdgeDim, KDim))
    lvn_sys_pos = where(mask_lcounterclock, lvn_sys_pos_true, False) 
    return lvn_sys_pos


@program
def btraj_dreg_stencil_01(
    lcounterclock: bool,
    p_vn: Field[[EdgeDim, KDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    lvn_sys_pos: Field[[EdgeDim, KDim], bool],
):
    _btraj_dreg_stencil_01(lcounterclock, p_vn, tangent_orientation, out=lvn_sys_pos)

