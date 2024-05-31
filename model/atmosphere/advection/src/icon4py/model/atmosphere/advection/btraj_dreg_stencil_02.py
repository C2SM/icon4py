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
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, sqrt, where
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import E2EC, ECDim, EdgeDim, KDim


@field_operator
def _btraj_dreg_stencil_02(
    p_vn: fa.EKfloatField,
    p_vt: fa.EKfloatField,
    edge_cell_length: Field[[ECDim], float],
    p_dt: float,
) -> fa.EKintField:
    lvn_pos = where(p_vn >= 0.0, True, False)
    traj_length = sqrt(p_vn * p_vn + p_vt * p_vt) * p_dt
    e2c_length = where(lvn_pos, edge_cell_length(E2EC[0]), edge_cell_length(E2EC[1]))
    opt_famask_dsl = where(traj_length > 1.25 * broadcast(e2c_length, (EdgeDim, KDim)), 1, 0)

    return opt_famask_dsl


@program(grid_type=GridType.UNSTRUCTURED)
def btraj_dreg_stencil_02(
    p_vn: fa.EKfloatField,
    p_vt: fa.EKfloatField,
    edge_cell_length: Field[[ECDim], float],
    p_dt: float,
    opt_famask_dsl: fa.EKintField,
):
    _btraj_dreg_stencil_02(p_vn, p_vt, edge_cell_length, p_dt, out=opt_famask_dsl)
