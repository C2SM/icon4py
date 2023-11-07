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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    Field,
    abs, # noqa: A004 abs is shadowing a Python builtin
    broadcast,
    where
)

from icon4py.model.common.dimension import CellDim, KDim


@field_operator
def _mo_velocity_advection_stencil_14(
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_w_con_c: Field[[CellDim, KDim], float],
    cfl_w_limit: float,
    dtime: float,
) -> tuple[
    Field[[CellDim, KDim], bool],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    cfl_clipping = where(
        abs(z_w_con_c) > cfl_w_limit * ddqz_z_half,
        broadcast(True, (CellDim, KDim)),
        False,
    )

    vcfl = where(cfl_clipping, z_w_con_c * dtime / ddqz_z_half, 0.0)

    z_w_con_c = where((cfl_clipping) & (vcfl < -0.85), -0.85 * ddqz_z_half / dtime, z_w_con_c)

    z_w_con_c = where((cfl_clipping) & (vcfl > 0.85), 0.85 * ddqz_z_half / dtime, z_w_con_c)

    return cfl_clipping, vcfl, z_w_con_c


@program(grid_type=GridType.UNSTRUCTURED)
def mo_velocity_advection_stencil_14(
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_w_con_c: Field[[CellDim, KDim], float],
    cfl_clipping: Field[[CellDim, KDim], bool],
    vcfl: Field[[CellDim, KDim], float],
    cfl_w_limit: float,
    dtime: float,
):
    _mo_velocity_advection_stencil_14(
        ddqz_z_half,
        z_w_con_c,
        cfl_w_limit,
        dtime,
        out=(cfl_clipping, vcfl, z_w_con_c),
    )
