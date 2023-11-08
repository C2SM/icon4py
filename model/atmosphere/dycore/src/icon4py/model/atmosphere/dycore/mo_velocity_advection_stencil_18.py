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
from gt4py.next.ffront.fbuiltins import (  # noqa: A004 # import gt4py builtin
    Field,
    abs,
    int32,
    minimum,
    neighbor_sum,
    where,
)

from icon4py.model.common.dimension import C2E2CO, C2E2CODim, CellDim, KDim


@field_operator
def _mo_velocity_advection_stencil_18(
    levmask: Field[[KDim], bool],
    cfl_clipping: Field[[CellDim, KDim], bool],
    owner_mask: Field[[CellDim], bool],
    z_w_con_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    area: Field[[CellDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    w: Field[[CellDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
    scalfac_exdiff: float,
    cfl_w_limit: float,
    dtime: float,
) -> Field[[CellDim, KDim], float]:
    difcoef = where(
        levmask & cfl_clipping & owner_mask,
        scalfac_exdiff
        * minimum(
            0.85 - cfl_w_limit * dtime,
            abs(z_w_con_c) * dtime / ddqz_z_half - cfl_w_limit * dtime,
        ),
        0.0,
    )

    ddt_w_adv = where(
        levmask & cfl_clipping & owner_mask,
        ddt_w_adv + difcoef * area * neighbor_sum(w(C2E2CO) * geofac_n2s, axis=C2E2CODim),
        ddt_w_adv,
    )

    return ddt_w_adv


@program(grid_type=GridType.UNSTRUCTURED)
def mo_velocity_advection_stencil_18(
    levmask: Field[[KDim], bool],
    cfl_clipping: Field[[CellDim, KDim], bool],
    owner_mask: Field[[CellDim], bool],
    z_w_con_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    area: Field[[CellDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    w: Field[[CellDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
    scalfac_exdiff: float,
    cfl_w_limit: float,
    dtime: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_velocity_advection_stencil_18(
        levmask,
        cfl_clipping,
        owner_mask,
        z_w_con_c,
        ddqz_z_half,
        area,
        geofac_n2s,
        w,
        ddt_w_adv,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        out=ddt_w_adv,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
