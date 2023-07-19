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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, where

from icon4py.atm_dyn_iconam.calculate_horizontal_gradients_for_turbulence import (
    _calculate_horizontal_gradients_for_turbulence,
)
from icon4py.atm_dyn_iconam.calculate_nabla2_for_w import _calculate_nabla2_for_w
from icon4py.common.dimension import C2E2CODim, CellDim, KDim


@field_operator
def _calculate_nabla2_for_w_and_turbulence_quantities(
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    w: Field[[CellDim, KDim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
    vert_idx: Field[[KDim], int32],
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:

    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    dwdx, dwdy = where(
        int32(0) < vert_idx,
        _calculate_horizontal_gradients_for_turbulence(
            w, geofac_grg_x, geofac_grg_y
        ),
        (dwdx, dwdy),
    )

    z_nabla2_c = _calculate_nabla2_for_w(w, geofac_n2s)

    return z_nabla2_c, dwdx, dwdy


@program
def calculate_nabla2_for_w_and_turbulence_quantities(
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    w: Field[[CellDim, KDim], float],
    z_nabla2_c: Field[[CellDim, KDim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
    vert_idx: Field[[KDim], int32],
):
    _calculate_nabla2_for_w_and_turbulence_quantities(
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        w,
        dwdx,
        dwdy,
        vert_idx,
        out=(z_nabla2_c, dwdx, dwdy),
    )
