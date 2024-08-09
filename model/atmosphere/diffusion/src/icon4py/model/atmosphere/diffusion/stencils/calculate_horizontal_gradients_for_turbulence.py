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
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2CO, C2E2CODim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_horizontal_gradients_for_turbulence(
    w: fa.CellKField[wpfloat],
    geofac_grg_x: Field[[dims.CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: Field[[dims.CellDim, C2E2CODim], wpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    dwdx_wp = neighbor_sum(geofac_grg_x * w(C2E2CO), axis=C2E2CODim)
    dwdy_wp = neighbor_sum(geofac_grg_y * w(C2E2CO), axis=C2E2CODim)
    return astype((dwdx_wp, dwdy_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_horizontal_gradients_for_turbulence(
    w: fa.CellKField[wpfloat],
    geofac_grg_x: Field[[dims.CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: Field[[dims.CellDim, C2E2CODim], wpfloat],
    dwdx: fa.CellKField[vpfloat],
    dwdy: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _calculate_horizontal_gradients_for_turbulence(
        w,
        geofac_grg_x,
        geofac_grg_y,
        out=(dwdx, dwdy),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
