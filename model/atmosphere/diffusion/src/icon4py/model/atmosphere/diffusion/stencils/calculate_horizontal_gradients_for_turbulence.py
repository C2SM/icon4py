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
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import C2E2CO, C2E2CODim, CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_horizontal_gradients_for_turbulence(
    w: fa.CKwpField,
    geofac_grg_x: Field[[CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: Field[[CellDim, C2E2CODim], wpfloat],
) -> tuple[fa.CKvpField, fa.CKvpField]:
    dwdx_wp = neighbor_sum(geofac_grg_x * w(C2E2CO), axis=C2E2CODim)
    dwdy_wp = neighbor_sum(geofac_grg_y * w(C2E2CO), axis=C2E2CODim)
    return astype((dwdx_wp, dwdy_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_horizontal_gradients_for_turbulence(
    w: fa.CKwpField,
    geofac_grg_x: Field[[CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: Field[[CellDim, C2E2CODim], wpfloat],
    dwdx: fa.CKvpField,
    dwdy: fa.CKvpField,
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
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
