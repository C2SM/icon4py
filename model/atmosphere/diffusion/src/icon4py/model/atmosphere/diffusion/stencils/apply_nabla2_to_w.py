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
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common.dimension import C2E2CO, C2E2CODim, CellDim, KDim


@field_operator
def _apply_nabla2_to_w(
    area: Field[[CellDim], float],
    z_nabla2_c: Field[[CellDim, KDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    w: Field[[CellDim, KDim], float],
    diff_multfac_w: float,
) -> Field[[CellDim, KDim], float]:
    w = w - diff_multfac_w * (area * area) * neighbor_sum(
        z_nabla2_c(C2E2CO) * geofac_n2s, axis=C2E2CODim
    )
    return w


@program(grid_type=GridType.UNSTRUCTURED)
def apply_nabla2_to_w(
    area: Field[[CellDim], float],
    z_nabla2_c: Field[[CellDim, KDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    w: Field[[CellDim, KDim], float],
    diff_multfac_w: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_nabla2_to_w(
        area,
        z_nabla2_c,
        geofac_n2s,
        w,
        diff_multfac_w,
        out=w,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
