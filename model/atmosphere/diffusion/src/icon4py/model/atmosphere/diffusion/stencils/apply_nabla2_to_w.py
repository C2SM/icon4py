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

import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, neighbor_sum

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2E2CO, C2E2CODim, CellDim, KHalfDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_nabla2_to_w(
    area: fa.CellField[wpfloat],
    z_nabla2_c: gtx.Field[[CellDim, KHalfDim], vpfloat],
    geofac_n2s: gtx.Field[[CellDim, C2E2CODim], wpfloat],
    w: gtx.Field[[CellDim, KHalfDim], wpfloat],
    diff_multfac_w: wpfloat,
) -> gtx.Field[[CellDim, KHalfDim], wpfloat]:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)

    w_wp = w - diff_multfac_w * (area * area) * neighbor_sum(
        z_nabla2_c_wp(C2E2CO) * geofac_n2s, axis=C2E2CODim
    )
    return w_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_nabla2_to_w(
    area: fa.CellField[wpfloat],
    z_nabla2_c: gtx.Field[[CellDim, KHalfDim], vpfloat],
    geofac_n2s: gtx.Field[[CellDim, C2E2CODim], wpfloat],
    w: gtx.Field[[CellDim, KHalfDim], wpfloat],
    diff_multfac_w: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            KHalfDim: (vertical_start, vertical_end + 1),
        },
    )
