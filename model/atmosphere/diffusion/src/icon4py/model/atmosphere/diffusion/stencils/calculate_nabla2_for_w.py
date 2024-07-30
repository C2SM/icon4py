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
def _calculate_nabla2_for_w(
    w: fa.CellKHalfField[wpfloat], geofac_n2s: gtx.Field[[CellDim, C2E2CODim], wpfloat]
) -> fa.CellKHalfField[vpfloat]:
    z_nabla2_c_wp = neighbor_sum(w(C2E2CO) * geofac_n2s, axis=C2E2CODim)
    return astype(z_nabla2_c_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_nabla2_for_w(
    w: gtx.Field[[CellDim, KHalfDim], wpfloat],
    geofac_n2s: gtx.Field[[CellDim, C2E2CODim], wpfloat],
    z_nabla2_c: fa.CellKHalfField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_nabla2_for_w(
        w,
        geofac_n2s,
        out=z_nabla2_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KHalfDim: (vertical_start, vertical_end + 1),
        },
    )
