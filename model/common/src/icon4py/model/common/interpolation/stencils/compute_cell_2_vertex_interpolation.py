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

import gt4py.next as gt
from gt4py.next import neighbor_sum

import icon4py.model.common.settings as settings
import icon4py.model.common.type_alias as types
from icon4py.model.common.dimension import V2C, CellDim, KDim, V2CDim, VertexDim


@gt.field_operator
def _compute_cell_2_vertex_interpolation(
    cell_in: gt.Field[[CellDim, KDim], types.wpfloat],
    c_int: gt.Field[[VertexDim, V2CDim], types.wpfloat],
) -> gt.Field[[VertexDim, KDim], types.wpfloat]:
    vert_out = neighbor_sum(c_int * cell_in(V2C), axis=V2CDim)
    return vert_out


@gt.program(grid_type=gt.GridType.UNSTRUCTURED, backend=settings.backend)
def compute_cell_2_vertex_interpolation(
    cell_in: gt.Field[[CellDim, KDim], types.wpfloat],
    c_int: gt.Field[[VertexDim, V2CDim], types.wpfloat],
    vert_out: gt.Field[[VertexDim, KDim], types.wpfloat],
    horizontal_start: gt.int32,
    horizontal_end: gt.int32,
    vertical_start: gt.int32,
    vertical_end: gt.int32,
):
    _compute_cell_2_vertex_interpolation(
        cell_in,
        c_int,
        out=vert_out,
        domain={
            VertexDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
