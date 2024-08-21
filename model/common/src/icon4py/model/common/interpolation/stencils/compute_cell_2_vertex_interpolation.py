# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import neighbor_sum

import icon4py.model.common.settings as settings
import icon4py.model.common.type_alias as types
from icon4py.model.common import dimension as dims
from icon4py.model.common.dimension import V2C, V2CDim


# TODO: this will have to be removed once domain allows for imports
VertexDim = dims.VertexDim
KDim = dims.KDim


@gtx.field_operator
def _compute_cell_2_vertex_interpolation(
    cell_in: gtx.Field[[dims.CellDim, dims.KDim], types.wpfloat],
    c_int: gtx.Field[[dims.VertexDim, V2CDim], types.wpfloat],
) -> gtx.Field[[dims.VertexDim, dims.KDim], types.wpfloat]:
    vert_out = neighbor_sum(c_int * cell_in(V2C), axis=V2CDim)
    return vert_out


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=settings.backend)
def compute_cell_2_vertex_interpolation(
    cell_in: gtx.Field[[dims.CellDim, dims.KDim], types.wpfloat],
    c_int: gtx.Field[[dims.VertexDim, V2CDim], types.wpfloat],
    vert_out: gtx.Field[[dims.VertexDim, dims.KDim], types.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute the interpolation from cell to vertex field.

    Args:
        cell_in: input cell field
        c_int: interpolation coefficients
        vert_out: (output) vertex field
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_cell_2_vertex_interpolation(
        cell_in,
        c_int,
        out=vert_out,
        domain={
            VertexDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
