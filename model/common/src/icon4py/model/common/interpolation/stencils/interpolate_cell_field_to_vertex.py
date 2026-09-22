# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import neighbor_sum

import icon4py.model.common.type_alias as types
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import V2C


@gtx.field_operator
def _interpolate_cell_field_to_vertex(
    cell_in: fa.CellKHalfField[gtx.float64],
    c_int: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], gtx.float64],
) -> fa.VertexKHalfField[gtx.float64]:
    vert_out = neighbor_sum(c_int * cell_in(V2C), axis=dims.V2CDim)
    return vert_out


@gtx.field_operator
def _interpolate_cell_field_to_vertex_wp(
    cell_in: fa.CellKHalfField[types.wpfloat],
    c_int: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], types.wpfloat],
) -> fa.VertexKHalfField[types.wpfloat]:
    vert_out = neighbor_sum(c_int * cell_in(V2C), axis=dims.V2CDim)
    return vert_out
