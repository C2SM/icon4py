# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C, E2CDim


@field_operator
def _cell_2_edge_interpolation(
    in_field: fa.CellKField[ta.wpfloat],
    coeff: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    """
    Interpolate a Cell Field to Edges.

    There is a special handling of lateral boundary edges in `subroutine cells2edges_scalar`
    in mo_icon_interpolation.f90 where the value is set to the one valid in_field value without
    multiplication by coeff. This essentially means: the skip value neighbor in the neighbor_sum
    is skipped and coeff needs to be 1 for this Edge index.
    """
    return neighbor_sum(in_field(E2C) * coeff, axis=E2CDim)


@program(grid_type=GridType.UNSTRUCTURED)
def cell_2_edge_interpolation(
    in_field: fa.CellKField[ta.wpfloat],
    coeff: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    out_field: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _cell_2_edge_interpolation(
        in_field,
        coeff,
        out=out_field,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
