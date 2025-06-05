# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import program
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _compute_first_vertical_derivative_at_cells(
    cell_kdim_field: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """
    This stencil computes the first vertical at cells
    """
    first_vertical_derivative = (cell_kdim_field - cell_kdim_field(Koff[1])) * inv_ddqz_z_full
    return first_vertical_derivative


@program(grid_type=GridType.UNSTRUCTURED)
def compute_first_vertical_derivative_at_cells(
    cell_kdim_field: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    first_vertical_derivative: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_first_vertical_derivative_at_cells(
        cell_kdim_field,
        inv_ddqz_z_full,
        out=first_vertical_derivative,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
