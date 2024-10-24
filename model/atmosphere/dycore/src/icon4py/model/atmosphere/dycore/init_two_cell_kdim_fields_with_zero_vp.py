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

from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_vp import (
    _init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _init_two_cell_kdim_fields_with_zero_vp() -> (
    tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]
):
    """Formerly known as _mo_solve_nonhydro_stencil_01."""
    return _init_cell_kdim_field_with_zero_vp(), _init_cell_kdim_field_with_zero_vp()


@program(grid_type=GridType.UNSTRUCTURED)
def init_two_cell_kdim_fields_with_zero_vp(
    cell_kdim_field_with_zero_vp_1: fa.CellKField[vpfloat],
    cell_kdim_field_with_zero_vp_2: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _init_two_cell_kdim_fields_with_zero_vp(
        out=(cell_kdim_field_with_zero_vp_1, cell_kdim_field_with_zero_vp_2),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
