# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_vp import (
    _init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _init_two_cell_kdim_fields_with_zero_vp() -> (
    tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]
):
    """Formerly known as _mo_solve_nonhydro_stencil_01."""
    return _init_cell_kdim_field_with_zero_vp(), _init_cell_kdim_field_with_zero_vp()


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
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
