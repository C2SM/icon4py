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

from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _init_two_cell_kdim_fields_with_zero_wp() -> (
    tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]
):
    """Formerly known as _mo_solve_nonhydro_stencil_46."""
    return _init_cell_kdim_field_with_zero_wp(), _init_cell_kdim_field_with_zero_wp()


@program(grid_type=GridType.UNSTRUCTURED)
def init_two_cell_kdim_fields_with_zero_wp(
    cell_kdim_field_with_zero_wp_1: fa.CellKField[wpfloat],
    cell_kdim_field_with_zero_wp_2: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _init_two_cell_kdim_fields_with_zero_wp(
        out=(cell_kdim_field_with_zero_wp_1, cell_kdim_field_with_zero_wp_2),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
