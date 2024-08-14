# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_vp import (
    _init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _init_two_cell_kdim_fields_with_zero_vp() -> (
    tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]
):
    """Formerly known as _mo_solve_nonhydro_stencil_01."""
    return _init_cell_kdim_field_with_zero_vp(), _init_cell_kdim_field_with_zero_vp()


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def init_two_cell_kdim_fields_with_zero_vp(
    cell_kdim_field_with_zero_vp_1: fa.CellKField[vpfloat],
    cell_kdim_field_with_zero_vp_2: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _init_two_cell_kdim_fields_with_zero_vp(
        out=(cell_kdim_field_with_zero_vp_1, cell_kdim_field_with_zero_vp_2),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
