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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.atmosphere.dycore.set_cell_kdim_field_to_zero_vp import (
    _set_cell_kdim_field_to_zero_vp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.model_backend import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _set_two_cell_kdim_fields_to_zero_vp() -> (
    tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], vpfloat]]
):
    """Formerly known as _mo_solve_nonhydro_stencil_01."""
    return _set_cell_kdim_field_to_zero_vp(), _set_cell_kdim_field_to_zero_vp()


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def set_two_cell_kdim_fields_to_zero_vp(
    cell_kdim_field_to_zero_vp_1: Field[[CellDim, KDim], vpfloat],
    cell_kdim_field_to_zero_vp_2: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_two_cell_kdim_fields_to_zero_vp(
        out=(cell_kdim_field_to_zero_vp_1, cell_kdim_field_to_zero_vp_2),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
