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
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32

from icon4py.model.atmosphere.dycore.set_cell_kdim_field_to_zero_wp import (
    _set_cell_kdim_field_to_zero_wp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _mo_solve_nonhydro_stencil_46() -> (
    tuple[Field[[CellDim, KDim], wpfloat], Field[[CellDim, KDim], wpfloat]]
):
    '''formerly known as _mo_solve_nonhydro_stencil_46.'''
    return _set_cell_kdim_field_to_zero_wp(), _set_cell_kdim_field_to_zero_wp()


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_46(
    w_nnew: Field[[CellDim, KDim], wpfloat],
    z_contr_w_fl_l: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_46(
        out=(w_nnew, z_contr_w_fl_l),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
