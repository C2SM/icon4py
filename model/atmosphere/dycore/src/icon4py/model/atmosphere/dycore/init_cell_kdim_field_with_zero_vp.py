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

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.model_backend import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _init_cell_kdim_field_with_zero_vp() -> Field[[CellDim, KDim], vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_03, _mo_solve_nonhydro_stencil_11_lower, _mo_solve_nonhydro_stencil_45, _mo_solve_nonhydro_stencil_45_b, or _mo_velocity_advection_stencil_12."""
    return broadcast(vpfloat("0.0"), (CellDim, KDim))


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def init_cell_kdim_field_with_zero_vp(
    field_with_zero_vp: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _init_cell_kdim_field_with_zero_vp(
        out=field_with_zero_vp,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
