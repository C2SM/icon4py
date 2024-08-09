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
from gt4py.next.ffront.fbuiltins import broadcast, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _apply_rayleigh_damping_mechanism(
    z_raylfac: fa.KField[wpfloat],
    w_1: fa.CellField[wpfloat],
    w: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_54."""
    z_raylfac = broadcast(z_raylfac, (dims.CellDim, dims.KDim))
    w_wp = z_raylfac * w + (wpfloat("1.0") - z_raylfac) * w_1
    return w_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_rayleigh_damping_mechanism(
    z_raylfac: fa.KField[wpfloat],
    w_1: fa.CellField[wpfloat],
    w: fa.CellKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_rayleigh_damping_mechanism(
        z_raylfac,
        w_1,
        w,
        out=w,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
