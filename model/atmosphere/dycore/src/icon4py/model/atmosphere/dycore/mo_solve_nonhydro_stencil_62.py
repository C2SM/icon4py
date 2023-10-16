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
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _mo_solve_nonhydro_stencil_62(
    w_now: Field[[CellDim, KDim], wpfloat],
    grf_tend_w: Field[[CellDim, KDim], wpfloat],
    dtime: wpfloat,
) -> Field[[CellDim, KDim], wpfloat]:
    w_new_wp = w_now + dtime * grf_tend_w
    return w_new_wp


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_62(
    w_now: Field[[CellDim, KDim], wpfloat],
    grf_tend_w: Field[[CellDim, KDim], wpfloat],
    w_new: Field[[CellDim, KDim], wpfloat],
    dtime: wpfloat,
):
    _mo_solve_nonhydro_stencil_62(w_now, grf_tend_w, dtime, out=w_new)
