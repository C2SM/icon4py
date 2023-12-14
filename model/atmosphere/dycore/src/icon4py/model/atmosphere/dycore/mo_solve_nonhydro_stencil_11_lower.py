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
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _mo_solve_nonhydro_stencil_11_lower() -> Field[[CellDim, KDim], vpfloat]:
    z_theta_v_pr_ic_vp = broadcast(vpfloat("0.0"), (CellDim, KDim))

    return z_theta_v_pr_ic_vp


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_11_lower(
    z_theta_v_pr_ic: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_11_lower(
        out=z_theta_v_pr_ic,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
