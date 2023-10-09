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
from gt4py.next.ffront.fbuiltins import Field, where, int32, broadcast

from icon4py.model.common.dimension import CellDim, KDim

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_11 import (
    _mo_velocity_advection_stencil_11,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_12 import (
    _mo_velocity_advection_stencil_12,
)


@field_operator
def _fused_mo_velocity_advection_11_12(
    w: Field[[CellDim, KDim], float],
    k: Field[[KDim], int32],
    nlev: int32,
) -> Field[[CellDim, KDim], float]:
    z_w_con_c = where(
        k < nlev,
        _mo_velocity_advection_stencil_11(w),
        _mo_velocity_advection_stencil_12(),
    )

    return z_w_con_c


@program(grid_type=GridType.UNSTRUCTURED)
def fused_mo_velocity_advection_11_12(
    w: Field[[CellDim, KDim], float],
    z_w_con_c: Field[[CellDim, KDim], float],
    k: Field[[KDim], int32],
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _fused_mo_velocity_advection_11_12(
        w,
        k,
        nlev,
        out=z_w_con_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end - 1),
        },
    )
    _fused_mo_velocity_advection_11_12(
        w,
        k,
        nlev,
        out=z_w_con_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_end - 1, vertical_end),
        },
    )
