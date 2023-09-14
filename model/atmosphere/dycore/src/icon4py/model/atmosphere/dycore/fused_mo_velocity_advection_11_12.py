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
from gt4py.next.ffront.fbuiltins import Field, where

from icon4py.model.common.dimension import CellDim, KDim

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_11 import _mo_velocity_advection_stencil_11
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_12 import _mo_velocity_advection_stencil_12


@field_operator
def _fused_mo_velocity_advection_stencil_11_12(
    w: Field[[CellDim, KDim], float],
    k: Field[[KDim], int],
    nlev: int,
):
    z_w_con_c = where(
        k < nlev,
        _mo_velocity_advection_stencil_11(w),
        _mo_velocity_advection_stencil_12(),
    )

    return z_w_con_c

@program(grid_type=GridType.UNSTRUCTURED)
def fused_mo_velocity_advection_stencil_11_12(
    w: Field[[CellDim, KDim], float],
    z_w_con_c: Field[[CellDim, KDim], float],
    vertical_lower: int,
    vertical_upper: int,
    horizontal_lower: int,
    horizontal_upper: int,
):
    _fused_mo_velocity_advection_stencil_11_12(w, out=z_w_con_c, domain={KDim: (0, vertical_upper - 1)})
    _fused_mo_velocity_advection_stencil_11_12(w, out=z_w_con_c, domain={KDim: (vertical_upper - 1, vertical_upper)})
