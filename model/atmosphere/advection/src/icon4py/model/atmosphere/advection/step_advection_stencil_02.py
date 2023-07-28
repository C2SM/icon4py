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

from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import maximum

from icon4py.model.common.dimension import CellDim, KDim, Koff


@field_operator
def _step_advection_stencil_02(
    p_rhodz_new: Field[[CellDim, KDim], float],
    p_mflx_contra_v: Field[[CellDim, KDim], float],
    deepatmo_divzl: Field[[KDim], float],
    deepatmo_divzu: Field[[KDim], float],
    p_dtime: float,
) -> Field[[CellDim, KDim], float]:
    return maximum(0.1 * p_rhodz_new, p_rhodz_new) - p_dtime * (
        p_mflx_contra_v(Koff[1]) * deepatmo_divzl - p_mflx_contra_v * deepatmo_divzu
    )


@program
def step_advection_stencil_02(
    p_rhodz_new: Field[[CellDim, KDim], float],
    p_mflx_contra_v: Field[[CellDim, KDim], float],
    deepatmo_divzl: Field[[KDim], float],
    deepatmo_divzu: Field[[KDim], float],
    p_dtime: float,
    rhodz_ast2: Field[[CellDim, KDim], float],
):
    _step_advection_stencil_02(
        p_rhodz_new,
        p_mflx_contra_v,
        deepatmo_divzl,
        deepatmo_divzu,
        p_dtime,
        out=rhodz_ast2,
    )
