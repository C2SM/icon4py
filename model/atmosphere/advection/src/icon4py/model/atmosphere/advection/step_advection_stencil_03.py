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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, maximum

from icon4py.model.common.dimension import CellDim, KDim


@field_operator
def _step_advection_stencil_03(
    p_tracer_now: Field[[CellDim, KDim], float],
    p_grf_tend_tracer: Field[[CellDim, KDim], float],
    p_dtime: float,
) -> Field[[CellDim, KDim], float]:
    p_tracer_new = maximum(0.0, p_tracer_now + p_dtime * p_grf_tend_tracer)
    return p_tracer_new


@program
def step_advection_stencil_03(
    p_tracer_now: Field[[CellDim, KDim], float],
    p_grf_tend_tracer: Field[[CellDim, KDim], float],
    p_tracer_new: Field[[CellDim, KDim], float],
    p_dtime: float,
):
    _step_advection_stencil_03(p_tracer_now, p_grf_tend_tracer, p_dtime, out=p_tracer_new)
