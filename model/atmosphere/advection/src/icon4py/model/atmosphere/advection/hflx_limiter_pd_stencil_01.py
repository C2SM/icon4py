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
from gt4py.next.ffront.fbuiltins import Field, broadcast, maximum, minimum

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, CEDim, CellDim, KDim


@field_operator
def _hflx_limiter_pd_stencil_01(
    geofac_div: Field[[CEDim], float],
    p_cc: fa.CellKField[float],
    p_rhodz_now: fa.CellKField[float],
    p_mflx_tracer_h: fa.EdgeKField[float],
    p_dtime: float,
    dbl_eps: float,
) -> fa.CellKField[float]:
    zero = broadcast(0.0, (CellDim, KDim))

    pm_0 = maximum(zero, p_mflx_tracer_h(C2E[0]) * geofac_div(C2CE[0]) * p_dtime)
    pm_1 = maximum(zero, p_mflx_tracer_h(C2E[1]) * geofac_div(C2CE[1]) * p_dtime)
    pm_2 = maximum(zero, p_mflx_tracer_h(C2E[2]) * geofac_div(C2CE[2]) * p_dtime)
    p_m = pm_0 + pm_1 + pm_2
    r_m = minimum(broadcast(1.0, (CellDim, KDim)), (p_cc * p_rhodz_now) / (p_m + dbl_eps))

    return r_m


@program
def hflx_limiter_pd_stencil_01(
    geofac_div: Field[[CEDim], float],
    p_cc: fa.CellKField[float],
    p_rhodz_now: fa.CellKField[float],
    p_mflx_tracer_h: fa.EdgeKField[float],
    r_m: fa.CellKField[float],
    p_dtime: float,
    dbl_eps: float,
):
    _hflx_limiter_pd_stencil_01(
        geofac_div,
        p_cc,
        p_rhodz_now,
        p_mflx_tracer_h,
        p_dtime,
        dbl_eps,
        out=r_m,
    )
