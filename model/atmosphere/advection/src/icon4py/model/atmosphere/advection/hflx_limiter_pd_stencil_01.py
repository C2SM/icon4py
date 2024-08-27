# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, maximum, minimum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, CellDim, KDim


@field_operator
def _hflx_limiter_pd_stencil_01(
    geofac_div: Field[[dims.CEDim], float],
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
    geofac_div: Field[[dims.CEDim], float],
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
