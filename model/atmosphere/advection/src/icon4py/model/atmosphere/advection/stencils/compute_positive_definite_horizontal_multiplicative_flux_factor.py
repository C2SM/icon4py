# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, maximum, minimum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, CellDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_positive_definite_horizontal_multiplicative_flux_factor(
    geofac_div: Field[[dims.CEDim], wpfloat],
    p_cc: fa.CellKField[wpfloat],
    p_rhodz_now: fa.CellKField[wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[wpfloat],
    p_dtime: wpfloat,
    dbl_eps: wpfloat,
) -> fa.CellKField[wpfloat]:
    zero = broadcast(wpfloat(0.0), (CellDim, KDim))

    pm_0 = maximum(zero, p_mflx_tracer_h(C2E[0]) * geofac_div(C2CE[0]) * p_dtime)
    pm_1 = maximum(zero, p_mflx_tracer_h(C2E[1]) * geofac_div(C2CE[1]) * p_dtime)
    pm_2 = maximum(zero, p_mflx_tracer_h(C2E[2]) * geofac_div(C2CE[2]) * p_dtime)
    p_m = pm_0 + pm_1 + pm_2
    r_m = minimum(broadcast(wpfloat(1.0), (CellDim, KDim)), (p_cc * p_rhodz_now) / (p_m + dbl_eps))

    return r_m


@program
def compute_positive_definite_horizontal_multiplicative_flux_factor(
    geofac_div: Field[[dims.CEDim], wpfloat],
    p_cc: fa.CellKField[wpfloat],
    p_rhodz_now: fa.CellKField[wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[wpfloat],
    r_m: fa.CellKField[wpfloat],
    p_dtime: wpfloat,
    dbl_eps: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_positive_definite_horizontal_multiplicative_flux_factor(
        geofac_div,
        p_cc,
        p_rhodz_now,
        p_mflx_tracer_h,
        p_dtime,
        dbl_eps,
        out=r_m,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
