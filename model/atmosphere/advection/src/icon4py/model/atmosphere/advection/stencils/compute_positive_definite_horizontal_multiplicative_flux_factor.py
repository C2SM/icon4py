# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import broadcast, maximum, minimum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2CE, C2E


@gtx.field_operator
def _compute_positive_definite_horizontal_multiplicative_flux_factor(
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    p_rhodz_now: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    dbl_eps: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    zero = broadcast(0.0, (dims.CellDim, dims.KDim))

    pm_0 = maximum(zero, p_mflx_tracer_h(C2E[0]) * geofac_div(C2CE[0]) * p_dtime)
    pm_1 = maximum(zero, p_mflx_tracer_h(C2E[1]) * geofac_div(C2CE[1]) * p_dtime)
    pm_2 = maximum(zero, p_mflx_tracer_h(C2E[2]) * geofac_div(C2CE[2]) * p_dtime)
    p_m = pm_0 + pm_1 + pm_2
    r_m = minimum(
        broadcast(1.0, (dims.CellDim, dims.KDim)),
        (p_cc * p_rhodz_now) / (p_m + dbl_eps),
    )

    return r_m


@gtx.program
def compute_positive_definite_horizontal_multiplicative_flux_factor(
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    p_rhodz_now: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    dbl_eps: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_positive_definite_horizontal_multiplicative_flux_factor(
        geofac_div,
        p_cc,
        p_rhodz_now,
        p_mflx_tracer_h,
        p_dtime,
        dbl_eps,
        out=r_m,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
