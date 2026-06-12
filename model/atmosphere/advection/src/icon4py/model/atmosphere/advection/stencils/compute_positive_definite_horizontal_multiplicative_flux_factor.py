# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import maximum, minimum, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_positive_definite_horizontal_multiplicative_flux_factor(
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    p_cc: fa.CellKField[wpfloat],
    p_rhodz_now: fa.CellKField[wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[wpfloat],
    p_dtime: wpfloat,
    wp_eps: wpfloat,
) -> fa.CellKField[wpfloat]:
    p_m = neighbor_sum(
        maximum(wpfloat(0.0), p_mflx_tracer_h(C2E) * geofac_div * p_dtime), axis=C2EDim
    )
    r_m = minimum(wpfloat(1.0), (p_cc * p_rhodz_now) / (p_m + wp_eps))
    return r_m


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_positive_definite_horizontal_multiplicative_flux_factor(
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    p_cc: fa.CellKField[wpfloat],
    p_rhodz_now: fa.CellKField[wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[wpfloat],
    r_m: fa.CellKField[wpfloat],
    p_dtime: wpfloat,
    wp_eps: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_positive_definite_horizontal_multiplicative_flux_factor(
        geofac_div=geofac_div,
        p_cc=p_cc,
        p_rhodz_now=p_rhodz_now,
        p_mflx_tracer_h=p_mflx_tracer_h,
        p_dtime=p_dtime,
        wp_eps=wp_eps,
        out=r_m,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
