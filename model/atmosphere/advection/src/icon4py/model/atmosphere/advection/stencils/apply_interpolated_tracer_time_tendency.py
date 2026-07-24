# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import maximum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _apply_interpolated_tracer_time_tendency(
    p_tracer_now: fa.CellKField[wpfloat],
    p_grf_tend_tracer: fa.CellKField[wpfloat],
    p_dtime: wpfloat,
) -> fa.CellKField[wpfloat]:
    p_tracer_new = maximum(wpfloat(0.0), p_tracer_now + p_dtime * p_grf_tend_tracer)
    return p_tracer_new


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_interpolated_tracer_time_tendency(
    p_tracer_now: fa.CellKField[wpfloat],
    p_grf_tend_tracer: fa.CellKField[wpfloat],
    p_tracer_new: fa.CellKField[wpfloat],
    p_dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_interpolated_tracer_time_tendency(
        p_tracer_now=p_tracer_now,
        p_grf_tend_tracer=p_grf_tend_tracer,
        p_dtime=p_dtime,
        out=p_tracer_new,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
