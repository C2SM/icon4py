# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32, where

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C, EdgeDim, KDim


@field_operator
def _hflx_limiter_pd_stencil_02(
    r_m: fa.CellKField[float],
    p_mflx_tracer_h: fa.EdgeKField[float],
) -> fa.EdgeKField[float]:
    p_mflx_tracer_h_out = where(
        p_mflx_tracer_h >= 0.0,
        p_mflx_tracer_h * r_m(E2C[0]),
        p_mflx_tracer_h * r_m(E2C[1]),
    )
    return p_mflx_tracer_h_out


@program
def hflx_limiter_pd_stencil_02(
    r_m: fa.CellKField[float],
    p_mflx_tracer_h: fa.EdgeKField[float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _hflx_limiter_pd_stencil_02(
        r_m,
        p_mflx_tracer_h,
        out=p_mflx_tracer_h,
        domain={EdgeDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
