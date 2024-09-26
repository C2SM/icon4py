# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import maximum, minimum, where

from icon4py.model.common import field_type_aliases as fa


@field_operator
def _hflx_limiter_mo_stencil_02(
    refin_ctrl: fa.CellField[gtx.int32],
    p_cc: fa.CellKField[float],
    z_tracer_new_low: fa.CellKField[float],
    z_tracer_max: fa.CellKField[float],
    z_tracer_min: fa.CellKField[float],
    lo_bound: gtx.int32,
    hi_bound: gtx.int32,
) -> tuple[
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
]:
    condition = (refin_ctrl == lo_bound) | (refin_ctrl == hi_bound)
    z_tracer_new_out = where(
        condition,
        minimum(1.1 * p_cc, maximum(0.9 * p_cc, z_tracer_new_low)),
        z_tracer_new_low,
    )

    z_tracer_max_out = where(condition, maximum(p_cc, z_tracer_new_out), z_tracer_max)
    z_tracer_min_out = where(condition, minimum(p_cc, z_tracer_new_out), z_tracer_min)

    return (z_tracer_new_out, z_tracer_max_out, z_tracer_min_out)


@program(grid_type=GridType.UNSTRUCTURED)
def hflx_limiter_mo_stencil_02(
    refin_ctrl: fa.CellField[gtx.int32],
    p_cc: fa.CellKField[float],
    z_tracer_new_low: fa.CellKField[float],
    z_tracer_max: fa.CellKField[float],
    z_tracer_min: fa.CellKField[float],
    lo_bound: gtx.int32,
    hi_bound: gtx.int32,
    z_tracer_new_low_out: fa.CellKField[float],
    z_tracer_max_out: fa.CellKField[float],
    z_tracer_min_out: fa.CellKField[float],
):
    _hflx_limiter_mo_stencil_02(
        refin_ctrl,
        p_cc,
        z_tracer_new_low,
        z_tracer_max,
        z_tracer_min,
        lo_bound,
        hi_bound,
        out=(z_tracer_new_low_out, z_tracer_max_out, z_tracer_min_out),
    )
