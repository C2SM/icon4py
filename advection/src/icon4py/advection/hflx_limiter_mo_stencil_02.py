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

from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import maximum, minimum, where

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _hflx_limiter_mo_stencil_02(
    refin_ctrl: Field[[CellDim], int],
    p_cc: Field[[CellDim, KDim], float],
    z_tracer_new_low: Field[[CellDim, KDim], float],
    lo_bound: int,
    hi_bound: int,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    condition = (refin_ctrl == lo_bound) | (refin_ctrl == hi_bound)
    z_tracer_new_tmp = where(
        condition,
        minimum(1.1 * p_cc, maximum(0.9 * p_cc, z_tracer_new_low)),
        z_tracer_new_low,
    )
    z_tracer_max = where(condition, maximum(p_cc, z_tracer_new_tmp), z_tracer_new_low)
    z_tracer_min = where(condition, minimum(p_cc, z_tracer_new_tmp), z_tracer_new_low)
    return z_tracer_max, z_tracer_min


@program
def hflx_limiter_mo_stencil_02(
    refin_ctrl: Field[[CellDim], int],
    p_cc: Field[[CellDim, KDim], float],
    z_tracer_new_low: Field[[CellDim, KDim], float],
    lo_bound: int,
    hi_bound: int,
    z_tracer_max: Field[[CellDim, KDim], float],
    z_tracer_min: Field[[CellDim, KDim], float],
):
    _hflx_limiter_mo_stencil_02(
        refin_ctrl,
        p_cc,
        z_tracer_new_low,
        lo_bound,
        hi_bound,
        out=(z_tracer_max, z_tracer_min),
    )
