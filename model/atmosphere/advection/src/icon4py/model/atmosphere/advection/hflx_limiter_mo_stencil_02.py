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

from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32, maximum, minimum, where
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim


@field_operator
def _hflx_limiter_mo_stencil_02(
    refin_ctrl: fa.CintField,
    p_cc: Field[[CellDim, KDim], float],
    z_tracer_new_low: Field[[CellDim, KDim], float],
    z_tracer_max: Field[[CellDim, KDim], float],
    z_tracer_min: Field[[CellDim, KDim], float],
    lo_bound: int32,
    hi_bound: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
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
    refin_ctrl: fa.CintField,
    p_cc: Field[[CellDim, KDim], float],
    z_tracer_new_low: Field[[CellDim, KDim], float],
    z_tracer_max: Field[[CellDim, KDim], float],
    z_tracer_min: Field[[CellDim, KDim], float],
    lo_bound: int32,
    hi_bound: int32,
    z_tracer_new_low_out: Field[[CellDim, KDim], float],
    z_tracer_max_out: Field[[CellDim, KDim], float],
    z_tracer_min_out: Field[[CellDim, KDim], float],
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
