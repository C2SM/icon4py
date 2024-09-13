# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_tendency(
    p_tracer_now: fa.CellKField[wpfloat],
    p_tracer_new: fa.CellKField[wpfloat],
    p_dtime: wpfloat,
) -> fa.CellKField[wpfloat]:
    opt_ddt_tracer_adv = (p_tracer_new - p_tracer_now) / p_dtime
    return opt_ddt_tracer_adv


@program(grid_type=GridType.UNSTRUCTURED)
def compute_tendency(
    p_tracer_now: fa.CellKField[wpfloat],
    p_tracer_new: fa.CellKField[wpfloat],
    opt_ddt_tracer_adv: fa.CellKField[wpfloat],
    p_dtime: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_tendency(
        p_tracer_now,
        p_tracer_new,
        p_dtime,
        out=opt_ddt_tracer_adv,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
