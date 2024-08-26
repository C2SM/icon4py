# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32, maximum

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim


@field_operator
def _step_advection_stencil_03(
    p_tracer_now: fa.CellKField[float],
    p_grf_tend_tracer: fa.CellKField[float],
    p_dtime: float,
) -> fa.CellKField[float]:
    p_tracer_new = maximum(0.0, p_tracer_now + p_dtime * p_grf_tend_tracer)
    return p_tracer_new


@program(grid_type=GridType.UNSTRUCTURED)
def step_advection_stencil_03(
    p_tracer_now: fa.CellKField[float],
    p_grf_tend_tracer: fa.CellKField[float],
    p_tracer_new: fa.CellKField[float],
    p_dtime: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _step_advection_stencil_03(
        p_tracer_now,
        p_grf_tend_tracer,
        p_dtime,
        out=p_tracer_new,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
