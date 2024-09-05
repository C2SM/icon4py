# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CellDim, KDim
from icon4py.model.common.settings import backend


@field_operator
def _integrate_tracer_horizontally(
    p_mflx_tracer_h: fa.EdgeKField[float],
    deepatmo_divh: fa.KField[float],
    tracer_now: fa.CellKField[float],
    rhodz_now: fa.CellKField[float],
    rhodz_new: fa.CellKField[float],
    geofac_div: Field[[dims.CEDim], float],
    p_dtime: float,
) -> fa.CellKField[float]:
    tracer_new_hor = (
        tracer_now * rhodz_now
        - p_dtime
        * deepatmo_divh
        * neighbor_sum(p_mflx_tracer_h(C2E) * geofac_div(C2CE), axis=C2EDim)
    ) / rhodz_new

    return tracer_new_hor


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def integrate_tracer_horizontally(
    p_mflx_tracer_h: fa.EdgeKField[float],
    deepatmo_divh: fa.KField[float],
    tracer_now: fa.CellKField[float],
    rhodz_now: fa.CellKField[float],
    rhodz_new: fa.CellKField[float],
    geofac_div: Field[[dims.CEDim], float],
    tracer_new_hor: fa.CellKField[float],
    p_dtime: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _integrate_tracer_horizontally(
        p_mflx_tracer_h,
        deepatmo_divh,
        tracer_now,
        rhodz_now,
        rhodz_new,
        geofac_div,
        p_dtime,
        out=tracer_new_hor,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
