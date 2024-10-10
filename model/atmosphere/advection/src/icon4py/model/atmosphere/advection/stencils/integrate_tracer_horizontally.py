# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2CE, C2E
from icon4py.model.common.settings import backend


@gtx.field_operator
def _integrate_tracer_horizontally(
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    tracer_new_hor = (
        tracer_now * rhodz_now
        - p_dtime
        * deepatmo_divh
        * neighbor_sum(p_mflx_tracer_h(C2E) * geofac_div(C2CE), axis=dims.C2EDim)
    ) / rhodz_new

    return tracer_new_hor


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def integrate_tracer_horizontally(
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    tracer_new_hor: fa.CellKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
