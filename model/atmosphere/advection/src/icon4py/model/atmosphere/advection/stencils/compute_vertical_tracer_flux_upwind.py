# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_vertical_tracer_flux_upwind(
    p_cc: fa.CellKField[wpfloat],
    p_mflx_contra_v: fa.CellKField[wpfloat],  # TODO(dastrm): should be KHalfDim
) -> fa.CellKField[wpfloat]:
    p_upflux = where(p_mflx_contra_v >= wpfloat(0.0), p_cc, p_cc(Koff[-1])) * p_mflx_contra_v
    return p_upflux


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vertical_tracer_flux_upwind(
    p_cc: fa.CellKField[wpfloat],
    p_mflx_contra_v: fa.CellKField[wpfloat],  # TODO(dastrm): should be KHalfDim
    p_upflux: fa.CellKField[wpfloat],  # TODO(dastrm): should be KHalfDim
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vertical_tracer_flux_upwind(
        p_cc,
        p_mflx_contra_v,
        out=p_upflux,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
