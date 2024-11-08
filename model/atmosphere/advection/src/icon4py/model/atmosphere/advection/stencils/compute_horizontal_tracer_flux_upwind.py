# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C
from icon4py.model.common.settings import backend


@gtx.field_operator
def _compute_horizontal_tracer_flux_upwind(
    p_cc: fa.CellKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    p_out_e = where(p_vn > 0.0, p_cc(E2C[0]), p_cc(E2C[1])) * p_mass_flx_e
    return p_out_e


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_horizontal_tracer_flux_upwind(
    p_cc: fa.CellKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    p_out_e: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_horizontal_tracer_flux_upwind(
        p_cc,
        p_mass_flx_e,
        p_vn,
        out=(p_out_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
