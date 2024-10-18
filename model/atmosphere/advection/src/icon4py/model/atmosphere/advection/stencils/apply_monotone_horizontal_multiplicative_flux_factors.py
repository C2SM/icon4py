# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import minimum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C
from icon4py.model.common.settings import backend


# TODO (dastrm): this stencil has no test


@gtx.field_operator
def _apply_monotone_horizontal_multiplicative_flux_factors(
    z_anti: fa.EdgeKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    r_p: fa.CellKField[ta.wpfloat],
    z_mflx_low: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    r_frac = where(
        z_anti >= 0.0,
        minimum(r_m(E2C[0]), r_p(E2C[1])),
        minimum(r_m(E2C[1]), r_p(E2C[0])),
    )
    return z_mflx_low + minimum(1.0, r_frac) * z_anti


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def apply_monotone_horizontal_multiplicative_flux_factors(
    z_anti: fa.EdgeKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    r_p: fa.CellKField[ta.wpfloat],
    z_mflx_low: fa.EdgeKField[ta.wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_monotone_horizontal_multiplicative_flux_factors(
        z_anti,
        r_m,
        r_p,
        z_mflx_low,
        out=p_mflx_tracer_h,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
