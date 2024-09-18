# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32, minimum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C
from icon4py.model.common.type_alias import wpfloat


# TODO (dastrm): this stencil has no test


@field_operator
def _apply_monotone_horizontal_multiplicative_flux_factors(
    z_anti: fa.EdgeKField[wpfloat],
    r_m: fa.CellKField[wpfloat],
    r_p: fa.CellKField[wpfloat],
    z_mflx_low: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    r_frac = where(
        z_anti >= wpfloat(0.0),
        minimum(r_m(E2C[0]), r_p(E2C[1])),
        minimum(r_m(E2C[1]), r_p(E2C[0])),
    )
    return z_mflx_low + minimum(wpfloat(1.0), r_frac) * z_anti


@program
def apply_monotone_horizontal_multiplicative_flux_factors(
    z_anti: fa.EdgeKField[wpfloat],
    r_m: fa.CellKField[wpfloat],
    r_p: fa.CellKField[wpfloat],
    z_mflx_low: fa.EdgeKField[wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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
