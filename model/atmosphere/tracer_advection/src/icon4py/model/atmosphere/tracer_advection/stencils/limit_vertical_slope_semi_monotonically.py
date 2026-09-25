# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import abs, minimum, where  # noqa: A004
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _limit_vertical_slope_semi_monotonically_inner(
    p_cc: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """Limit the vertical slope for interior levels."""
    p_cc_min = minimum(minimum(p_cc(dims.KDim - 1), p_cc), p_cc(dims.KDim + 1))
    slope_l = minimum(abs(z_slope), 2.0 * (p_cc - p_cc_min))
    return where(z_slope >= 0.0, slope_l, -slope_l)


@gtx.field_operator
def _limit_vertical_slope_semi_monotonically_last(
    p_cc: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """Limit the vertical slope for the last level."""
    p_cc_min = minimum(p_cc(dims.KDim - 1), p_cc)
    slope_l = minimum(abs(z_slope), 2.0 * (p_cc - p_cc_min))
    return where(z_slope >= 0.0, slope_l, -slope_l)


@gtx.field_operator
def _limit_vertical_slope_semi_monotonically(
    p_cc: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
    elev: gtx.int32,
) -> fa.CellKField[ta.wpfloat]:
    return concat_where(
        dims.KDim == elev,
        _limit_vertical_slope_semi_monotonically_last(p_cc=p_cc, z_slope=z_slope),
        _limit_vertical_slope_semi_monotonically_inner(p_cc=p_cc, z_slope=z_slope),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def limit_vertical_slope_semi_monotonically(
    p_cc: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
    elev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _limit_vertical_slope_semi_monotonically(
        p_cc=p_cc,
        z_slope=z_slope,
        elev=elev,
        out=z_slope,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
