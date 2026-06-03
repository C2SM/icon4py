# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import maximum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _apply_horizontal_density_increment(
    p_rhodz_new: fa.CellKField[wpfloat],
    p_mflx_contra_v: fa.CellKField[wpfloat],
    deepatmo_divzl: fa.KField[wpfloat],
    deepatmo_divzu: fa.KField[wpfloat],
    p_dtime: wpfloat,
) -> fa.CellKField[wpfloat]:
    return maximum(wpfloat(0.1) * p_rhodz_new, p_rhodz_new) - p_dtime * (
        p_mflx_contra_v(Koff[1]) * deepatmo_divzl - p_mflx_contra_v * deepatmo_divzu
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_horizontal_density_increment(
    p_rhodz_new: fa.CellKField[wpfloat],
    p_mflx_contra_v: fa.CellKField[wpfloat],
    deepatmo_divzl: fa.KField[wpfloat],
    deepatmo_divzu: fa.KField[wpfloat],
    rhodz_ast2: fa.CellKField[wpfloat],
    p_dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_horizontal_density_increment(
        p_rhodz_new,
        p_mflx_contra_v,
        deepatmo_divzl,
        deepatmo_divzu,
        p_dtime,
        out=rhodz_ast2,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
