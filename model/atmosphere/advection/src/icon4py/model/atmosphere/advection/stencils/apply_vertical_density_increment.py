# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend


@gtx.field_operator
def _apply_vertical_density_increment(
    rhodz_ast: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    k_offset_up_low = p_dtime * (
        p_mflx_contra_v(Koff[1]) * deepatmo_divzl - p_mflx_contra_v * deepatmo_divzu
    )
    return rhodz_ast + k_offset_up_low


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def apply_vertical_density_increment(
    rhodz_ast: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    rhodz_ast2: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_vertical_density_increment(
        rhodz_ast,
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
