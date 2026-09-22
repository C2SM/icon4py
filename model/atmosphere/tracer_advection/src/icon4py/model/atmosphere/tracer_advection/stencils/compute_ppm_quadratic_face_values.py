# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _compute_ppm_quadratic_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
) -> fa.CellKHalfField[ta.wpfloat]:
    hgt = p_cellhgt_mc_now(dims.KHalfDim + 0.5)
    hgt_m1 = p_cellhgt_mc_now(dims.KHalfDim - 0.5)
    cc = p_cc(dims.KHalfDim + 0.5)
    cc_m1 = p_cc(dims.KHalfDim - 0.5)
    p_face = cc * (1.0 - (hgt / hgt_m1)) + (hgt / (hgt_m1 + hgt)) * ((hgt / hgt_m1) * cc + cc_m1)

    return p_face


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_quadratic_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    p_face: fa.CellKHalfField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_quadratic_face_values(
        p_cc=p_cc,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        out=p_face,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )
