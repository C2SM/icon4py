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
def _compute_ppm_quartic_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
) -> fa.CellKHalfField[ta.wpfloat]:
    hgt_m2 = p_cellhgt_mc_now(dims.KHalfDim - 1.5)
    hgt_m1 = p_cellhgt_mc_now(dims.KHalfDim - 0.5)
    hgt = p_cellhgt_mc_now(dims.KHalfDim + 0.5)
    hgt_p1 = p_cellhgt_mc_now(dims.KHalfDim + 1.5)
    cc_m1 = p_cc(dims.KHalfDim - 0.5)
    cc = p_cc(dims.KHalfDim + 0.5)
    slope_m1 = z_slope(dims.KHalfDim - 0.5)
    slope = z_slope(dims.KHalfDim + 0.5)

    zgeo1 = hgt_m1 / (hgt_m1 + hgt)
    zgeo2 = 1.0 / (hgt_m2 + hgt_m1 + hgt + hgt_p1)
    zgeo3 = (hgt_m2 + hgt_m1) / (2.0 * hgt_m1 + hgt)
    zgeo4 = (hgt_p1 + hgt) / (2.0 * hgt + hgt_m1)

    p_face = (
        cc_m1
        + zgeo1 * (cc - cc_m1)
        + zgeo2
        * (
            (2.0 * hgt * zgeo1) * (zgeo3 - zgeo4) * (cc - cc_m1)
            - zgeo3 * hgt_m1 * slope
            + zgeo4 * hgt * slope_m1
        )
    )

    return p_face


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_quartic_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
    p_face: fa.CellKHalfField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_quartic_face_values(
        p_cc=p_cc,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        z_slope=z_slope,
        out=p_face,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )
