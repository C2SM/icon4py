# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, exp, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_nudgecoeffs(
    refin_ctrl: fa.EdgeField[gtx.int32],
    grf_nudge_start_e: gtx.int32,
    max_nudging_coefficient: wpfloat,
    nudge_efold_width: wpfloat,
    nudge_zone_width: gtx.int32,
) -> fa.EdgeField[wpfloat]:
    return where(
        ((refin_ctrl > 0) & (refin_ctrl <= (2 * nudge_zone_width + (grf_nudge_start_e - 3)))),
        max_nudging_coefficient
        * exp((-(astype(refin_ctrl - grf_nudge_start_e, wpfloat))) / (2.0 * nudge_efold_width)),
        0.0,
    )


# TODO (@halungge) not registered in factory
@program(grid_type=GridType.UNSTRUCTURED)
def compute_nudgecoeffs(
    nudging_coefficients_for_edges: fa.EdgeField[wpfloat],
    refin_ctrl: fa.EdgeField[gtx.int32],
    grf_nudge_start_e: gtx.int32,
    max_nudging_coefficient: wpfloat,
    nudge_efold_width: wpfloat,
    nudge_zone_width: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    """Compute nudging coefficient for edges based the grid refinement level of an edge."""
    _compute_nudgecoeffs(
        refin_ctrl,
        grf_nudge_start_e,
        max_nudging_coefficient,
        nudge_efold_width,
        nudge_zone_width,
        out=nudging_coefficients_for_edges,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )
