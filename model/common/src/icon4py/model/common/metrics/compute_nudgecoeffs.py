# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, exp, int32, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


# TODO: this will have to be removed once domain allows for imports
EdgeDim = dims.EdgeDim


@field_operator
def _compute_nudgecoeffs(
    refin_ctrl: fa.EdgeField[int32],
    grf_nudge_start_e: int32,
    nudge_max_coeffs: wpfloat,
    nudge_efold_width: wpfloat,
    nudge_zone_width: int32,
) -> fa.EdgeField[wpfloat]:
    return where(
        ((refin_ctrl > 0) & (refin_ctrl <= (2 * nudge_zone_width + (grf_nudge_start_e - 3)))),
        nudge_max_coeffs
        * exp((-(astype(refin_ctrl - grf_nudge_start_e, wpfloat))) / (2.0 * nudge_efold_width)),
        0.0,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def compute_nudgecoeffs(
    nudgecoeffs_e: fa.EdgeField[wpfloat],
    refin_ctrl: fa.EdgeField[int32],
    grf_nudge_start_e: int32,
    nudge_max_coeffs: wpfloat,
    nudge_efold_width: wpfloat,
    nudge_zone_width: int32,
    horizontal_start: int32,
    horizontal_end: int32,
):
    """Compute nudging coefficient for edges based the grid refinement level of an edge."""
    _compute_nudgecoeffs(
        refin_ctrl,
        grf_nudge_start_e,
        nudge_max_coeffs,
        nudge_efold_width,
        nudge_zone_width,
        out=nudgecoeffs_e,
        domain={EdgeDim: (horizontal_start, horizontal_end)},
    )
