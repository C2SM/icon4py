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
from gt4py.next.ffront.fbuiltins import Field, int32, exp, astype, where

from icon4py.model.common.dimension import EdgeDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _calc_nudgecoeffs(
    refin_ctrl: Field[[EdgeDim], int32],
    grf_nudge_start_e: int32,
    nudge_max_coeffs: wpfloat,
) -> Field[[EdgeDim], wpfloat]:

    return where( ( (refin_ctrl > int32(0)) & (refin_ctrl <=  (grf_nudge_start_e + int32(9)) ) ) ,nudge_max_coeffs * exp(-(astype(refin_ctrl-grf_nudge_start_e,wpfloat)/ 4.0)),0.0)


@program(grid_type=GridType.UNSTRUCTURED)
def calc_nudgecoeffs(
    refin_ctrl: Field[[EdgeDim], int32],
    grf_nudge_start_e: int32,
    nudge_max_coeffs: wpfloat,
    nudgecoeffs_e: Field[[EdgeDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
):
    _calc_nudgecoeffs(
        refin_ctrl,
        grf_nudge_start_e,
        nudge_max_coeffs,
        out=nudgecoeffs_e,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
        },
    )

