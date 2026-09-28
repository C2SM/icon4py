# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.tridiagonal import (
    _solve_tridiagonal_matrix_back_substitution_on_half_levels_mixed_precision,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_tridiagonal_matrix_for_w_back_substitution(
    z_q: fa.CellKHalfField[vpfloat],
    w: fa.CellKHalfField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _solve_tridiagonal_matrix_back_substitution_on_half_levels_mixed_precision(
        q=z_q,
        d_prime=w,
        out=w,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )
