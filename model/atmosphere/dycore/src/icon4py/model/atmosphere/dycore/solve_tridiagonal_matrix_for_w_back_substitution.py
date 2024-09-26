# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import program, scan_operator
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@scan_operator(axis=dims.KDim, forward=False, init=wpfloat("0.0"))
def _solve_tridiagonal_matrix_for_w_back_substitution_scan(
    w_state: wpfloat, z_q: vpfloat, w: wpfloat
) -> wpfloat:
    """Formerly known as _mo_solve_nonhydro_stencil_53_scan."""
    return w + w_state * astype(z_q, wpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def solve_tridiagonal_matrix_for_w_back_substitution(
    z_q: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _solve_tridiagonal_matrix_for_w_back_substitution_scan(
        z_q,
        w,
        out=w,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
