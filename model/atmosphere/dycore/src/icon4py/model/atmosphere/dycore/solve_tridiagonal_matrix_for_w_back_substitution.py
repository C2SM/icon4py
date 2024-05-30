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
from gt4py.next.ffront.decorator import program, scan_operator
from gt4py.next.ffront.fbuiltins import Field, astype, int32
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@scan_operator(axis=KDim, forward=False, init=wpfloat("0.0"))
def _solve_tridiagonal_matrix_for_w_back_substitution_scan(
    w_state: wpfloat, z_q: vpfloat, w: wpfloat
) -> wpfloat:
    """Formerly known as _mo_solve_nonhydro_stencil_53_scan."""
    return w + w_state * astype(z_q, wpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def solve_tridiagonal_matrix_for_w_back_substitution(
    z_q: Field[[CellDim, KDim], vpfloat],
    w: fa.CKwpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _solve_tridiagonal_matrix_for_w_back_substitution_scan(
        z_q,
        w,
        out=w,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
