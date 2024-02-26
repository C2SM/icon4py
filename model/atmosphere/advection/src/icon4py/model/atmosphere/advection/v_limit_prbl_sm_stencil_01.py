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
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    Field,
    FieldOffset,
    abs,
    int32,
    where,
)

from icon4py.model.common.dimension import CellDim, KDim


Koff = FieldOffset("Koff", source=KDim, target=(KDim,))


@field_operator
def _v_limit_prbl_sm_stencil_01(
    p_face: Field[[CellDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], int32]:
    z_delta = p_face - p_face(Koff[1])
    z_a6i = 6.0 * (p_cc - 0.5 * (p_face + p_face(Koff[1])))

    l_limit = where(abs(z_delta) < -1.0 * z_a6i, int32(1), int32(0))

    return l_limit


@program(grid_type=GridType.UNSTRUCTURED)
def v_limit_prbl_sm_stencil_01(
    p_face: Field[[CellDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
    l_limit: Field[[CellDim, KDim], int32],
):
    _v_limit_prbl_sm_stencil_01(
        p_face,
        p_cc,
        out=l_limit,
    )
