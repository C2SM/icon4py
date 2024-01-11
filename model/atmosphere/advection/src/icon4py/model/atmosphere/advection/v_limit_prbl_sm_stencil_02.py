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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, FieldOffset, int32, minimum, where

from icon4py.model.common.dimension import CellDim, KDim


Koff = FieldOffset("Koff", source=KDim, target=(KDim,))


@field_operator
def _v_limit_prbl_sm_stencil_02(
    l_limit: Field[[CellDim, KDim], int32],
    p_face: Field[[CellDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    q_face_up, q_face_low = where(
        l_limit != int32(0),
        where(
            (p_cc < minimum(p_face, p_face(Koff[1]))),
            (p_cc, p_cc),
            where(
                p_face > p_face(Koff[1]),
                (3.0 * p_cc - 2.0 * p_face(Koff[1]), p_face(Koff[1])),
                (p_face, 3.0 * p_cc - 2.0 * p_face),
            ),
        ),
        (p_face, p_face(Koff[1])),
    )

    return q_face_up, q_face_low


@program(grid_type="unstructured")
def v_limit_prbl_sm_stencil_02(
    l_limit: Field[[CellDim, KDim], int32],
    p_face: Field[[CellDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
    p_face_up: Field[[CellDim, KDim], float],
    p_face_low: Field[[CellDim, KDim], float],
):
    _v_limit_prbl_sm_stencil_02(
        l_limit,
        p_face,
        p_cc,
        out=(p_face_up, p_face_low),
    )
