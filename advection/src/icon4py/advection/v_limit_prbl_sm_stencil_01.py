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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, FieldOffset, abs, minimum, where

from icon4py.common.dimension import CellDim, KDim


Koff = FieldOffset("Koff", source=KDim, target=(KDim,))


@field_operator
def _v_limit_prbl_sm_stencil_01(
    p_face: Field[[CellDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:

    z_delta = p_face - p_face(Koff[1])
    z_a6i = -6.0 * (p_cc - 0.5 * (p_face + p_face(Koff[1])))

    q_face_up, q_face_low = where(
        abs(z_delta) < z_a6i,
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


@program
def v_limit_prbl_sm_stencil_01(
    p_face: Field[[CellDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
    p_face_up: Field[[CellDim, KDim], float],
    p_face_low: Field[[CellDim, KDim], float],
):
    _v_limit_prbl_sm_stencil_01(
        p_face,
        p_cc,
        out=(p_face_up, p_face_low),
    )
