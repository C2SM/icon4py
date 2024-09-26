# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import FieldOffset, minimum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa


Koff = FieldOffset("Koff", source=dims.KDim, target=(dims.KDim,))


@field_operator
def _v_limit_prbl_sm_stencil_02(
    l_limit: fa.CellKField[gtx.int32],
    p_face: fa.CellKField[float],
    p_cc: fa.CellKField[float],
) -> tuple[fa.CellKField[float], fa.CellKField[float]]:
    q_face_up, q_face_low = where(
        l_limit != 0,
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


@program(grid_type=GridType.UNSTRUCTURED)
def v_limit_prbl_sm_stencil_02(
    l_limit: fa.CellKField[gtx.int32],
    p_face: fa.CellKField[float],
    p_cc: fa.CellKField[float],
    p_face_up: fa.CellKField[float],
    p_face_low: fa.CellKField[float],
):
    _v_limit_prbl_sm_stencil_02(
        l_limit,
        p_face,
        p_cc,
        out=(p_face_up, p_face_low),
    )
