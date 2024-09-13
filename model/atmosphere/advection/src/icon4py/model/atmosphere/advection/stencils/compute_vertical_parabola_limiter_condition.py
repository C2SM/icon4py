# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    FieldOffset,
    abs,
    int32,
    where,
)

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


Koff = FieldOffset("Koff", source=dims.KDim, target=(dims.KDim,))


@field_operator
def _compute_vertical_parabola_limiter_condition(
    p_face: fa.CellKField[wpfloat],
    p_cc: fa.CellKField[wpfloat],
) -> fa.CellKField[int32]:
    z_delta = p_face - p_face(Koff[1])
    z_a6i = wpfloat(6.0) * (p_cc - wpfloat(0.5) * (p_face + p_face(Koff[1])))

    l_limit = where(abs(z_delta) < -wpfloat(1.0) * z_a6i, 1, 0)

    return l_limit


@program(grid_type=GridType.UNSTRUCTURED)
def compute_vertical_parabola_limiter_condition(
    p_face: fa.CellKField[wpfloat],
    p_cc: fa.CellKField[wpfloat],
    l_limit: fa.CellKField[int32],
):
    _compute_vertical_parabola_limiter_condition(
        p_face,
        p_cc,
        out=l_limit,
    )
