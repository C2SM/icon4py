# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    FieldOffset,
    abs,
    where,
)

from icon4py.model.common import dimension as dims, field_type_aliases as fa


Koff = FieldOffset("Koff", source=dims.KDim, target=(dims.KDim,))


@field_operator
def _v_limit_prbl_sm_stencil_01(
    p_face: fa.CellKField[float],
    p_cc: fa.CellKField[float],
) -> fa.CellKField[gtx.int32]:
    z_delta = p_face - p_face(Koff[1])
    z_a6i = 6.0 * (p_cc - 0.5 * (p_face + p_face(Koff[1])))

    l_limit = where(abs(z_delta) < -1.0 * z_a6i, 1, 0)

    return l_limit


@program(grid_type=gtx.GridType.UNSTRUCTURED)
def v_limit_prbl_sm_stencil_01(
    p_face: fa.CellKField[float],
    p_cc: fa.CellKField[float],
    l_limit: fa.CellKField[gtx.int32],
):
    _v_limit_prbl_sm_stencil_01(
        p_face,
        p_cc,
        out=l_limit,
    )
