# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, where

from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_vp import (
    _init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _init_two_cell_kdim_fields_index_with_zero_vp(
    field_index_with_zero_1: fa.CellKField[vpfloat],
    field_index_with_zero_2: fa.CellKField[vpfloat],
    k: fa.KField[gtx.int32],
    k1: gtx.int32,
    k2: gtx.int32,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_45 and _mo_solve_nonhydro_stencil_45_b."""
    k = broadcast(k, (dims.CellDim, dims.KDim))

    field_index_with_zero_1 = where(
        (k == k1), _init_cell_kdim_field_with_zero_vp(), field_index_with_zero_1
    )
    field_index_with_zero_2 = where(
        (k == k2), _init_cell_kdim_field_with_zero_vp(), field_index_with_zero_2
    )

    return field_index_with_zero_1, field_index_with_zero_2


@program(grid_type=GridType.UNSTRUCTURED)
def init_two_cell_kdim_fields_index_with_zero_vp(
    field_index_with_zero_1: fa.CellKField[vpfloat],
    field_index_with_zero_2: fa.CellKField[vpfloat],
    k: fa.KField[gtx.int32],
    k1: gtx.int32,
    k2: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _init_two_cell_kdim_fields_index_with_zero_vp(
        field_index_with_zero_1,
        field_index_with_zero_2,
        k,
        k1,
        k2,
        out=(field_index_with_zero_1, field_index_with_zero_2),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
