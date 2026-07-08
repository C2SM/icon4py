# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _update_two_cell_kdim_fields_with_tendency(
    field_1: fa.CellKField[wpfloat],
    field_2: fa.CellKField[wpfloat],
    tendency_1: fa.CellKField[wpfloat],
    tendency_2: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Advance two cell K fields by one time step with their tendencies:

        new_field_i = field_i + tendency_i * dtime
    """
    new_field_1 = field_1 + tendency_1 * dtime
    new_field_2 = field_2 + tendency_2 * dtime
    return new_field_1, new_field_2


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_two_cell_kdim_fields_with_tendency(  # noqa: PLR0917 [too-many-positional-arguments]
    field_1: fa.CellKField[wpfloat],
    field_2: fa.CellKField[wpfloat],
    tendency_1: fa.CellKField[wpfloat],
    tendency_2: fa.CellKField[wpfloat],
    new_field_1: fa.CellKField[wpfloat],
    new_field_2: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_two_cell_kdim_fields_with_tendency(
        field_1=field_1,
        field_2=field_2,
        tendency_1=tendency_1,
        tendency_2=tendency_2,
        dtime=dtime,
        out=(new_field_1, new_field_2),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
