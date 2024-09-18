# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


# TODO (dastrm): move this highly generic stencil to common
# TODO (dastrm): this stencil has no test


@field_operator
def _init_constant_cell_kdim_field(value: wpfloat) -> fa.CellKField[wpfloat]:
    return broadcast(value, (dims.CellDim, dims.KDim))


@program(grid_type=GridType.UNSTRUCTURED)
def init_constant_cell_kdim_field(
    field: fa.CellKField[wpfloat],
    value: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _init_constant_cell_kdim_field(
        value,
        out=field,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
