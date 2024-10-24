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
from gt4py.next.ffront.fbuiltins import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _init_cell_kdim_field_with_zero_vp() -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_03, _mo_solve_nonhydro_stencil_11_lower, _mo_solve_nonhydro_stencil_45, _mo_solve_nonhydro_stencil_45_b, or _mo_velocity_advection_stencil_12."""
    return broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim))


@program(grid_type=GridType.UNSTRUCTURED)
def init_cell_kdim_field_with_zero_vp(
    field_with_zero_vp: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _init_cell_kdim_field_with_zero_vp(
        out=field_with_zero_vp,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
