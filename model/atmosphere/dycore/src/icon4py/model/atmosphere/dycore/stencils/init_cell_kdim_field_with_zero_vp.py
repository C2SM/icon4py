# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _init_cell_kdim_field_with_zero_vp() -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_03, _mo_solve_nonhydro_stencil_11_lower, _mo_solve_nonhydro_stencil_45, _mo_solve_nonhydro_stencil_45_b, or _mo_velocity_advection_stencil_12."""
    return broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
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
