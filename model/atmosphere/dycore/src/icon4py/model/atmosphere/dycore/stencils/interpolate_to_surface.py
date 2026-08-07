# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KHalfDim
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _interpolate_to_surface(
    wgtfacq_c: fa.CellKField[vpfloat],
    interpolant: fa.CellKField[vpfloat],
) -> fa.CellKHalfField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_04."""
    interpolation_to_surface = (
        wgtfacq_c(KHalfDim - 0.5) * interpolant(KHalfDim - 0.5)
        + wgtfacq_c(KHalfDim - 1.5) * interpolant(KHalfDim - 1.5)
        + wgtfacq_c(KHalfDim - 2.5) * interpolant(KHalfDim - 2.5)
    )
    return interpolation_to_surface


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_to_surface(
    wgtfacq_c: fa.CellKField[vpfloat],
    interpolant: fa.CellKField[vpfloat],
    interpolation_to_surface: fa.CellKHalfField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_to_surface(
        wgtfacq_c=wgtfacq_c,
        interpolant=interpolant,
        out=interpolation_to_surface,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )
