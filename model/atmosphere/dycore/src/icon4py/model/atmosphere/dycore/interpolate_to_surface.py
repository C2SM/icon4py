# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _interpolate_to_surface(
    wgtfacq_c: fa.CellKField[vpfloat],
    interpolant: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_04."""
    interpolation_to_surface = (
        wgtfacq_c(Koff[-1]) * interpolant(Koff[-1])
        + wgtfacq_c(Koff[-2]) * interpolant(Koff[-2])
        + wgtfacq_c(Koff[-3]) * interpolant(Koff[-3])
    )
    return interpolation_to_surface


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_to_surface(
    wgtfacq_c: fa.CellKField[vpfloat],
    interpolant: fa.CellKField[vpfloat],
    interpolation_to_surface: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _interpolate_to_surface(
        wgtfacq_c,
        interpolant,
        out=interpolation_to_surface,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
