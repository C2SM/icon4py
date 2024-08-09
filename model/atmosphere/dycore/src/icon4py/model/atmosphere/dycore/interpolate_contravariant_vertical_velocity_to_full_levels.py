# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _interpolate_contravariant_vertical_velocity_to_full_levels(
    z_w_con_c: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly know as _mo_velocity_advection_stencil_15."""
    z_w_con_c_full_vp = vpfloat("0.5") * (z_w_con_c + z_w_con_c(Koff[1]))
    return z_w_con_c_full_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_contravariant_vertical_velocity_to_full_levels(
    z_w_con_c: fa.CellKField[vpfloat],
    z_w_con_c_full: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _interpolate_contravariant_vertical_velocity_to_full_levels(
        z_w_con_c,
        out=z_w_con_c_full,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
