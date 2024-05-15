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
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _correct_contravariant_vertical_velocity(
    z_w_con_c: Field[[CellDim, KDim], vpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_13."""
    z_w_con_c_vp = z_w_con_c - w_concorr_c
    return z_w_con_c_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def correct_contravariant_vertical_velocity(
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    z_w_con_c: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _correct_contravariant_vertical_velocity(
        z_w_con_c,
        w_concorr_c,
        out=z_w_con_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
