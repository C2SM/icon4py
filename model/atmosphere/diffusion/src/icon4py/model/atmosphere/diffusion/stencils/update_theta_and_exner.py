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
from gt4py.next.ffront.fbuiltins import Field, astype, int32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.model_backend import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _update_theta_and_exner(
    z_temp: Field[[CellDim, KDim], vpfloat],
    area: Field[[CellDim], wpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    exner: Field[[CellDim, KDim], wpfloat],
    rd_o_cvd: vpfloat,
) -> tuple[Field[[CellDim, KDim], wpfloat], Field[[CellDim, KDim], wpfloat]]:
    rd_o_cvd_wp, z_temp_wp = astype((rd_o_cvd, z_temp), wpfloat)

    z_theta = theta_v
    theta_v = theta_v + (area * z_temp_wp)
    exner = exner * (wpfloat("1.0") + rd_o_cvd_wp * (theta_v / z_theta - wpfloat("1.0")))
    return theta_v, exner


@program(grid_type=GridType.UNSTRUCTURED)
def update_theta_and_exner(
    z_temp: Field[[CellDim, KDim], vpfloat],
    area: Field[[CellDim], wpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    exner: Field[[CellDim, KDim], wpfloat],
    rd_o_cvd: vpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _update_theta_and_exner(
        z_temp,
        area,
        theta_v,
        exner,
        rd_o_cvd,
        out=(theta_v, exner),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
