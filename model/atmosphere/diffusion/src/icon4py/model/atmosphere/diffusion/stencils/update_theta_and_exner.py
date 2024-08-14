# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _update_theta_and_exner(
    z_temp: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: vpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    rd_o_cvd_wp, z_temp_wp = astype((rd_o_cvd, z_temp), wpfloat)

    z_theta = theta_v
    theta_v = theta_v + (area * z_temp_wp)
    exner = exner * (wpfloat("1.0") + rd_o_cvd_wp * (theta_v / z_theta - wpfloat("1.0")))
    return theta_v, exner


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def update_theta_and_exner(
    z_temp: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
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
