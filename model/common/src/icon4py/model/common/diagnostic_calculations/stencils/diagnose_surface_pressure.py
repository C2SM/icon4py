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
from gt4py.next.ffront.fbuiltins import Field, exp, int32, log

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _diagnose_surface_pressure(
    exner: Field[[CellDim, KDim], vpfloat],
    temperature: Field[[CellDim, KDim], vpfloat],
    ddqz_z_full: Field[[CellDim, KDim], wpfloat],
    cpd_o_rd: wpfloat,
    p0ref: wpfloat,
    grav_o_rd: wpfloat,
) -> Field[[CellDim, KDim], vpfloat]:
    pressure_sfc = p0ref * exp(
        cpd_o_rd * log(exner(Koff[-3]))
        + grav_o_rd
        * (
            ddqz_z_full(Koff[-1]) / temperature(Koff[-1])
            + ddqz_z_full(Koff[-2]) / temperature(Koff[-2])
            + 0.5 * ddqz_z_full(Koff[-3]) / temperature(Koff[-3])
        )
    )
    return pressure_sfc


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def diagnose_surface_pressure(
    exner: Field[[CellDim, KDim], vpfloat],
    temperature: Field[[CellDim, KDim], vpfloat],
    ddqz_z_full: Field[[CellDim, KDim], wpfloat],
    pressure_sfc: Field[[CellDim, KDim], vpfloat],
    cpd_o_rd: wpfloat,
    p0ref: wpfloat,
    grav_o_rd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _diagnose_surface_pressure(
        exner,
        temperature,
        ddqz_z_full,
        cpd_o_rd,
        p0ref,
        grav_o_rd,
        out=pressure_sfc,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
