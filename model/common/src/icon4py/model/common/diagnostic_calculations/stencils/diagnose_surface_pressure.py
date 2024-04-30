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

from icon4py.model.common.dimension import CellDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _diagnose_surface_pressure(
    exner_nlev_minus2: Field[[CellDim], vpfloat],
    temperature_nlev: Field[[CellDim], vpfloat],
    temperature_nlev_minus1: Field[[CellDim], vpfloat],
    temperature_nlev_minus2: Field[[CellDim], vpfloat],
    ddqz_z_full_nlev: Field[[CellDim], wpfloat],
    ddqz_z_full_nlev_minus1: Field[[CellDim], wpfloat],
    ddqz_z_full_nlev_minus2: Field[[CellDim], wpfloat],
    cpd_o_rd: wpfloat,
    p0ref: wpfloat,
    grav_o_rd: wpfloat,
) -> Field[[CellDim], vpfloat]:
    pressure_sfc = p0ref * exp(
        cpd_o_rd * log(exner_nlev_minus2)
        + grav_o_rd
        * (
            ddqz_z_full_nlev / temperature_nlev
            + ddqz_z_full_nlev_minus1 / temperature_nlev_minus1
            + 0.5 * ddqz_z_full_nlev_minus2 / temperature_nlev_minus2
        )
    )
    return pressure_sfc


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def diagnose_surface_pressure(
    exner_nlev_minus2: Field[[CellDim], vpfloat],
    temperature_nlev: Field[[CellDim], vpfloat],
    temperature_nlev_minus1: Field[[CellDim], vpfloat],
    temperature_nlev_minus2: Field[[CellDim], vpfloat],
    ddqz_z_full_nlev: Field[[CellDim], wpfloat],
    ddqz_z_full_nlev_minus1: Field[[CellDim], wpfloat],
    ddqz_z_full_nlev_minus2: Field[[CellDim], wpfloat],
    pressure_sfc: Field[[CellDim], vpfloat],
    cpd_o_rd: wpfloat,
    p0ref: wpfloat,
    grav_o_rd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
):
    _diagnose_surface_pressure(
        exner_nlev_minus2,
        temperature_nlev,
        temperature_nlev_minus1,
        temperature_nlev_minus2,
        ddqz_z_full_nlev,
        ddqz_z_full_nlev_minus1,
        ddqz_z_full_nlev_minus2,
        cpd_o_rd,
        p0ref,
        grav_o_rd,
        out=pressure_sfc,
        domain={
            CellDim: (horizontal_start, horizontal_end),
        },
    )
