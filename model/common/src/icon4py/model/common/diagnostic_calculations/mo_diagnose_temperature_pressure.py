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
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, E2CDim, E2C

@field_operator
def _mo_diagnose_temperature(
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],

) -> Field[[EdgeDim, KDim], float]:
    temperature = theta_v * exner
    return temperature

@field_operator
def _mo_diagnose_pressure(
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],

) -> Field[[EdgeDim, KDim], float]:
    temperature = theta_v * exner
    return temperature

@program(grid_type=GridType.UNSTRUCTURED)
def mo_diagnose_temperature_pressure(
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    temperature: Field[[CellDim, KDim], float],
    pressure: Field[[CellDim, KDim], float],
    pressure_ifc: Field[[CellDim, KDim], float],
    ddqz_z_full: Field[[CellDim, KDim], float],
    pressure_sfc: Field[[CellDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_diagnose_temperature(
        theta_v,
        exner,
        out=temperature,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
