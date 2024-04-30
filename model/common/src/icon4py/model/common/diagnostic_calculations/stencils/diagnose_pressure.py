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
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import Field, exp, int32, sqrt

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@scan_operator(axis=KDim, forward=False, init=(True, 0.0, 0.0))
def _scan_pressure(
    state: tuple[bool, float, float],
    ddqz_z_full: float,
    temperature: float,
    pressure_sfc: float,
):
    if state[0]:
        pressure_first_level = pressure_sfc * exp(-ddqz_z_full / temperature)
        pressure = sqrt(pressure_sfc * pressure_first_level)
        return False, pressure, pressure_first_level
    else:
        pressure_interface = state[1] * exp(-ddqz_z_full / temperature)
        pressure = sqrt(state[1] * pressure_interface)
        return False, pressure, pressure_interface


@field_operator
def _diagnose_pressure(
    ddqz_z_full: Field[[CellDim, KDim], wpfloat],
    temperature: Field[[CellDim, KDim], vpfloat],
    pressure_sfc: Field[[CellDim], vpfloat],
) -> tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], vpfloat]]:
    redundant, pressure, pressure_ifc = _scan_pressure(ddqz_z_full, temperature, pressure_sfc)
    return pressure, pressure_ifc


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def diagnose_pressure(
    ddqz_z_full: Field[[CellDim, KDim], wpfloat],
    temperature: Field[[CellDim, KDim], vpfloat],
    pressure_sfc: Field[[CellDim], vpfloat],
    pressure: Field[[CellDim, KDim], vpfloat],
    pressure_ifc: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _diagnose_pressure(
        ddqz_z_full,
        temperature,
        pressure_sfc,
        out=(pressure, pressure_ifc),
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
