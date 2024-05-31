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
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@scan_operator(axis=KDim, forward=False, init=(0.0, 0.0, True))
def _scan_pressure(
    state: tuple[vpfloat, vpfloat, bool],
    ddqz_z_full: vpfloat,
    temperature: vpfloat,
    pressure_sfc: vpfloat,
    grav_o_rd: wpfloat,
):
    pressure_interface = (
        pressure_sfc * exp(-grav_o_rd * ddqz_z_full / temperature)
        if state[2]
        else state[1] * exp(-grav_o_rd * ddqz_z_full / temperature)
    )
    pressure = (
        sqrt(pressure_sfc * pressure_interface) if state[2] else sqrt(state[1] * pressure_interface)
    )
    return pressure, pressure_interface, False


@field_operator
def _diagnose_pressure(
    ddqz_z_full: fa.CKwpField,
    temperature: fa.CKvpField,
    pressure_sfc: Field[[CellDim], vpfloat],
    grav_o_rd: wpfloat,
) -> tuple[fa.CKvpField, fa.CKvpField]:
    pressure, pressure_ifc, _ = _scan_pressure(ddqz_z_full, temperature, pressure_sfc, grav_o_rd)
    return pressure, pressure_ifc


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def diagnose_pressure(
    ddqz_z_full: fa.CKwpField,
    temperature: fa.CKvpField,
    pressure_sfc: Field[[CellDim], vpfloat],
    pressure: fa.CKvpField,
    pressure_ifc: fa.CKvpField,
    grav_o_rd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _diagnose_pressure(
        ddqz_z_full,
        temperature,
        pressure_sfc,
        grav_o_rd,
        out=(pressure, pressure_ifc),
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
