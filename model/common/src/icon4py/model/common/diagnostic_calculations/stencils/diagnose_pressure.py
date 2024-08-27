# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import Field, exp, int32, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@scan_operator(axis=dims.KDim, forward=False, init=(0.0, 0.0, True))
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
    ddqz_z_full: fa.CellKField[wpfloat],
    temperature: fa.CellKField[vpfloat],
    pressure_sfc: Field[[dims.CellDim], vpfloat],
    grav_o_rd: wpfloat,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    pressure, pressure_ifc, _ = _scan_pressure(ddqz_z_full, temperature, pressure_sfc, grav_o_rd)
    return pressure, pressure_ifc


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def diagnose_pressure(
    ddqz_z_full: fa.CellKField[wpfloat],
    temperature: fa.CellKField[vpfloat],
    pressure_sfc: Field[[dims.CellDim], vpfloat],
    pressure: fa.CellKField[vpfloat],
    pressure_ifc: fa.CellKField[vpfloat],
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
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
