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

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@scan_operator(axis=KDim, forward=False, init=(0.0, 0.0, True))
def _scan_pressure(
    state: tuple[wpfloat, wpfloat, bool],
    ddqz_z_full: wpfloat,
    virtual_temperature: wpfloat,
    pressure_sfc: wpfloat,
    grav_o_rd: wpfloat,
):
    pressure_interface = (
        pressure_sfc * exp(-grav_o_rd * ddqz_z_full / virtual_temperature)
        if state[2]
        else state[1] * exp(-grav_o_rd * ddqz_z_full / virtual_temperature)
    )
    pressure = (
        sqrt(pressure_sfc * pressure_interface) if state[2] else sqrt(state[1] * pressure_interface)
    )
    return pressure, pressure_interface, False


@field_operator
def _diagnose_pressure(
    ddqz_z_full: fa.CellKField[wpfloat],
    virtual_temperature: fa.CellKField[wpfloat],
    pressure_sfc: Field[[CellDim], wpfloat],
    grav_o_rd: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Update pressure by assuming hydrostatic balance (dp/dz = -rho g = p g / Rd / Tv).
    Note that virtual temperature is used in the equation.

    Args:
        ddqz_z_full: vertical grid spacing at full levels [m]
        virtual_temperature: air virtual temperature [K]
        pressure_sfc: surface air pressure [Pa]
        grav_o_rd: gravitational constant / dry air constant [K kg m/s2/J]
    Returns:
        pressure at full levels, pressure at half levels (excluding surface level)
    """
    pressure, pressure_ifc, _ = _scan_pressure(
        ddqz_z_full, virtual_temperature, pressure_sfc, grav_o_rd
    )
    return pressure, pressure_ifc


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def diagnose_pressure(
    ddqz_z_full: fa.CellKField[wpfloat],
    virtual_temperature: fa.CellKField[wpfloat],
    pressure_sfc: Field[[CellDim], wpfloat],
    pressure: fa.CellKField[wpfloat],
    pressure_ifc: fa.CellKField[wpfloat],
    grav_o_rd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _diagnose_pressure(
        ddqz_z_full,
        virtual_temperature,
        pressure_sfc,
        grav_o_rd,
        out=(pressure, pressure_ifc),
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
