# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import exp, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.settings import backend


@scan_operator(axis=dims.KDim, forward=False, init=(0.0, 0.0, True))
def _scan_pressure(
    state: tuple[ta.wpfloat, ta.wpfloat, bool],
    ddqz_z_full: ta.wpfloat,
    virtual_temperature: ta.wpfloat,
    surface_pressure: ta.wpfloat,
    grav_o_rd: ta.wpfloat,
):
    pressure_interface = (
        surface_pressure * exp(-grav_o_rd * ddqz_z_full / virtual_temperature)
        if state[2]
        else state[1] * exp(-grav_o_rd * ddqz_z_full / virtual_temperature)
    )
    pressure = (
        sqrt(surface_pressure * pressure_interface)
        if state[2]
        else sqrt(state[1] * pressure_interface)
    )
    return pressure, pressure_interface, False


@field_operator
def _diagnose_pressure(
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    surface_pressure: gtx.Field[gtx.Dims[dims.CellDim], ta.wpfloat],
    grav_o_rd: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """
    Update pressure by assuming hydrostatic balance (dp/dz = -rho g = p g / Rd / Tv).
    Note that virtual temperature is used in the equation.

    Args:
        ddqz_z_full: vertical grid spacing at full levels [m]
        virtual_temperature: air virtual temperature [K]
        surface_pressure: surface air pressure [Pa]
        grav_o_rd: gravitational constant / dry air constant [K kg m/s2/J]
    Returns:
        pressure at full levels, pressure at half levels (excluding surface level)
    """
    pressure, pressure_ifc, _ = _scan_pressure(
        ddqz_z_full, virtual_temperature, surface_pressure, grav_o_rd
    )
    return pressure, pressure_ifc


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def diagnose_pressure(
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    surface_pressure: gtx.Field[gtx.Dims[dims.CellDim], ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    pressure_ifc: fa.CellKField[ta.wpfloat],
    grav_o_rd: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _diagnose_pressure(
        ddqz_z_full,
        virtual_temperature,
        surface_pressure,
        grav_o_rd,
        out=(pressure, pressure_ifc),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
