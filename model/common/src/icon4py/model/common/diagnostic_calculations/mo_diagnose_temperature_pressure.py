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
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, exp, log, sqrt

from icon4py.model.common.dimension import CellDim, KDim

@field_operator
def _mo_diagnose_temperature(
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    temperature = theta_v * exner
    return temperature

@field_operator
def _mo_diagnose_pressure_sfc(
    exner_nlev_minus2: Field[[CellDim], float],
    temperature_nlev: Field[[CellDim], float],
    temperature_nlev_minus1: Field[[CellDim], float],
    temperature_nlev_minus2: Field[[CellDim], float],
    ddqz_z_full_nlev: Field[[CellDim], float],
    ddqz_z_full_nlev_minus1: Field[[CellDim], float],
    ddqz_z_full_nlev_minus2: Field[[CellDim], float],
    cpd_o_rd: float,
    p0ref: float,
    grav_o_rd: float,
) -> Field[[CellDim], float]:
    pressure_sfc = (
        p0ref * exp(cpd_o_rd * log(exner_nlev_minus2) + grav_o_rd * (ddqz_z_full_nlev / temperature_nlev + ddqz_z_full_nlev_minus1 / temperature_nlev_minus1 + 0.5 * ddqz_z_full_nlev_minus2 / temperature_nlev_minus2))
    )
    return pressure_sfc

@scan_operator(axis=KDim,forward=False,init=(True,0.0,0.0))
def _scan_pressure(
    state: tuple[bool, float, float],
    ddqz_z_full: float,
    temperature: float,
    pressure_sfc: float,
):
    if state[0]:
        pressure_first_level = pressure_sfc * exp( -ddqz_z_full / temperature )
        pressure = sqrt( pressure_sfc * pressure_first_level )
        return False, pressure, pressure_first_level
    else:
        pressure_interface = state[1] * exp( -ddqz_z_full / temperature )
        pressure = sqrt( state[1] * pressure_interface )
        return False, pressure, pressure_interface

@field_operator
def _mo_diagnose_pressure(
    ddqz_z_full: Field[[CellDim, KDim], float],
    temperature: Field[[CellDim, KDim], float],
    pressure_sfc: Field[[CellDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    redundant, pressure, pressure_ifc = _scan_pressure(
        ddqz_z_full,
        temperature,
        pressure_sfc
    )
    return pressure, pressure_ifc

@program(grid_type=GridType.UNSTRUCTURED)
def mo_diagnose_temperature(
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    temperature: Field[[CellDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_diagnose_temperature(
        theta_v,
        exner,
        out=(
            temperature
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )

@program(grid_type=GridType.UNSTRUCTURED)
def mo_diagnose_pressure_sfc(
    exner_nlev_minus2: Field[[CellDim], float],
    temperature_nlev: Field[[CellDim], float],
    temperature_nlev_minus1: Field[[CellDim], float],
    temperature_nlev_minus2: Field[[CellDim], float],
    ddqz_z_full_nlev: Field[[CellDim], float],
    ddqz_z_full_nlev_minus1: Field[[CellDim], float],
    ddqz_z_full_nlev_minus2: Field[[CellDim], float],
    pressure_sfc: Field[[CellDim], float],
    cpd_o_rd: float,
    p0ref: float,
    grav_o_rd: float,
    horizontal_start: int32,
    horizontal_end: int32,
):
    _mo_diagnose_pressure_sfc(
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

@program
def mo_diagnose_pressure(
    ddqz_z_full: Field[[CellDim, KDim], float],
    temperature: Field[[CellDim, KDim], float],
    pressure_sfc: Field[[CellDim], float],
    pressure: Field[[CellDim, KDim], float],
    pressure_ifc: Field[[CellDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_diagnose_pressure(
        ddqz_z_full,
        temperature,
        pressure_sfc,
        out=(
            pressure,
            pressure_ifc
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end)
        },
    )
