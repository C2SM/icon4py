# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.thermo import _potential_temperature
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_potential_temperatures(
    temperature_atm: fa.CellField[wpfloat],
    pressure_atm: fa.CellField[wpfloat],
    temperature_sfc: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """
    Compute the dry potential temperature of the lowest atmospheric level and of
    the surface.

    Port of 'compute_atm_potential_temperature' / 'compute_sfc_potential_temperature'
    (mo_vdf_diag_smag.f90), dry part only; the virtual potential temperatures
    (needed by the moist Richardson diagnostic) are added with that diagnostic.

    Args:
        temperature_atm: air temperature at the lowest full level [K]
        pressure_atm: air pressure at the lowest full level [Pa]
        temperature_sfc: surface temperature [K]
        surface_pressure: surface pressure [Pa]

    Returns:
        (theta_atm, theta_sfc): dry potential temperatures [K]
    """
    theta_atm = _potential_temperature(temperature_atm, pressure_atm)
    theta_sfc = _potential_temperature(temperature_sfc, surface_pressure)
    return theta_atm, theta_sfc


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_potential_temperatures(
    temperature_atm: fa.CellField[wpfloat],
    pressure_atm: fa.CellField[wpfloat],
    temperature_sfc: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    theta_atm: fa.CellField[wpfloat],
    theta_sfc: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_potential_temperatures(
        temperature_atm=temperature_atm,
        pressure_atm=pressure_atm,
        temperature_sfc=temperature_sfc,
        surface_pressure=surface_pressure,
        out=(theta_atm, theta_sfc),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
