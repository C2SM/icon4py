# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.thermo import (
    _sat_pres_ice,
    _sat_pres_water,
    _specific_humidity,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_saturation_humidity(
    temperature_sfc: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    over_ice: bool,
) -> fa.CellField[wpfloat]:
    """
    Compute the surface saturation specific humidity.

    Port of 'compute_sfc_sat_spec_humidity' (mo_tmx_surface.f90:698-704):
    saturation vapour pressure over ice for the sea-ice tile, over liquid water
    for ocean and land, then converted to a specific humidity at the surface
    pressure. ``over_ice`` selects the tile branch (compile-time static).

    Args:
        temperature_sfc: surface temperature [K]
        surface_pressure: surface pressure [Pa]
        over_ice: True for the sea-ice tile, False for ocean/land

    Returns:
        surface saturation specific humidity [kg/kg]
    """
    if over_ice:
        vapor_pressure = _sat_pres_ice(temperature_sfc)
    else:
        vapor_pressure = _sat_pres_water(temperature_sfc)
    return _specific_humidity(vapor_pressure, surface_pressure)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_saturation_humidity(
    temperature_sfc: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    qsat_sfc: fa.CellField[wpfloat],
    over_ice: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_saturation_humidity(
        temperature_sfc=temperature_sfc,
        surface_pressure=surface_pressure,
        over_ice=over_ice,
        out=qsat_sfc,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
