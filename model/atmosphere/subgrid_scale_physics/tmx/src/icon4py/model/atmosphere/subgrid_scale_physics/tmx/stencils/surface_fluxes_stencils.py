# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Stencils of the tmx surface fluxes."""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.physics.thermodynamics.compute_moisture import specific_humidity_on_cells
from icon4py.model.common.physics.thermodynamics.compute_pressure import sat_pres_water_on_cells
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_prescribed_surface_fluxes(
    surface_temperature: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    shflx: wpfloat,
    lhflx: wpfloat,
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """Sensible heat flux and evapotranspiration from the prescribed kinematic fluxes."""
    saturation_qv = specific_humidity_on_cells(
        sat_pres_water_on_cells(surface_temperature), surface_pressure
    )
    surface_density = surface_pressure / (
        PhysicsConstants.rd
        * surface_temperature
        * (wpfloat("1.0") + PhysicsConstants.rv_o_rd_minus_1 * saturation_qv)
    )
    return -shflx * PhysicsConstants.cvd * surface_density, -lhflx * surface_density


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_prescribed_surface_fluxes(
    surface_temperature: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    sensible_heat_flux: fa.CellField[wpfloat],
    evapotranspiration: fa.CellField[wpfloat],
    shflx: wpfloat,
    lhflx: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_prescribed_surface_fluxes(
        surface_temperature,
        surface_pressure,
        shflx,
        lhflx,
        out=(sensible_heat_flux, evapotranspiration),
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )
