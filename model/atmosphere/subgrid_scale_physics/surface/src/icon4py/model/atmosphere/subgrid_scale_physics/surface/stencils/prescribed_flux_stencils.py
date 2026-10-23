# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Stencils of the prescribed-flux surface scheme (``isrfc_type = 1``)."""

import gt4py.next as gtx

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.dimension import CellDim
from icon4py.model.common.physics.thermodynamics.compute_moisture import _specific_humidity_on_cells
from icon4py.model.common.physics.thermodynamics.compute_pressure import _sat_pres_water_on_cells
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_density(
    surface_temperature: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """
    Air density at the surface from the surface pressure and the saturation specific humidity
    at the surface temperature: ``compute_sfc_sat_spec_humidity`` (mo_tmx_surface.f90, water
    surface) and ``compute_sfc_density`` (mo_vdf_diag_smag.f90).
    """
    saturation_specific_humidity = _specific_humidity_on_cells(
        _sat_pres_water_on_cells(surface_temperature), surface_pressure
    )
    return surface_pressure / (
        PhysicsConstants.rd
        * surface_temperature
        * (wpfloat("1.0") + PhysicsConstants.rv_o_rd_minus_1 * saturation_specific_humidity)
    )


@gtx.field_operator
def _compute_prescribed_surface_fluxes(
    surface_temperature: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    temperature: fa.CellField[wpfloat],
    shflx: wpfloat,
    lhflx: wpfloat,
) -> tuple[
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
]:
    """
    The ``isrfc_type == 1`` branch of ``compute_sfc_fluxes`` and ``compute_energy_fluxes``
    (mo_tmx_surface.f90): the prescribed kinematic fluxes ``shflx`` [K m/s] and ``lhflx`` [m/s]
    times the surface density, and the surface fluxes of internal energy they carry.

    ``temperature`` is the air temperature at the lowest level (``ptm1`` in the surface code).
    The wind stress and ``q_snocpymlt`` are not touched, as in ICON's early return.
    """
    surface_density = _compute_surface_density(surface_temperature, surface_pressure)
    evapotranspiration = -lhflx * surface_density
    sensible_heat_flux = -shflx * PhysicsConstants.cvd * surface_density
    latent_heat_flux = (
        -lhflx
        * surface_density
        * (
            PhysicsConstants.lvc
            + (PhysicsConstants.cvv - PhysicsConstants.cpl) * surface_temperature
        )
    )
    thermal_energy_flux = sensible_heat_flux
    vapor_energy_flux = (
        temperature * evapotranspiration * (PhysicsConstants.cvv - PhysicsConstants.cvd)
    )
    return (
        evapotranspiration,
        sensible_heat_flux,
        latent_heat_flux,
        thermal_energy_flux,
        vapor_energy_flux,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_prescribed_surface_fluxes(
    surface_temperature: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    temperature: fa.CellField[wpfloat],
    shflx: wpfloat,
    lhflx: wpfloat,
    evapotranspiration: fa.CellField[wpfloat],
    sensible_heat_flux: fa.CellField[wpfloat],
    latent_heat_flux: fa.CellField[wpfloat],
    thermal_energy_flux: fa.CellField[wpfloat],
    vapor_energy_flux: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_prescribed_surface_fluxes(
        surface_temperature,
        surface_pressure,
        temperature,
        shflx,
        lhflx,
        out=(
            evapotranspiration,
            sensible_heat_flux,
            latent_heat_flux,
            thermal_energy_flux,
            vapor_energy_flux,
        ),
        domain={CellDim: (horizontal_start, horizontal_end)},
    )
