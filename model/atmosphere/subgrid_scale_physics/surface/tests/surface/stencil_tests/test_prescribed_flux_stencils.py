# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.surface.stencils.prescribed_flux_stencils import (
    compute_prescribed_surface_fluxes,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.physics.thermodynamics import compute_moisture, compute_pressure
from icon4py.model.testing import stencil_tests


class TestComputePrescribedSurfaceFluxes(stencil_tests.StencilTest):
    PROGRAM = compute_prescribed_surface_fluxes
    OUTPUTS = (
        "evapotranspiration",
        "sensible_heat_flux",
        "latent_heat_flux",
        "thermal_energy_flux",
        "vapor_energy_flux",
    )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        surface_temperature: np.ndarray,
        surface_pressure: np.ndarray,
        temperature: np.ndarray,
        shflx: float,
        lhflx: float,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        # compute_sfc_sat_spec_humidity (water surface) and compute_sfc_density
        saturation_qv = compute_moisture.specific_humidity(
            compute_pressure.sat_pres_water(surface_temperature), surface_pressure
        )
        surface_density = surface_pressure / (
            constants.RD * surface_temperature * (1.0 + constants.RV_O_RD_MINUS_1 * saturation_qv)
        )
        # the isrfc_type == 1 branch of compute_sfc_fluxes, then compute_energy_fluxes
        evapotranspiration = -lhflx * surface_density
        sensible_heat_flux = -shflx * constants.CVD * surface_density
        latent_heat_flux = (
            -lhflx
            * surface_density
            * (
                constants.LATENT_HEAT_FOR_VAPORISATION_INVARIANT
                + (constants.CVV - constants.CPL) * surface_temperature
            )
        )
        thermal_energy_flux = sensible_heat_flux
        vapor_energy_flux = temperature * evapotranspiration * (constants.CVV - constants.CVD)

        cells = slice(horizontal_start, horizontal_end)
        outputs = {}
        for name, values in (
            ("evapotranspiration", evapotranspiration),
            ("sensible_heat_flux", sensible_heat_flux),
            ("latent_heat_flux", latent_heat_flux),
            ("thermal_energy_flux", thermal_energy_flux),
            ("vapor_energy_flux", vapor_energy_flux),
        ):
            outputs[name] = np.zeros_like(surface_pressure)
            outputs[name][cells] = values[cells]
        return outputs

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return dict(
            surface_temperature=data_alloc.random_field(dims.CellDim, low=270.0, high=310.0),
            surface_pressure=data_alloc.random_field(dims.CellDim, low=9.0e4, high=1.05e5),
            temperature=data_alloc.random_field(dims.CellDim, low=260.0, high=305.0),
            shflx=0.1,
            lhflx=2.0e-5,
            evapotranspiration=data_alloc.zero_field(dims.CellDim),
            sensible_heat_flux=data_alloc.zero_field(dims.CellDim),
            latent_heat_flux=data_alloc.zero_field(dims.CellDim),
            thermal_energy_flux=data_alloc.zero_field(dims.CellDim),
            vapor_energy_flux=data_alloc.zero_field(dims.CellDim),
            # narrower than the field, which the grid's zones are not
            horizontal_start=1,
            horizontal_end=grid.num_cells - 1,
        )
