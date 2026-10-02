# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.surface_fluxes import (
    compute_prescribed_surface_fluxes,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.physics.thermodynamics import compute_moisture, compute_pressure
from icon4py.model.testing import stencil_tests


class TestComputePrescribedSurfaceFluxes(stencil_tests.StencilTest):
    PROGRAM = compute_prescribed_surface_fluxes
    OUTPUTS = ("sensible_heat_flux", "evapotranspiration")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        surface_temperature: np.ndarray,
        surface_pressure: np.ndarray,
        shflx: float,
        lhflx: float,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        saturation_qv = compute_moisture.specific_humidity(
            compute_pressure.sat_pres_water(surface_temperature), surface_pressure
        )
        surface_density = surface_pressure / (
            constants.RD * surface_temperature * (1.0 + constants.RV_O_RD_MINUS_1 * saturation_qv)
        )
        cells = slice(horizontal_start, horizontal_end)
        sensible_heat_flux = np.zeros_like(surface_pressure)
        evapotranspiration = np.zeros_like(surface_pressure)
        sensible_heat_flux[cells] = -shflx * constants.CVD * surface_density[cells]
        evapotranspiration[cells] = -lhflx * surface_density[cells]
        return dict(sensible_heat_flux=sensible_heat_flux, evapotranspiration=evapotranspiration)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return dict(
            surface_temperature=data_alloc.random_field(dims.CellDim, low=270.0, high=310.0),
            surface_pressure=data_alloc.random_field(dims.CellDim, low=9.0e4, high=1.05e5),
            sensible_heat_flux=data_alloc.zero_field(dims.CellDim),
            evapotranspiration=data_alloc.zero_field(dims.CellDim),
            shflx=0.1,
            lhflx=2.0e-5,
            # narrower than the field, which the grid's zones are not
            horizontal_start=1,
            horizontal_end=grid.num_cells - 1,
        )
