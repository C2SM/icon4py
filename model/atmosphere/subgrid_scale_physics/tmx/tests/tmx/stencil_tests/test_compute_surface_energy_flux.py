# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_surface_energy_flux import (
    compute_surface_energy_flux,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def surface_energy_flux_reference(
    grid: base.Grid,
    *,
    sensible_heat_flux: np.ndarray,
    evapotranspiration: np.ndarray,
    temperature_sfc: np.ndarray,
    use_internal_energy: bool,
    **kwargs: Any,
) -> dict:
    if use_internal_energy:
        ufts = sensible_heat_flux
        ufvs = temperature_sfc * evapotranspiration * (constants.CVV - constants.CVD)
        flux_x = ufts + ufvs
    else:
        flux_x = sensible_heat_flux * constants.CPD / constants.CVD
    return dict(flux_x=flux_x)


def surface_energy_flux_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid, use_internal_energy: bool
) -> dict[str, Any]:
    return dict(
        sensible_heat_flux=data_alloc.random_field(
            dims.CellDim, low=-200.0, high=200.0, dtype=wpfloat
        ),
        evapotranspiration=data_alloc.random_field(
            dims.CellDim, low=-1.0e-4, high=1.0e-4, dtype=wpfloat
        ),
        temperature_sfc=data_alloc.random_field(dims.CellDim, low=220.0, high=320.0, dtype=wpfloat),
        flux_x=data_alloc.zero_field(dims.CellDim, dtype=wpfloat),
        use_internal_energy=use_internal_energy,
        horizontal_start=0,
        horizontal_end=gtx.int32(grid.num_cells),
    )


# Static-params variants: prove that the config bool can be passed both as a regular
# runtime scalar ("none") and as a static (compile-time) argument selecting the variant.
STATIC_VARIANTS = {
    "none": (),
    "compile_time_variant": ("use_internal_energy",),
}


class TestComputeSurfaceEnergyFluxInternal(stencil_tests.StencilTest):
    PROGRAM = compute_surface_energy_flux
    OUTPUTS = ("flux_x",)
    STATIC_PARAMS = STATIC_VARIANTS

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return surface_energy_flux_reference(grid, **kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return surface_energy_flux_input_data(data_alloc, grid, use_internal_energy=True)


class TestComputeSurfaceEnergyFluxDryStatic(stencil_tests.StencilTest):
    PROGRAM = compute_surface_energy_flux
    OUTPUTS = ("flux_x",)
    STATIC_PARAMS = STATIC_VARIANTS

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return surface_energy_flux_reference(grid, **kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return surface_energy_flux_input_data(data_alloc, grid, use_internal_energy=False)
