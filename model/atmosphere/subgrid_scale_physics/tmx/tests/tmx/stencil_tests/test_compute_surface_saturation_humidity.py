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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_saturation_humidity import (
    compute_surface_saturation_humidity,
)
from icon4py.model.common import dimension as dims, thermodynamic_functions as thermo_host
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def surface_saturation_humidity_reference(
    connectivities: dict[gtx.Dimension, np.ndarray],
    *,
    temperature_sfc: np.ndarray,
    surface_pressure: np.ndarray,
    over_ice: bool,
    **kwargs: Any,
) -> dict:
    # reuse the host thermodynamic functions as the reference
    vapor_pressure = (
        thermo_host.sat_pres_ice(temperature_sfc)
        if over_ice
        else thermo_host.sat_pres_water(temperature_sfc)
    )
    qsat_sfc = thermo_host.specific_humidity(vapor_pressure, surface_pressure)
    return dict(qsat_sfc=qsat_sfc)


def surface_saturation_humidity_input_data(grid: base.Grid, over_ice: bool) -> dict[str, Any]:
    return dict(
        temperature_sfc=data_alloc.random_field(
            grid, dims.CellDim, low=240.0, high=310.0, dtype=wpfloat
        ),
        surface_pressure=data_alloc.random_field(
            grid, dims.CellDim, low=9.0e4, high=1.05e5, dtype=wpfloat
        ),
        qsat_sfc=data_alloc.zero_field(grid, dims.CellDim, dtype=wpfloat),
        over_ice=over_ice,
        horizontal_start=0,
        horizontal_end=gtx.int32(grid.num_cells),
    )


STATIC_VARIANTS = {
    "none": (),
    "compile_time_variant": ("over_ice",),
}


class TestComputeSurfaceSaturationHumidityWater(StencilTest):
    PROGRAM = compute_surface_saturation_humidity
    OUTPUTS = ("qsat_sfc",)
    STATIC_PARAMS = STATIC_VARIANTS

    reference = staticmethod(surface_saturation_humidity_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return surface_saturation_humidity_input_data(grid, over_ice=False)


class TestComputeSurfaceSaturationHumidityIce(StencilTest):
    PROGRAM = compute_surface_saturation_humidity
    OUTPUTS = ("qsat_sfc",)
    STATIC_PARAMS = STATIC_VARIANTS

    reference = staticmethod(surface_saturation_humidity_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return surface_saturation_humidity_input_data(grid, over_ice=True)
