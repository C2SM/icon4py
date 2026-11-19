# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the SurfaceComponent wrapper: the component contract, not the physics."""

import datetime

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.surface import (
    component as surface_component,
    config as surface_config,
    data as surface_data,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc


_CONFIG = surface_config.SurfaceConfig(
    surface_flux_type=surface_config.SurfaceFluxType.FIXED_HEAT_FLUXES,
    kinematic_sensible_heat_flux=0.1,
    kinematic_latent_heat_flux=0.0,
    sea_surface_temperature=303.15,
)
_TIME = datetime.datetime(2008, 9, 1, tzinfo=datetime.UTC)


def _inputs(grid):
    return {
        "surface_pressure": data_alloc.constant_field(grid, 1.0e5, dims.CellDim),
        "temperature": data_alloc.constant_field(grid, 300.0, dims.CellDim),
    }


def test_call_returns_the_declared_outputs_from_its_own_buffers():
    grid = simple.simple_grid()
    component = surface_component.SurfaceComponent(grid=grid, config=_CONFIG)

    outputs = component(_inputs(grid), _TIME)

    assert set(outputs) == set(surface_data.OUTPUTS_PROPERTIES)
    assert np.all(outputs["sensible_heat_flux"].asnumpy() != 0.0)
    assert np.all(outputs["surface_temperature"].asnumpy() == 303.15)


def test_bound_buffers_are_written_in_place():
    grid = simple.simple_grid()
    component = surface_component.SurfaceComponent(grid=grid, config=_CONFIG)
    buffers = {
        name: data_alloc.zero_field(grid, dims.CellDim) for name in surface_data.OUTPUTS_PROPERTIES
    }
    component.bind_output_buffers(buffers)
    # the constant surface temperature is filled at binding time
    assert np.all(buffers["surface_temperature"].asnumpy() == 303.15)

    outputs = component(_inputs(grid), _TIME)

    for name, buffer in buffers.items():
        assert outputs[name] is buffer
    assert np.all(buffers["sensible_heat_flux"].asnumpy() != 0.0)
