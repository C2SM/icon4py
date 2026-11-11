# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the typed surface component: the component contract, not the physics."""

import datetime
from typing import Any

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.atmosphere.subgrid_scale_physics.surface import (
    component as surface_component,
    config as surface_config,
    surface_states,
)
from icon4py.model.common import framework as fw
from icon4py.model.common.grid import base as base_grid
from icon4py.model.testing import component_tests


_CONFIG = surface_config.SurfaceConfig(
    surface_flux_type=surface_config.SurfaceFluxType.FIXED_HEAT_FLUXES,
    kinematic_sensible_heat_flux=0.1,
    kinematic_latent_heat_flux=0.0,
    sea_surface_temperature=303.15,
)


class TestSurface(component_tests.ComponentTest, component_tests.CollectInputTest):
    collect_input = staticmethod(surface_component.collect_input)

    @pytest.fixture
    def component(self, grid: base_grid.Grid) -> surface_component.Surface:
        return surface_component.Surface(grid=grid, config=_CONFIG)

    @pytest.fixture
    def state(self, grid: base_grid.Grid) -> physics_state.PhysicsState:
        return fw.allocate(
            physics_state.PhysicsState, grid, None, step_start=datetime.datetime(2008, 9, 1)
        )

    def test_run_writes_the_fluxes_and_the_surface_temperature(
        self,
        component: surface_component.Surface,
        grid: base_grid.Grid,
        output: surface_component.Surface.Output,
    ) -> None:
        values = {"pressure_ifc": 1.0e5, "temperature": 300.0}
        input = fw.allocate(
            surface_component.Surface.Input, grid, None, fill=lambda name, _: values[name]
        )

        component.run(input, output)

        assert np.all(output.sensible_heat_flux.asnumpy() != 0.0)
        assert np.all(output.surface_temperature.asnumpy() == 303.15)

    def test_run_hands_the_scheme_views_of_the_lowest_levels(
        self,
        grid: base_grid.Grid,
        input: surface_component.Surface.Input,
        output: surface_component.Surface.Output,
    ) -> None:
        seen: list[surface_states.SurfaceInputState] = []

        def compute(*, input_state: surface_states.SurfaceInputState, **_: Any) -> None:
            seen.append(input_state)

        component = surface_component.Surface(grid=grid, config=_CONFIG, compute=compute)

        component.run(input, output)

        (scheme_input,) = seen
        np.testing.assert_array_equal(
            scheme_input.surface_pressure.asnumpy(),
            input.pressure_ifc.asnumpy()[:, grid.num_levels],
        )
        np.testing.assert_array_equal(
            scheme_input.temperature.asnumpy(), input.temperature.asnumpy()[:, grid.num_levels - 1]
        )
        assert np.shares_memory(scheme_input.surface_pressure.ndarray, input.pressure_ifc.ndarray)
        assert np.shares_memory(scheme_input.temperature.ndarray, input.temperature.ndarray)
