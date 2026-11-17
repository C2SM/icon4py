# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


"""Tests of the typed tmx component's contract: its input and output states."""

import dataclasses
import datetime

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.atmosphere.subgrid_scale_physics.tmx import (
    component as tmx_component,
    config as tmx_config,
    tmx_states,
)
from icon4py.model.common import framework as fw
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base as base_grid, simple, states as grid_states
from icon4py.model.common.states import surface_flux_state
from icon4py.model.testing import component_tests


_FLUXES = tuple(field.name for field in dataclasses.fields(surface_flux_state.SurfaceFluxState))


def _physics_state(**kwargs: object) -> physics_state.PhysicsState:
    return fw.allocate(
        physics_state.PhysicsState,
        simple.simple_grid(),
        None,
        step_start=datetime.datetime(2008, 9, 1),
        **kwargs,
    )


class TestTmx(component_tests.ComponentTest, component_tests.CollectInputTest):
    collect_input = staticmethod(tmx_component.collect_input)

    @pytest.fixture
    def component(self, grid: base_grid.Grid) -> tmx_component.Tmx:
        return tmx_component.Tmx(
            grid=grid,
            config=tmx_config.TmxConfig(),
            metric_state=component_tests.random_dataclass(tmx_states.TmxMetricState, grid),
            interpolation_state=component_tests.random_dataclass(
                tmx_states.TmxInterpolationState, grid
            ),
            edge_params=component_tests.random_dataclass(grid_states.EdgeParams, grid),
            cell_params=component_tests.random_dataclass(grid_states.CellParams, grid),
            dtime=datetime.timedelta(seconds=10),
            exchange=decomposition.SingleNodeExchange(),
        )

    @pytest.fixture
    def state(self) -> physics_state.PhysicsState:
        return _physics_state()

    @pytest.mark.parametrize(
        "present, missing", [(_FLUXES, "air_mass"), (("air_mass",), "evapotranspiration")]
    )
    def test_collect_input_requires_the_air_mass_and_the_surface_fluxes(
        self, present: tuple[str, ...], missing: str
    ) -> None:
        with pytest.raises(ValueError, match=missing):
            tmx_component.collect_input(_physics_state(only=present))
