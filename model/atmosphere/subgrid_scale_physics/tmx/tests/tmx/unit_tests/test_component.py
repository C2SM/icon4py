# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the tmx component on the simple grid, with random static states."""

import dataclasses
import datetime
from typing import Any

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.atmosphere.subgrid_scale_physics.tmx import (
    component as tmx_component,
    config as tmx_config,
    tmx_states,
)
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.components import framework as fw
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base as base_grid, simple

from .utils import random_static_states


DTIME = datetime.timedelta(seconds=300)

# (low, high) of the random inputs, by leaf name
_INPUT_RANGES = {
    "temperature": (250.0, 300.0),
    "virtual_temperature": (250.0, 300.0),
    "pressure": (1.0e4, 1.0e5),
    "pressure_ifc": (1.0e4, 1.0e5),
    "u": (-10.0, 10.0),
    "v": (-10.0, 10.0),
    "w": (-1.0, 1.0),
    "rho": (0.5, 1.3),
}


@dataclasses.dataclass
class RecordingFluxProvider:
    """Writes a constant sensible heat flux and records the half-level pressure it is handed."""

    pressures: list[Any] = dataclasses.field(default_factory=list)

    def compute(
        self,
        *,
        pressure_ifc: fa.CellKHalfField[ta.wpfloat],
        out: tmx_states.TmxSurfaceFluxState,
    ) -> None:
        self.pressures.append(pressure_ifc)
        for field in dataclasses.fields(out):
            getattr(out, field.name).ndarray[...] = 0.0
        # NDArrayObject declares no __setitem__
        sensible_heat_flux: Any = out.sensible_heat_flux.ndarray
        sensible_heat_flux[...] = 10.0


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid(num_levels=10)


def _component(grid: base_grid.Grid, provider: RecordingFluxProvider) -> tmx_component.TmxComponent:
    static_states = random_static_states(grid, None)
    return tmx_component.TmxComponent(
        grid=grid,
        config=tmx_config.TmxConfig(),
        dtime=DTIME,
        metric_state=static_states.metric_state,
        interpolation_state=static_states.interpolation_state,
        edge_params=static_states.edge_params,
        cell_params=static_states.cell_params,
        surface_fluxes=provider,
        backend=None,
        exchange=decomposition.SingleNodeExchange(),
    )


def _inputs(grid: base_grid.Grid, seed: int) -> tmx_component.TmxComponent.Input:
    rng = np.random.default_rng(seed)

    def fill(name: str, shape: tuple[int, ...]) -> np.ndarray:
        low, high = _INPUT_RANGES.get(name, (0.0, 1.0e-3))  # the tracers
        return rng.uniform(low, high, shape)

    return fw.allocate(tmx_component.TmxComponent.Input, grid, None, fill=fill)


def test_collect_input_views_the_entry_state(grid: base_grid.Grid) -> None:
    entry = fw.allocate(physics_state.EntryState, grid, None)

    inputs = tmx_component.collect_input(entry)

    for declaration, field in inputs.leaves():
        assert field is getattr(entry, declaration.name), declaration.name


def test_run_writes_where_the_caller_says(grid: base_grid.Grid) -> None:
    component = _component(grid, RecordingFluxProvider())
    view = fw.allocate(tmx_component.TmxComponent.Output, grid, None)

    out = component.run(_inputs(grid, seed=0), out=view)

    assert out is view
    for declaration, field in view.leaves():
        assert np.all(np.isfinite(field.data.asnumpy())), declaration.name
    assert np.any(view.tend_temperature.data.asnumpy() != 0.0)
    # the component's own buffers were never needed
    assert "output" not in vars(component)


def test_run_hands_the_provider_the_half_level_pressure_of_each_step(
    grid: base_grid.Grid,
) -> None:
    provider = RecordingFluxProvider()
    component = _component(grid, provider)
    first, second = _inputs(grid, seed=1), _inputs(grid, seed=2)

    component.run(first)
    component.run(second)

    assert len(provider.pressures) == 2
    assert provider.pressures[0] is first.pressure_ifc.data
    assert provider.pressures[1] is second.pressure_ifc.data
