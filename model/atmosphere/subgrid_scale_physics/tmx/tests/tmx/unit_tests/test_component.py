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
    tmx_states,
)
from icon4py.model.common import dimension as dims, framework as fw, quantity
from icon4py.model.common.grid import simple
from icon4py.model.common.states import surface_flux_state


_FLUXES = tuple(field.name for field in dataclasses.fields(surface_flux_state.SurfaceFluxState))


def _physics_state(**kwargs: object) -> physics_state.PhysicsState:
    return fw.allocate(
        physics_state.PhysicsState,
        simple.simple_grid(),
        None,
        step_start=datetime.datetime(2008, 9, 1),
        **kwargs,
    )


def test_collect_input_views_the_physics_state() -> None:
    state = _physics_state()

    input = tmx_component.collect_input(state)

    for name in fw.fields(input):
        assert getattr(input, name) is getattr(state, name), name


def test_collect_input_requires_the_air_mass() -> None:
    state = _physics_state(only=_FLUXES)
    with pytest.raises(ValueError, match="air_mass"):
        tmx_component.collect_input(state)


def test_collect_input_requires_the_surface_fluxes() -> None:
    state = _physics_state(only=("air_mass",))
    with pytest.raises(ValueError, match="evapotranspiration"):
        tmx_component.collect_input(state)


def test_input_covers_the_tmx_input_state_and_the_surface_fluxes() -> None:
    input = fw.allocate(tmx_component.Tmx.Input, simple.simple_grid(), None)
    expected = {field.name for field in dataclasses.fields(tmx_states.TmxInputState)}
    assert set(fw.fields(input)) == expected | set(_FLUXES)


def test_output_is_the_tendency_state_plus_km_and_kh() -> None:
    output = fw.allocate(tmx_component.Tmx.Output, simple.simple_grid(), None)
    tendencies = {field.name for field in dataclasses.fields(tmx_states.TmxTendencyState)}
    assert set(fw.fields(output, kind=quantity.Tendency)) == tendencies
    for leaf in (output.km, output.kh):
        assert not issubclass(leaf.quantity, quantity.Tendency)
        assert leaf.quantity.dims == (dims.CellDim, dims.KDim)
