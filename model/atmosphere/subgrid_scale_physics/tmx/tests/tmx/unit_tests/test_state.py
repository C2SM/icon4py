# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test of the tmx ComponentState adapter: pure input mapping, no copies."""

import dataclasses
import types

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import (
    data as tmx_data,
    state as tmx_state,
    tmx_states,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.states import model
from icon4py.model.common.states.surface_flux_state import SurfaceFluxState


def _facade(grid, air_mass):
    tracers = types.SimpleNamespace(qv="QV", qc="QC", qi="QI", qr="QR", qs="QS", qg="QG")
    return types.SimpleNamespace(
        diagnostics=types.SimpleNamespace(
            temperature="TA", virtual_temperature="TV", pressure="P", u="U", v="V"
        ),
        w="W",
        rho="RHO",
        tracers=tracers,
        air_mass=air_mass,
        surface=types.SimpleNamespace(fluxes=SurfaceFluxState.allocate(grid)),
    )


def test_as_component_input_maps_the_facade_without_copies():
    grid = simple.simple_grid()
    facade = _facade(grid, air_mass="MAIR")

    inputs = tmx_state.State().as_component_input(facade)

    assert set(inputs) == set(tmx_data.INPUTS_PROPERTIES)
    assert inputs["temperature"] == "TA" and inputs["w"] == "W" and inputs["qg"] == "QG"
    assert inputs["air_mass"] == "MAIR"
    for name in tmx_data.FLUX_PORTS:
        assert inputs[name] is getattr(facade.surface.fluxes, name)  # the surface state's fields


def test_as_component_input_requires_the_air_mass():
    grid = simple.simple_grid()
    with pytest.raises(ValueError, match="air mass"):
        tmx_state.State().as_component_input(_facade(grid, air_mass=None))


def test_inputs_cover_the_tmx_input_state():
    """Every field of the TmxInputState is an input of the component."""
    state_fields = {field.name for field in dataclasses.fields(tmx_states.TmxInputState)}
    assert state_fields <= set(tmx_data.INPUTS_PROPERTIES)


def test_outputs_are_the_tendency_state_plus_km_and_kh():
    tendencies = {field.name for field in dataclasses.fields(tmx_states.TmxTendencyState)}
    declared = {
        name
        for name, props in tmx_data.OUTPUTS_PROPERTIES.items()
        if props.kind is model.FieldKind.TENDENCY
    }
    assert declared == tendencies
    for name in ("km", "kh"):
        props = tmx_data.OUTPUTS_PROPERTIES[name]
        assert props.kind is not model.FieldKind.TENDENCY
        assert props.dims == (dims.CellDim, dims.KDim)
