# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test of the surface ComponentState adapter: pure input mapping, views of the lowest level."""

import types

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.surface import (
    data as surface_data,
    state as surface_state,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.states import model
from icon4py.model.common.utils import data_allocation as data_alloc


def test_as_component_input_takes_the_lowest_level_as_views():
    grid = simple.simple_grid()
    pressure_ifc = data_alloc.random_field(grid, dims.CellDim, dims.KHalfDim)
    temperature = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
    state = types.SimpleNamespace(
        diagnostics=types.SimpleNamespace(pressure_ifc=pressure_ifc, temperature=temperature)
    )

    inputs = surface_state.State(grid=grid).as_component_input(state)

    assert set(inputs) == set(surface_data.INPUTS_PROPERTIES)
    np.testing.assert_array_equal(
        inputs["surface_pressure"].asnumpy(), pressure_ifc.asnumpy()[:, grid.num_levels]
    )
    np.testing.assert_array_equal(
        inputs["temperature"].asnumpy(), temperature.asnumpy()[:, grid.num_levels - 1]
    )
    # views of the physics state's fields, not copies
    assert np.shares_memory(inputs["surface_pressure"].ndarray, pressure_ifc.ndarray)
    assert np.shares_memory(inputs["temperature"].ndarray, temperature.ndarray)


def test_outputs_are_diagnostics_that_declare_dims():
    """No tendencies, and every output declares dims: the PhysicsState allocates from them."""
    for props in surface_data.OUTPUTS_PROPERTIES.values():
        assert props.kind is not model.FieldKind.TENDENCY
        assert props.dims == (dims.CellDim,)
