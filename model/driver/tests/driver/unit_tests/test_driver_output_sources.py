# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


"""Unit tests for the output sources of the model state (``driver_output_sources``).

Data-free: the ``simple_grid`` and synthetic states, so no serialized/grid test data.
"""

import gt4py.next as gtx
import numpy as np
import pytest
import xarray as xr

from icon4py.model.common import dimension as dims, field, framework as fw
from icon4py.model.common.framework import quantities as qty, states
from icon4py.model.common.grid import base, simple
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver_output_sources
from icon4py.model.testing.fixtures import backend


@pytest.fixture
def grid() -> base.Grid:
    return simple.simple_grid()


@pytest.fixture
def prognostic_state(grid: base.Grid) -> states.PrognosticState:
    return fw.allocate(states.PrognosticState, grid, None)


#: The prognostic output variables with the dims of their fields, stated independently of
#: the production code so the assertions stay a genuine check. A field on interface levels
#: is expected on ``KHalfDim``, which is what decides the netCDF vertical dimension name.
_EXPECTED_DIMS: dict[str, tuple[gtx.Dimension, gtx.Dimension]] = {
    "air_density": (dims.CellDim, dims.KDim),
    "dimensionless_exner_function": (dims.CellDim, dims.KDim),
    "virtual_potential_temperature": (dims.CellDim, dims.KDim),
    "upward_air_velocity": (dims.CellDim, dims.KHalfDim),
    "normal_velocity": (dims.EdgeDim, dims.KDim),
}

#: UGRID dimension names of the horizontal dimensions.
_UGRID_DIM_NAMES: dict[gtx.Dimension, str] = {
    dims.CellDim: "cell",
    dims.EdgeDim: "edge",
    dims.VertexDim: "vertex",
}


def _horizontal_size(grid: base.Grid, dim: gtx.Dimension) -> int:
    return {
        dims.CellDim: grid.num_cells,
        dims.EdgeDim: grid.num_edges,
        dims.VertexDim: grid.num_vertices,
    }[dim]


def test_assembles_all_prognostic_variables(
    prognostic_state: states.PrognosticState, grid: base.Grid
) -> None:
    state = driver_output_sources.to_dataarrays(
        driver_output_sources.by_standard_name(prognostic_state)
    )

    assert set(state.keys()) == set(driver_output_sources.PROGNOSTIC_VARIABLES)
    for name, da in state.items():
        assert isinstance(da, xr.DataArray)
        horizontal_dim, vertical_dim = _EXPECTED_DIMS[name]
        on_half_levels = vertical_dim == dims.KHalfDim

        vertical_name = "half_level" if on_half_levels else "level"
        assert da.dims == (_UGRID_DIM_NAMES[horizontal_dim], vertical_name)

        vertical_size = grid.num_levels + 1 if on_half_levels else grid.num_levels
        assert da.shape == (_horizontal_size(grid, horizontal_dim), vertical_size)


def test_dataarrays_carry_cf_and_ugrid_metadata(
    prognostic_state: states.PrognosticState,
) -> None:
    state = driver_output_sources.to_dataarrays(
        driver_output_sources.by_standard_name(prognostic_state)
    )

    air_density = state["air_density"]
    # CF metadata from the quantity
    assert air_density.attrs["standard_name"] == qty.RhoOnCellK.standard_name
    assert air_density.attrs["long_name"] == qty.RhoOnCellK.long_name
    assert air_density.attrs["units"] == qty.RhoOnCellK.units
    assert "icon_var_name" not in air_density.attrs
    # UGRID metadata added by io.utils.to_data_array for the horizontal dimension
    assert air_density.attrs["location"] == "face"
    assert air_density.attrs["mesh"] == "mesh"
    assert air_density.attrs["coordinates"] == "clon clat"

    # edge field gets the edge location mapping
    assert state["normal_velocity"].attrs["location"] == "edge"


def test_a_leaf_without_a_standard_name_is_not_offered() -> None:
    class _State(fw.State):
        rho: field.Field[qty.RhoOnCellK]
        rho_ic: field.Field[qty.RhoOnCellKHalf]

    state = fw.allocate(_State, simple.simple_grid(), None)

    assert qty.RhoOnCellKHalf.standard_name is None
    assert driver_output_sources.standard_names(_State) == {"rho": "air_density"}
    assert set(driver_output_sources.by_standard_name(state)) == {"air_density"}


def test_data_is_host_numpy(
    grid: base.Grid,
    backend: gtx.typing.Backend | None,
) -> None:
    """The buffer handed to the writers must be a host numpy array, not a device array.

    Parameterized on the backend (``--backend``) so that with a GPU backend the inputs
    really are device buffers and the host transfer is exercised.
    """
    prognostic_state = fw.allocate(states.PrognosticState, grid, backend)
    state = driver_output_sources.to_dataarrays(
        driver_output_sources.by_standard_name(prognostic_state)
    )
    for da in state.values():
        assert isinstance(da.data, np.ndarray)


def test_the_tracer_variables_are_the_active_tracers() -> None:
    config = states.TracerConfig(qv=True, qr=True)
    assert driver_output_sources.tracer_variables(config) == (
        "specific_humidity",
        "specific_rain_content",
    )
    assert driver_output_sources.tracer_variables(None) == ()


def test_only_the_active_tracers_are_assembled(
    grid: base.Grid, backend: gtx.typing.Backend | None
) -> None:
    tracer_state = states.TracerState(
        qv=field.Field(
            qty.QvOnCellK, data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        )
    )

    state = driver_output_sources.to_dataarrays(
        driver_output_sources.by_standard_name(tracer_state)
    )

    assert set(state) == {"specific_humidity"}
    assert state["specific_humidity"].dims == ("cell", "level")
    assert state["specific_humidity"].attrs["location"] == "face"
    # host numpy, also from a device buffer
    assert isinstance(state["specific_humidity"].data, np.ndarray)
