# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests of the driver's IO component (``driver_io.IOMonitor``).

Data-free: the ``simple_grid``, zero states and a stub writer in place of the common one.
"""

import datetime
import pathlib
import uuid
from typing import Any, cast

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest
import xarray as xr
from gt4py.next import backend as gtx_backend

from icon4py.model.common.components import framework as fw, quantities as qty, states
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import base, simple
from icon4py.model.common.io import io as common_io, writers
from icon4py.model.driver import driver_io

from ..fixtures import *  # noqa: F403


START = datetime.datetime(2000, 1, 1)

# the output variable of each Input leaf, as the output files name them
EXPECTED_LEAVES: dict[str, str] = {
    "air_density": "rho",
    "exner_function": "exner",
    "virtual_potential_temperature": "theta_v",
    "upward_air_velocity": "w",
    "normal_velocity": "vn",
    "eastward_wind": "u",
    "northward_wind": "v",
    "temperature": "temperature",
    "virtual_temperature": "virtual_temperature",
    "pressure": "pressure",
}

# (UGRID horizontal dimension, vertical dimension) of each output variable
EXPECTED_DIMS: dict[str, tuple[str, str]] = {
    name: ("cell", "level") for name in EXPECTED_LEAVES
} | {"upward_air_velocity": ("cell", "half_level"), "normal_velocity": ("edge", "level")}


class StubWriter:
    """The writer's `at_capture_time`/`store` contract: `store` advances the schedule."""

    def __init__(self, captures: list[bool] | None = None) -> None:
        self.captures = captures
        self.stored: list[tuple[dict[str, xr.DataArray], datetime.datetime]] = []

    def at_capture_time(self) -> bool:
        return True if self.captures is None else self.captures[len(self.stored)]

    def store(self, state: dict[str, xr.DataArray], model_time: datetime.datetime) -> None:
        self.stored.append((dict(state), model_time))


class DuplicatedIOMonitor(driver_io.IOMonitor):
    """Two leaves of one quantity: both would be written as `air_density`."""

    class Input(driver_io.IOMonitor.Input):
        rho_again: fw.Field[qty.RhoOnCellK]


def make_monitor(
    grid: base.Grid, writer: StubWriter, variables: list[str] | None = None
) -> driver_io.IOMonitor:
    return driver_io.IOMonitor(
        grid=grid,
        writer=cast(common_io.IOMonitor, writer),
        variables=driver_io.DEFAULT_OUTPUT_VARIABLES if variables is None else variables,
    )


def make_inputs(
    grid: base.Grid,
    allocator: gtx_typing.Allocator | None = None,
    simulation_time: datetime.datetime = START,
) -> driver_io.IOMonitor.Input:
    prognostics = fw.allocate(states.PrognosticState, grid, allocator)
    diagnostics = fw.allocate(states.Diagnostics, grid, allocator)
    return driver_io.IOMonitor.Input(
        rho=prognostics.rho,
        w=prognostics.w,
        vn=prognostics.vn,
        exner=prognostics.exner,
        theta_v=prognostics.theta_v,
        temperature=diagnostics.temperature,
        virtual_temperature=diagnostics.virtual_temperature,
        pressure=diagnostics.pressure,
        u=diagnostics.u,
        v=diagnostics.v,
        simulation_time=simulation_time,
    )


def stored_state(grid: base.Grid, allocator: gtx_typing.Allocator | None = None) -> Any:
    writer = StubWriter()
    make_monitor(grid, writer).run(make_inputs(grid, allocator))
    ((state, _),) = writer.stored
    return state


@pytest.fixture
def grid() -> base.Grid:
    return simple.simple_grid()


def test_selects_each_variable_from_its_leaf(grid: base.Grid) -> None:
    monitor = make_monitor(grid, StubWriter())

    assert monitor.selected == EXPECTED_LEAVES


def test_default_variables_are_every_variable_in_file_order() -> None:
    assert list(EXPECTED_LEAVES) == driver_io.DEFAULT_OUTPUT_VARIABLES


def test_variables_subset(grid: base.Grid) -> None:
    writer = StubWriter()
    monitor = make_monitor(grid, writer, ["air_density", "normal_velocity"])

    monitor.run(make_inputs(grid))

    assert monitor.selected == {"air_density": "rho", "normal_velocity": "vn"}
    assert set(writer.stored[0][0]) == {"air_density", "normal_velocity"}


def test_unknown_variable_raises(grid: base.Grid) -> None:
    with pytest.raises(ValueError, match=r"Unknown output variable 'not_a_field'.*air_density"):
        make_monitor(grid, StubWriter(), ["air_density", "not_a_field"])


def test_two_leaves_with_one_output_name_are_rejected(grid: base.Grid) -> None:
    with pytest.raises(AssertionError, match=r"'rho' and 'rho_again'.*'air_density'"):
        DuplicatedIOMonitor(
            grid=grid, writer=cast(common_io.IOMonitor, StubWriter()), variables=["air_density"]
        )


def test_dataarrays_carry_the_dims_of_their_leaf(grid: base.Grid) -> None:
    state = stored_state(grid)

    sizes = {"cell": grid.num_cells, "edge": grid.num_edges}
    levels = {"level": grid.num_levels, "half_level": grid.num_levels + 1}
    assert set(state) == set(EXPECTED_LEAVES)
    for name, (horizontal, vertical) in EXPECTED_DIMS.items():
        assert state[name].dims == (horizontal, vertical)
        assert state[name].shape == (sizes[horizontal], levels[vertical])


def test_written_attributes_are_those_of_the_leaf_quantity(grid: base.Grid) -> None:
    """The file attributes are the quantity's CF attributes plus the UGRID association."""
    inputs = make_inputs(grid)
    state = stored_state(grid)

    for name, leaf in EXPECTED_LEAVES.items():
        quantity = getattr(inputs, leaf).quantity
        attrs = writers.data_variable_attributes(state[name])
        assert set(state[name].attrs) == {*writers.DATA_VARIABLE_ATTRIBUTES}
        assert (attrs["standard_name"], attrs["long_name"], attrs["units"]) == (
            quantity.standard_name,
            quantity.long_name,
            quantity.units,
        )
    # the file name and the CF standard_name of a variable may differ
    assert state["exner_function"].attrs["standard_name"] == "dimensionless_exner_function"


def test_data_is_host_numpy(grid: base.Grid, backend: gtx_backend.Backend[Any] | None) -> None:
    """With a GPU backend the leaves are device buffers and the host transfer is exercised."""
    state = stored_state(grid, allocator=backend)

    for da in state.values():
        assert isinstance(da.data, np.ndarray)


def test_run_stores_every_call_and_assembles_at_capture_time_only(grid: base.Grid) -> None:
    writer = StubWriter(captures=[True, False, True])
    monitor = make_monitor(grid, writer)
    times = [START + datetime.timedelta(seconds=s) for s in (0, 10, 20)]

    for t in times:
        out = monitor.run(make_inputs(grid, simulation_time=t))
        assert isinstance(out, fw.Empty)

    assert [t for _, t in writer.stored] == times
    assert [set(state) for state, _ in writer.stored] == [
        set(EXPECTED_LEAVES),
        set(),
        set(EXPECTED_LEAVES),
    ]


def test_stored_data_is_the_leaf_at_store_time(grid: base.Grid) -> None:
    writer = StubWriter()
    inputs = make_inputs(grid)
    rho: Any = inputs.rho.data.ndarray
    rho[...] = 1.5

    make_monitor(grid, writer).run(inputs)

    np.testing.assert_array_equal(writer.stored[0][0]["air_density"].data, rho)


def test_create_io_monitor_builds_one_field_group_with_every_variable(
    grid: base.Grid, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common writer is replaced by a recorder so the test needs no real grid file."""
    recorded: dict[str, Any] = {}

    class RecordingWriter:
        def __init__(self, **kwargs: Any) -> None:
            recorded.update(kwargs)

    monkeypatch.setattr(common_io, "IOMonitor", RecordingWriter)

    monitor = driver_io.create_io_monitor(
        output_path=tmp_path,
        grid_file_path=tmp_path / "grid.nc",
        grid=grid,
        vertical_grid=None,  # type: ignore[arg-type] # not used by the recorder
        dtime=datetime.timedelta(seconds=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
        decomposition_info=None,
    )

    assert isinstance(monitor, driver_io.IOMonitor)
    assert isinstance(monitor.writer, RecordingWriter)
    assert monitor.selected == EXPECTED_LEAVES
    config = recorded["config"]
    assert isinstance(config, common_io.IOConfig)
    assert config.output_path == str(tmp_path)
    (field_group,) = config.field_groups
    assert list(field_group.variables) == driver_io.DEFAULT_OUTPUT_VARIABLES
    assert field_group.output_interval == 1
    assert field_group.backend == common_io.OutputBackend.ZARR
    assert field_group.mode == common_io.OutputMode.DISTRIBUTED
    assert field_group.basename == driver_io.DEFAULT_OUTPUT_BASENAME
    # the string grid id is converted to a UUID at the IO boundary
    assert recorded["grid_id"] == uuid.UUID(grid.id)
    assert recorded["grid_file_name"] == tmp_path / "grid.nc"
