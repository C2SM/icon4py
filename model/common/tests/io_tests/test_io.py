# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
import pathlib
import re
from typing import Union

import gt4py.next as gtx
import numpy as np
import pytest
import uxarray as ux
import xarray as xr

import icon4py.model.common.exceptions as errors
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, simple, vertical as v_grid
from icon4py.model.common.io import data, ugrid
from icon4py.model.common.io.io import (
    FieldGroupIOConfig,
    FieldGroupMonitor,
    IOConfig,
    IOMonitor,
    generate_name,
    to_delta,
)
from icon4py.model.common.test_utils import datatest_utils, grid_utils, helpers


UNLIMITED = None
simple_grid = simple.SimpleGrid()

grid_file = datatest_utils.GRIDS_PATH.joinpath(
    datatest_utils.R02B04_GLOBAL, grid_utils.GLOBAL_GRIDFILE
)
global_grid = grid_utils.get_icon_grid_from_gridfile(datatest_utils.GLOBAL_EXPERIMENT, on_gpu=False)


def model_state(grid: base.BaseGrid) -> dict[str, xr.DataArray]:
    rho = helpers.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    exner = helpers.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    theta_v = helpers.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    w = helpers.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=np.float32)
    vn = helpers.random_field(grid, dims.EdgeDim, dims.KDim, dtype=np.float32)
    return {
        "air_density": data.to_data_array(rho, data.PROGNOSTIC_CF_ATTRIBUTES["air_density"]),
        "exner_function": data.to_data_array(
            exner, data.PROGNOSTIC_CF_ATTRIBUTES["exner_function"]
        ),
        "theta_v": data.to_data_array(
            theta_v,
            data.PROGNOSTIC_CF_ATTRIBUTES["virtual_potential_temperature"],
            is_on_interface=False,
        ),
        "upward_air_velocity": data.to_data_array(
            w,
            data.PROGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"],
            is_on_interface=True,
        ),
        "normal_velocity": data.to_data_array(
            vn, data.PROGNOSTIC_CF_ATTRIBUTES["normal_velocity"], is_on_interface=False
        ),
    }


def state_values() -> xr.DataArray:
    state = model_state(simple_grid)
    for v in state.values():
        yield v


@pytest.mark.parametrize("num", range(1, 6))
@pytest.mark.parametrize("slot", ["DAY", "day", "Day", "days", "DAyS"])
def test_to_delta_days(num, slot):
    assert to_delta("DAY") == dt.timedelta(days=1)
    assert to_delta(f"{num} {slot}") == dt.timedelta(days=num)


@pytest.mark.parametrize("num", range(1, 5))
@pytest.mark.parametrize("slot", ["HOUR", "hour", "Hour", "hours", "HOURS"])
def test_to_delta_hours(num, slot):
    assert to_delta("HOUR") == dt.timedelta(hours=1)
    assert to_delta(f"{num} {slot}") == dt.timedelta(hours=num)


@pytest.mark.parametrize("num", [0, 2, 44, 4, 5])
@pytest.mark.parametrize("slot", ["second", "SECOND", "SEConds", "SECONDS"])
def test_to_delta_secs(num, slot):
    assert to_delta(f"{num} {slot}") == dt.timedelta(seconds=num)


@pytest.mark.parametrize("num", [0, 2, 3, 4, 5])
@pytest.mark.parametrize("slot", ["MINUTE", "Minute", "minutes", "MINUTES"])
def test_to_delta_mins(num, slot):
    assert to_delta(f"{num} {slot}") == dt.timedelta(minutes=num)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("output.nc", "output_0002.nc"),
        ("outxxput_20220101.xc", "outxxput_20220101_0002.nc"),
        ("output_20220101T000000_x", "output_20220101T000000_x_0002.nc"),
    ],
)
def test_generate_name(name, expected):
    counter = 2
    assert expected == generate_name(name, counter)


def is_valid_uxgrid(file: Union[pathlib.Path, str]) -> bool:
    import uxarray as ux

    grid = ux.open_grid(file)
    try:
        grid.validate()
        return True
    except RuntimeError:
        return False


def test_io_monitor_create_output_path(test_path):
    path_name = test_path.absolute().as_posix() + "/output"
    vertical_config = v_grid.VerticalGridConfig(num_levels=simple_grid.num_levels)
    vertical_params = v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=gtx.as_field((dims.KDim,), np.linspace(12000.0, 0.0, simple_grid.num_levels + 1)),
        vct_b=None,
    )
    config = IOConfig(field_groups=[], output_path=path_name)
    monitor = IOMonitor(
        config,
        vertical_params,
        simple_grid.config.horizontal_config,
        grid_file,
        simple_grid.id,
    )
    assert monitor.path.exists()
    assert monitor.path.is_dir()


def test_io_monitor_write_ugrid_file(test_path):
    path_name = test_path.absolute().as_posix() + "/output"
    vertical_config = v_grid.VerticalGridConfig(num_levels=simple_grid.num_levels)
    vertical_params = v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=gtx.as_field((dims.KDim,), np.linspace(12000.0, 0.0, simple_grid.num_levels + 1)),
        vct_b=None,
    )

    config = IOConfig(field_groups=[], output_path=path_name)
    monitor = IOMonitor(
        config,
        vertical_params,
        simple_grid.config.horizontal_config,
        grid_file,
        "simple_grid",
    )
    ugrid_file = monitor.path.iterdir().__next__().absolute()
    assert "ugrid.nc" in ugrid_file.name
    assert is_valid_uxgrid(ugrid_file)


@pytest.mark.parametrize(
    "variables",
    (
        ["air_density", "exner_function", "upward_air_velocity"],
        ["normal_velocity", "upward_air_velocity", "theta_v"],
    ),
)
def test_io_monitor_write_and_read_ugrid_dataset(test_path, variables):
    path_name = test_path.absolute().as_posix() + "/output"
    grid = grid_utils.get_icon_grid_from_gridfile(datatest_utils.GLOBAL_EXPERIMENT, on_gpu=False)
    vertical_config = v_grid.VerticalGridConfig(num_levels=grid.num_levels)
    vertical_params = v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=gtx.as_field((dims.KDim,), np.linspace(12000.0, 0.0, grid.num_levels + 1)),
        vct_b=None,
    )

    state = model_state(grid)
    configured_output_start = "2024-01-01T12:00:00"
    field_configs = [
        FieldGroupIOConfig(
            output_interval="HOUR",
            start_time=configured_output_start,
            filename="icon4py_dummy_output",
            variables=variables,
            nc_comment="Writing dummy data from icon4py for testing.",
        )
    ]
    config = IOConfig(field_groups=field_configs, output_path=path_name)
    monitor = IOMonitor(
        config,
        vertical_params,
        grid.config.horizontal_config,
        grid_file,
        grid.id,
    )
    start_time = dt.datetime.fromisoformat(configured_output_start)
    monitor.store(state, start_time)
    time = start_time + dt.timedelta(minutes=30)
    monitor.store(state, time)
    time = time + dt.timedelta(minutes=30)
    monitor.store(state, time)
    time = time + dt.timedelta(minutes=60)
    monitor.store(state, time)
    monitor.close()

    assert len([f for f in monitor.path.iterdir() if f.is_file()]) == 1 + len(field_configs)
    uxds = read_back_as_uxarray(monitor.path)
    for var in variables:
        assert var in uxds.variables
        if var in ["air_density", "exner_function", "theta_v"]:
            assert uxds[var].shape == (3, grid.num_levels, grid.num_cells)
        elif var == "upward_air_velocity":
            assert uxds[var].shape == (3, grid.num_levels + 1, grid.num_cells)
        elif var == "normal_velocity":
            assert uxds[var].shape == (3, grid.num_levels, grid.num_edges)


def test_fieldgroup_monitor_write_dataset_file_roll(test_path):
    grid = grid_utils.get_icon_grid_from_gridfile(datatest_utils.GLOBAL_EXPERIMENT, on_gpu=False)
    vertical_config = v_grid.VerticalGridConfig(num_levels=grid.num_levels)
    vertical_params = v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=gtx.as_field((dims.KDim,), np.linspace(12000.0, 0.0, grid.num_levels + 1)),
        vct_b=None,
    )

    state = model_state(grid)
    configured_output_start = "2024-01-01T12:00:00"
    filename_stub = "icon4py_dummy_output"
    config = FieldGroupIOConfig(
        output_interval="HOUR",
        start_time=configured_output_start,
        filename=filename_stub,
        variables=["air_density", "exner_function", "upward_air_velocity"],
        timesteps_per_file=1,
    )
    monitor = FieldGroupMonitor(
        config,
        vertical=vertical_params,
        horizontal=grid.config.horizontal_config,
        grid_id=grid.id,
        output_path=test_path,
    )
    time = dt.datetime.fromisoformat(configured_output_start)
    for _ in range(4):
        monitor.store(state, time)
        time = time + dt.timedelta(hours=1)
    assert len([f for f in monitor.output_path.iterdir() if f.is_file()]) == 4
    expected_name = re.compile(filename_stub + "_\\d{4}.nc")
    for f in monitor.output_path.iterdir():
        if f.is_file():
            assert expected_name.match(f.name)

            with ugrid.load_data_file(f) as ds:
                assert ds.sizes["time"] == 1
                assert ds.sizes["level"] == grid.num_levels
                assert ds.sizes["cell"] == grid.num_cells
                assert ds.sizes["interface_level"] == grid.num_levels + 1
                assert ds.variables["air_density"].shape == (
                    1,
                    grid.num_levels,
                    grid.num_cells,
                )
                assert ds.variables["exner_function"].shape == (
                    1,
                    grid.num_levels,
                    grid.num_cells,
                )
                assert ds.variables["upward_air_velocity"].shape == (
                    1,
                    grid.num_levels + 1,
                    grid.num_cells,
                )


def read_back_as_uxarray(path: pathlib.Path):
    ugrid_file = None
    data_files = []
    for f in path.iterdir():
        if f.is_file():
            if "_ugrid.nc" in f.name:
                ugrid_file = f.absolute()
            else:
                data_files.append(f.absolute())
    uxds = ux.open_dataset(ugrid_file, data_files[0])
    return uxds


def test_fieldgroup_monitor_output_time_updates_upon_store(test_path):
    config, group_monitor = create_field_group_monitor(test_path, simple_grid)
    configured_start_time = dt.datetime.fromisoformat(config.start_time)
    step_time = configured_start_time
    state = model_state(simple_grid)
    group_monitor.store(state, step_time)
    assert group_monitor.next_output_time > configured_start_time
    one_hour_later = step_time + dt.timedelta(hours=1)
    assert group_monitor.next_output_time == one_hour_later


def test_fieldgroup_monitor_no_output_on_not_matching_time(test_path):
    start_time_str = "2024-01-01T00:00:00"
    config, group_monitor = create_field_group_monitor(test_path, simple_grid, start_time_str)
    start_time = dt.datetime.fromisoformat(config.start_time)
    step_time = start_time
    state = model_state(simple_grid)
    group_monitor.store(state, step_time)
    assert group_monitor.next_output_time > start_time
    one_hour_later = step_time + dt.timedelta(hours=1)
    assert group_monitor.next_output_time == one_hour_later
    ten_minutes_later = step_time + dt.timedelta(minutes=10)
    group_monitor.store(state, ten_minutes_later)
    assert group_monitor.next_output_time == one_hour_later


def test_fieldgroup_monitor_output_time_initialized_from_config(test_path):
    configured_start_time = "2024-01-01T00:00:00"
    config, group_monitor = create_field_group_monitor(
        test_path, simple_grid, configured_start_time
    )
    assert group_monitor.next_output_time == dt.datetime.fromisoformat(configured_start_time)


def test_fieldgroup_monitor_no_output_before_start_time(test_path):
    configured_start_time = "2024-01-01T00:00:00"
    start_time = dt.datetime.fromisoformat(configured_start_time)
    config, group_monitor = create_field_group_monitor(
        test_path, simple_grid, configured_start_time
    )
    step_time = dt.datetime.fromisoformat("2023-12-12T00:12:00")
    assert start_time > step_time
    group_monitor.store(model_state(simple_grid), step_time)
    assert group_monitor.next_output_time == start_time
    group_monitor.close()
    assert len([f for f in group_monitor.output_path.iterdir() if f.is_file()]) == 0


def create_field_group_monitor(test_path, grid, start_time="2024-01-01T00:00:00"):
    config = FieldGroupIOConfig(
        start_time=start_time,
        filename="test_empty.nc",
        output_interval="1 HOUR",
        variables=["exner_function", "air_density"],
    )
    vertical_config = v_grid.VerticalGridConfig(num_levels=simple_grid.num_levels)
    vertical_params = v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=gtx.as_field((dims.KDim,), np.linspace(12000.0, 0.0, simple_grid.num_levels + 1)),
        vct_b=None,
    )

    group_monitor = FieldGroupMonitor(
        config,
        vertical=vertical_params,
        horizontal=grid.config.horizontal_config,
        grid_id=grid.id,
        output_path=test_path,
    )
    return config, group_monitor


@pytest.mark.parametrize(
    "start_time, filename, interval, variables, message",
    [
        (
            "2023-04-04T11:00:00",
            "",
            "1 HOUR",
            ["exner_function", "air_density"],
            "Output filename is missing.",
        ),
        (
            "2023-04-04T11:00:00",
            "/vars/prognostics.nc",
            "1 HOUR",
            ["exner_function", "air_density"],
            "absolute path",
        ),
        (
            "2023-04-04T11:00:00",
            "vars/prognostics.nc",
            "1 HOUR",
            [],
            "No variables provided for output.",
        ),
        (
            "2023-04-04T11:00:00",
            "vars/prognostics.nc",
            "",
            ["air_density, exner_function"],
            "No output interval provided.",
        ),
    ],
)
def test_fieldgroup_config_validate_filename(start_time, filename, interval, variables, message):
    with pytest.raises(errors.InvalidConfigError) as err:
        FieldGroupIOConfig(
            start_time=start_time,
            filename=filename,
            output_interval=interval,
            variables=variables,
        )
    assert message in str(err.value)


def test_fieldgroup_monitor_constructs_output_path_and_filepattern(test_path):
    config = FieldGroupIOConfig(
        start_time="2023-04-04T11:00:00",
        filename="vars/prognostics.nc",
        output_interval="1 HOUR",
        variables=["exner_function", "air_density"],
    )
    vertical_size = simple_grid.config.vertical_size
    horizontal_size = simple_grid.config.horizontal_config
    group_monitor = FieldGroupMonitor(
        config,
        vertical=vertical_size,
        horizontal=horizontal_size,
        grid_id=simple_grid.id,
        output_path=test_path,
    )
    assert group_monitor.output_path == test_path.joinpath("vars")
    assert group_monitor.output_path.exists()
    assert group_monitor.output_path.is_dir()
    assert "prognostics" in group_monitor._file_name_pattern


def test_fieldgroup_monitor_throw_exception_on_missing_field(test_path):
    config = FieldGroupIOConfig(
        start_time="2023-04-04T11:00:00",
        filename="vars/prognostics.nc",
        output_interval="1 HOUR",
        variables=["exner_function", "air_density", "foo"],
    )
    vertical_size = simple_grid.config.vertical_size
    horizontal_size = simple_grid.config.horizontal_config
    group_monitor = FieldGroupMonitor(
        config,
        vertical=vertical_size,
        horizontal=horizontal_size,
        grid_id=simple_grid.id,
        output_path=test_path,
    )
    with pytest.raises(errors.IncompleteStateError, match="Field 'foo' is missing in state"):
        group_monitor.store(
            model_state(simple_grid), dt.datetime.fromisoformat("2023-04-04T11:00:00")
        )
