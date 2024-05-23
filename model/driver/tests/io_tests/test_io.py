# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import pytest
import uxarray as ux
import xarray as xr
from gt4py.next.ffront.fbuiltins import float32

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid.base import BaseGrid
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.common.test_utils.datatest_utils import (
    GLOBAL_EXPERIMENT,
    GRIDS_PATH,
    R02B04_GLOBAL,
)
from icon4py.model.common.test_utils.grid_utils import GLOBAL_GRIDFILE, get_icon_grid_from_gridfile
from icon4py.model.common.test_utils.helpers import random_field
from icon4py.model.driver.io.data import (
    PROGNOSTIC_CF_ATTRIBUTES,
    to_data_array,
)
from icon4py.model.driver.io.io import (
    FieldGroupIoConfig,
    FieldGroupMonitor,
    InvalidConfigError,
    IoConfig,
    IoMonitor,
    generate_name,
    to_delta,
)
from icon4py.model.driver.io.ugrid import load_data_file


UNLIMITED = None
simple_grid = SimpleGrid()

grid_file = GRIDS_PATH.joinpath(R02B04_GLOBAL, GLOBAL_GRIDFILE)
global_grid = get_icon_grid_from_gridfile(GLOBAL_EXPERIMENT, on_gpu=False)


def model_state(grid: BaseGrid) -> dict[str, xr.DataArray]:
    rho = random_field(grid, CellDim, KDim, dtype=float32)
    exner = random_field(grid, CellDim, KDim, dtype=float32)
    theta_v = random_field(grid, CellDim, KDim, dtype=float32)
    w = random_field(grid, CellDim, KDim, extend={KDim: 1}, dtype=float32)
    vn = random_field(grid, EdgeDim, KDim , dtype=float32)
    return {
        "air_density": to_data_array(rho, PROGNOSTIC_CF_ATTRIBUTES["air_density"]),
        "exner_function": to_data_array(exner, PROGNOSTIC_CF_ATTRIBUTES["exner_function"]),
        "theta_v": to_data_array(
            theta_v,
            PROGNOSTIC_CF_ATTRIBUTES["virtual_potential_temperature"],
            is_on_interface=False,
        ),
        "upward_air_velocity": to_data_array(
            w, PROGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"], is_on_interface=True
        ),
        "normal_velocity": to_data_array(
            vn, PROGNOSTIC_CF_ATTRIBUTES["normal_velocity"], is_on_interface=False
        ),
    }


def state_values() -> xr.DataArray:
    state = model_state(SimpleGrid())
    for v in state.values():
        yield v


@pytest.mark.parametrize("num", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("slot", ["DAY", "day", "Day", "days", "DAyS"])
def test_to_delta_hrs(num, slot):
    assert to_delta("DAY") == timedelta(hours=1)
    assert to_delta(f"{num} {slot}") == timedelta(hours=num)

@pytest.mark.parametrize("num", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("slot", ["HOUR", "hour", "Hour", "hours", "HOURS"])
def test_to_delta_hrs(num, slot):
    assert to_delta("HOUR") == timedelta(hours=1)
    assert to_delta(f"{num} {slot}") == timedelta(hours=num)


@pytest.mark.parametrize("num", [0, 2, 44, 4, 5])
@pytest.mark.parametrize("slot", ["second", "SECOND", "seconds", "SECONDS"])
def test_to_delta_secs(num, slot):
    assert to_delta(f"{num} {slot}") == timedelta(seconds=num)


@pytest.mark.parametrize("num", [0, 2, 3, 4, 5])
@pytest.mark.parametrize("slot", ["MINUTE", "Minute", "minutes", "MINUTES"])
def test_to_delta_mins(num, slot):
    assert to_delta(f"{num} {slot}") == timedelta(minutes=num)


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


def is_valid_uxgrid(file: Union[Path, str]) -> bool:
    import uxarray as ux

    grid = ux.open_grid(file)
    try:
        grid.validate()
        return True
    except RuntimeError:
        return False


def test_io_monitor_create_output_path(test_path):
    path_name = test_path.absolute().as_posix() + "/output"
    config = IoConfig(field_groups=[], output_path=path_name)
    monitor = IoMonitor(
        config,
        VerticalGridSize(10),
        SimpleGrid().config.horizontal_config,
        grid_file,
        "simple_grid",
    )
    assert monitor.path.exists()
    assert monitor.path.is_dir()


def test_io_monitor_write_ugrid_file(test_path):
    path_name = test_path.absolute().as_posix() + "/output"
    config = IoConfig(field_groups=[], output_path=path_name)
    monitor = IoMonitor(
        config,
        VerticalGridSize(10),
        SimpleGrid().config.horizontal_config,
        grid_file,
        "simple_grid",
    )
    ugrid_file = monitor.path.iterdir().__next__().absolute()
    assert "ugrid.nc" in ugrid_file.name
    assert is_valid_uxgrid(ugrid_file)


def test_io_monitor_write_dataset(test_path):
    path_name = test_path.absolute().as_posix() + "/output"
    grid, grid_id = get_icon_grid_from_gridfile(GLOBAL_EXPERIMENT, on_gpu=False)
    state = model_state(grid)
    configured_output_start = "2024-01-01T12:00:00"
    field_configs = [
        FieldGroupIoConfig(
            output_interval="HOUR",
            start_time=configured_output_start,
            filename="icon4py_dummy_output",
            variables=["air_density", "exner_function", "upward_air_velocity"],
            nc_comment="Writing dummy data from icon4py for testing.",
        )
    ]
    config = IoConfig(field_groups=field_configs, output_path=path_name)
    monitor = IoMonitor(
        config,
        VerticalGridSize(grid.num_levels),
        grid.config.horizontal_config,
        grid_file,
        str(grid_id),
    )
    start_time = datetime.fromisoformat(configured_output_start)
    monitor.store(state, start_time)
    time = start_time + timedelta(minutes=30)
    monitor.store(state, time)
    time = time + timedelta(minutes=30)
    monitor.store(state, time)
    time = time + timedelta(minutes=60)
    monitor.store(state, time)
    monitor.close()

    assert len([f for f in monitor.path.iterdir() if f.is_file()]) == 1 + len(field_configs)
    uxds = read_back_as_uxarray(monitor.path.iterdir())
    assert uxds["air_density"].shape == (3, grid.num_levels, grid.num_cells)
    assert uxds["exner_function"].shape == (3, grid.num_levels, grid.num_cells)
    assert uxds["upward_air_velocity"].shape == (3, grid.num_levels + 1, grid.num_cells)


def test_fieldgroup_monitor_write_dataset_file_roll(test_path):
    grid, grid_id = get_icon_grid_from_gridfile(GLOBAL_EXPERIMENT, on_gpu=False)
    state = model_state(grid)
    configured_output_start = "2024-01-01T12:00:00"
    filename_stub = "icon4py_dummy_output"
    config = FieldGroupIoConfig(
        output_interval="HOUR",
        start_time=configured_output_start,
        filename=filename_stub,
        variables=["air_density", "exner_function", "upward_air_velocity"],
        timesteps_per_file=1,
    )
    monitor = FieldGroupMonitor(
        config,
        vertical=VerticalGridSize(grid.num_levels),
        horizontal=grid.config.horizontal_config,
        grid_id=str(grid_id),
        output_path=test_path,
    )
    time = datetime.fromisoformat(configured_output_start)
    for _ in range(4):
        monitor.store(state, time)
        time = time + timedelta(hours=1)
    assert len([f for f in monitor.output_path.iterdir() if f.is_file()]) == 4
    expected_name = re.compile(filename_stub + "_\d{4}.nc")
    for f in monitor.output_path.iterdir():
        if f.is_file():
            assert expected_name.match(f.name)
            with load_data_file(f) as ds:
                assert ds.sizes["time"] == 1
                assert ds.sizes["level"] == grid.num_levels
                assert ds.sizes["cell"] == grid.num_cells
                assert ds.sizes["interface_level"] == grid.num_levels + 1
                assert ds.variables["air_density"].shape == (1, grid.num_levels, grid.num_cells)
                assert ds.variables["exner_function"].shape == (1, grid.num_levels, grid.num_cells)
                assert ds.variables["upward_air_velocity"].shape == (
                    1,
                    grid.num_levels + 1,
                    grid.num_cells,
                )


def read_back_as_uxarray(path: Path):
    ugrid_file = None
    data_files = []
    for f in path:
        if f.is_file():
            if "_ugrid.nc" in f.name:
                ugrid_file = f.absolute()
            else:
                data_files.append(f.absolute())
    uxds = ux.open_dataset(ugrid_file, data_files[0])
    return uxds


def test_fieldgroup_monitor_output_time_updates_upon_store(test_path):
    grid = SimpleGrid()
    config = FieldGroupIoConfig(
        start_time="2022-01-01T00:00:00",
        filename="{base_name}_{time}.nc",
        output_interval="1 HOUR",
        variables=["exner_function", "air_density"],
    )
    step_time = datetime.fromisoformat(config.start_time)
    vertical_size = VerticalGridSize(10)
    horizontal_size = SimpleGrid().config.horizontal_config
    io_system = FieldGroupMonitor(
        config,
        vertical=vertical_size,
        horizontal=horizontal_size,
        grid_id="simple_grid",
        output_path=test_path,
    )
    assert io_system.next_output_time == datetime.fromisoformat(config.start_time)
    state = model_state(grid)
    io_system.store(state, step_time)
    assert io_system.next_output_time == datetime.fromisoformat("2022-01-01T01:00:00")

    step_time = datetime.fromisoformat("2022-01-01T01:10:00")
    io_system.store(state, step_time)
    assert io_system.next_output_time == datetime.fromisoformat("2022-01-01T01:00:00")


@pytest.mark.parametrize(
    ("config, message"),
    [
        (
            FieldGroupIoConfig(
                start_time="2023-04-04T11:00:00",
                filename="/vars/prognostics.nc",
                output_interval="1 HOUR",
                variables=["exner_function", "air_density"],
            ),
            "absolute path",
        ),
        (
            FieldGroupIoConfig(
                start_time="2023-04-04T11:00:00",
                filename="vars/prognostics.nc",
                output_interval="1 HOUR",
                variables=[],
            ),
            "No variables provided for output.",
        ),
        (
            FieldGroupIoConfig(
                start_time="2023-04-04T11:00:00",
                filename="vars/prognostics.nc",
                output_interval="",
                variables=["air_density, exner_function"],
            ),
            "No output interval provided.",
        ),
    ],
)
def test_fieldgroup_config_validate_filename(config, message):
    with pytest.raises(InvalidConfigError) as err:
        config.validate()
    assert message in str(err.value)


def test_fieldgroup_monitor_constructs_output_path_and_filepattern(test_path):
    config = FieldGroupIoConfig(
        start_time="2023-04-04T11:00:00",
        filename="vars/prognostics.nc",
        output_interval="1 HOUR",
        variables=["exner_function", "air_density"],
    )
    vertical_size = VerticalGridSize(10)
    horizontal_size = SimpleGrid().config.horizontal_config
    group_monitor = FieldGroupMonitor(
        config,
        vertical=vertical_size,
        horizontal=horizontal_size,
        grid_id="simple_grid",
        output_path=test_path,
    )
    assert group_monitor.output_path == test_path.joinpath("vars")
    assert group_monitor.output_path.exists()
    assert group_monitor.output_path.is_dir()
    assert "prognostics" in group_monitor._file_name_pattern
