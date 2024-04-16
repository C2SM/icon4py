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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import pytest
import xarray as xr
from cftime import date2num
from gt4py.next.ffront.fbuiltins import float32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.grid.vertical import VerticalGridSize, VerticalModelParams
from icon4py.model.common.test_utils.datatest_utils import (
    GLOBAL_EXPERIMENT,
    GRIDS_PATH,
    R02B04_GLOBAL,
)
from icon4py.model.common.test_utils.grid_utils import GLOBAL_GRIDFILE, get_icon_grid_from_gridfile
from icon4py.model.common.test_utils.helpers import random_field
from icon4py.model.driver.io.cf_utils import INTERFACE_LEVEL_NAME, LEVEL_NAME
from icon4py.model.driver.io.data import (
    PROGNOSTIC_CF_ATTRIBUTES,
    to_data_array,
)
from icon4py.model.driver.io.io import (
    FieldGroupMonitor,
    FieldIoConfig,
    IoConfig,
    IoMonitor,
    NetcdfWriter,
    filter_by_standard_name,
    generate_name,
    to_delta,
)


UNLIMITED = None

grid_file = GRIDS_PATH.joinpath(R02B04_GLOBAL, GLOBAL_GRIDFILE)
global_grid = get_icon_grid_from_gridfile(GLOBAL_EXPERIMENT, on_gpu=False)
simple_grid = SimpleGrid()
grid = simple_grid
rho = random_field(grid, CellDim, KDim, dtype=float32)
exner = random_field(grid, CellDim, KDim, dtype=float32)
theta_v = random_field(grid, CellDim, KDim, dtype=float32)
w = random_field(grid, CellDim, KDim,extend={KDim:1}, dtype=float32)

model_state = {"air_density": to_data_array(rho, PROGNOSTIC_CF_ATTRIBUTES["air_density"]),
         "exner_function": to_data_array(exner, PROGNOSTIC_CF_ATTRIBUTES["exner_function"]),
         "theta_v": to_data_array(theta_v, PROGNOSTIC_CF_ATTRIBUTES["virtual_potential_temperature"]),      
         "upward_air_velocity": to_data_array(w, PROGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"])}


@pytest.mark.parametrize("num", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("slot", ["HOUR", "hour", "Hour"])
def test_to_delta_hrs(num, slot):
    assert to_delta("HOUR") == timedelta(hours=1)
    assert to_delta(f"{num} {slot}") == timedelta(hours=num)


@pytest.mark.parametrize("num", [0, 2, 44, 4, 5])
@pytest.mark.parametrize("slot", ["second", "SECOND"])
def test_to_delta_secs(num, slot):
    assert to_delta(f"{num} {slot}") == timedelta(seconds=num)
    
@pytest.mark.parametrize("num", [0, 2, 3, 4, 5])
@pytest.mark.parametrize("slot", ["MINUTE", "Minute"])
def test_to_delta_mins(num, slot):
    assert to_delta(f"{num} {slot}") == timedelta(minutes=num)


@pytest.mark.parametrize("name, expected", [("output.nc", "output_2.nc"),
                                            ("outxxput_20220101.xc", "outxxput_20220101_2.nc"),
                                            ("output_20220101T000000_x",
                                             "output_20220101T000000_x_2.nc")])
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
    config = IoConfig(base_name="test_", field_configs=[], output_path= path_name)
    monitor = IoMonitor(config, VerticalGridSize(10), SimpleGrid().config.horizontal_config,
                        grid_file, "simple_grid")
    assert monitor.path.exists()
    assert monitor.path.is_dir()

    
def test_io_monitor_write_ugrid_file(test_path):
    path_name = test_path.absolute().as_posix() + "/output"
    config = IoConfig(base_name="test_", field_configs=[],output_path=path_name)
    monitor = IoMonitor(config, VerticalGridSize(10), SimpleGrid().config.horizontal_config,
                        grid_file, "simple_grid")
    ugrid_file = monitor.path.iterdir().__next__().absolute()
    assert "ugrid.nc" in ugrid_file.name
    assert is_valid_uxgrid(ugrid_file)

@pytest.mark.fail   
@pytest.mark.datatest
def test_fieldgroup_monitor_fields_copied_on_store(grid_savepoint):
    heights = grid_savepoint.vct_a()
    config = FieldIoConfig(
        filename_pattern="_output_20220101.nc",
        start_time="2022-01-01T00:00:00",
        output_interval="1 HOUR",
        variables=["exner_function", "air_density"],
    )
    vertical = VerticalModelParams(heights)
    horizontal = grid_savepoint.config.horizontal_config
    io_system = FieldGroupMonitor(config, horizontal=horizontal, vertical=vertical)
    io_system.store(model_state, datetime.fromisoformat(config.start_time))
    #state["normal_velocity"] = 2.0
    assert False


def test_fieldgroup_monitor_output_time_updates_upon_store():
    config = FieldIoConfig(
        start_time="2022-01-01T00:00:00",
        filename_pattern="{base_name}_{time}.nc",
        output_interval="1 HOUR",
        variables=["exner_function", "air_density"],
    )
    step_time = datetime.fromisoformat(config.start_time)
    vertical_size = VerticalGridSize(10)
    horizontal_size = SimpleGrid().config.horizontal_config
    io_system = FieldGroupMonitor(config,vertical=vertical_size, horizontal=horizontal_size, grid_id="simple_grid")
    assert io_system.next_output_time == datetime.fromisoformat(config.start_time)
    
    io_system.store(model_state, step_time)
    assert io_system.next_output_time == datetime.fromisoformat("2022-01-01T01:00:00")

    step_time = datetime.fromisoformat("2022-01-01T01:10:00")
    io_system.store(model_state, step_time)
    assert io_system.next_output_time == datetime.fromisoformat("2022-01-01T01:00:00")
    # TODO (magdalena) how to deal with non time matches? That is if the model time jumps over the output time


def test_initialize_datastore_creates_dimensions(test_path, random_name ):
    dataset, grid = initialize_dataset(test_path, random_name)

    assert dataset["title"] == "test"
    assert dataset["institution"] == "EXCLAIM - ETH Zurich"
    assert len(dataset.dims) == 6
    assert dataset.dims["level"].size == grid.num_levels
    assert dataset.dims["interface_level"].size == grid.num_levels + 1
    assert dataset.dims["cell"].size == grid.num_cells
    assert dataset.dims["vertex"].size == grid.num_vertices
    assert dataset.dims["edge"].size == grid.num_edges
    assert dataset.dims["time"].size == 0
    # TODO assert dims["time"] is unlimited
    
    heights = xr.DataArray(random_field(grid, CellDim, KDim, dtype=float32), attrs={"units":"m", "long_name":"sample height levels", "short_name":"height"})
    #assert dataset.variables["times"].ncattrs["units"] == "seconds since 1970-01-01 00:00:00"
    #assert dataset.variables["times"].ncattrs["calendar"] is not None


def initialize_dataset(test_path, random_name):
    grid = SimpleGrid()
    vertical = VerticalGridSize(grid.num_levels)
    horizontal = grid.config.horizontal_config
    fname = str(test_path.absolute()) +"/"+ random_name + ".nc"
    dataset = NetcdfWriter(fname, vertical, horizontal,
                           global_attrs={"title": "test", "institution": "EXCLAIM - ETH Zurich"})
    dataset.initialize_dataset()
    return dataset, grid

    
def test_initialize_datastore_time_var(test_path, random_name):
    dataset, grid = initialize_dataset(test_path, random_name)
    time_var = dataset.variables["times"]
    assert time_var.dimensions == ("time",)
    assert time_var.units == "seconds since 1970-01-01 00:00:00"
    assert time_var.calendar == "proleptic_gregorian"
    assert time_var.long_name == "time"
    assert time_var.standard_name == "time"
    assert len(time_var) == 0
    
def test_initialize_datastore_vertical_model_levels(test_path, random_name):
    dataset, grid = initialize_dataset(test_path, random_name)
    vertical = dataset.variables["levels"]
    assert vertical.units == "1"
    assert vertical.dimensions == ("level",)
    assert vertical.long_name == "model full levels"
    assert vertical.standard_name == LEVEL_NAME
    assert vertical.datatype == np.int32
    assert len(vertical) == grid.num_levels
    assert np.all(vertical == np.arange(grid.num_levels))

def test_initialize_datastore_interface_levels(test_path, random_name):
    dataset, grid = initialize_dataset(test_path, random_name)
    interface_levels = dataset.variables["interface_levels"]
    assert interface_levels.units == "1"
    assert interface_levels.datatype == np.int32
    assert interface_levels.long_name == "model interface levels"
    assert interface_levels.standard_name == INTERFACE_LEVEL_NAME
    assert len(interface_levels) == grid.num_levels + 1
    assert np.all(interface_levels == np.arange(grid.num_levels + 1))




@pytest.mark.parametrize("value", ["air_density", "upward_air_velocity"])
def test_filter_by_metadata(value):
    assert filter_by_standard_name(model_state, value) == {value: model_state[value]}
    
def test_filter_by_metadata_custom_name():
    assert filter_by_standard_name(model_state, "virtual_potential_temperature") == {"theta_v": model_state["theta_v"]}
def test_filter_by_metadata_empty():
    assert filter_by_standard_name(model_state, "does_not_exist") == {}

def test_append_timeslice(test_path, random_name):
    dataset, grid = initialize_dataset(test_path, random_name)
    time = datetime.now()
    assert len(dataset.variables["times"]) == 0
    slice1   = {}
    dataset.append(slice1, time)
    assert len(dataset.variables["times"]) == 1
    # TODO assert variable with 3 dims (time, horizontal, vertical) 
    #  check default order in CF conventions
    time1 = time + timedelta(hours=1)
    dataset.append(slice1, time1)
    assert len(dataset.variables["times"]) == 2
    time2 = time1 + timedelta(hours=1)
    dataset.append(slice1, time2)
    assert len(dataset.variables["times"]) == 3
    time_units = dataset.variables["times"].units
    cal = dataset.variables["times"].calendar
    assert np.all(dataset.variables["times"][:] == np.array(date2num((time, time1, time2),
                                                                  units=time_units, calendar=cal)))


def test_append_timeslice_create_new_var(test_path, random_name):
    dataset, grid = initialize_dataset(test_path, random_name)
    time = datetime.now()
    assert len(dataset.variables["times"]) == 0
    assert "air_density" not in dataset.variables
    model_state = {"air_density": to_data_array(rho, PROGNOSTIC_CF_ATTRIBUTES["air_density"])}
    dataset.append(model_state, time)
    assert len(dataset.variables["times"]) == 1
    assert "air_density" in dataset.variables
    assert dataset.variables["air_density"].dimensions == ("time", "cell", "level")
    assert dataset.variables["air_density"].shape == (1, grid.num_cells, grid.num_levels)
    assert np.allclose(dataset.variables["air_density"][0], rho.ndarray)

def test_append_timeslice_existing_var(test_path, random_name):
    dataset, grid = initialize_dataset(test_path, random_name)
    time = datetime.now()
    model_state = {"air_density": to_data_array(rho, PROGNOSTIC_CF_ATTRIBUTES["air_density"])}
    dataset.append(model_state, time)
    assert len(dataset.variables["times"]) == 1
    assert "air_density" in dataset.variables
    new_rho = random_field(grid, CellDim, KDim, dtype=float32)
    model_state["air_density"] = to_data_array(new_rho, PROGNOSTIC_CF_ATTRIBUTES["air_density"])
    new_time = time + timedelta(hours=1)
    dataset.append(model_state, new_time)
    
    assert len(dataset.variables["times"]) == 2
    assert dataset.variables["air_density"].shape == (2, grid.num_cells, grid.num_levels)
    assert np.allclose(dataset.variables["air_density"][1], new_rho.ndarray)