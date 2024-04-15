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

import pytest
import xarray as xr
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
from icon4py.model.driver.io.data import (
    PROGNOSTIC_CF_ATTRIBUTES,
    to_data_array,
)
from icon4py.model.driver.io.io import (
    DatasetStore,
    FieldGroupMonitor,
    FieldIoConfig,
    IoConfig,
    IoMonitor,
    generate_name,
    to_delta,
)


UNLIMITED = None

grid = get_icon_grid_from_gridfile(GLOBAL_EXPERIMENT, on_gpu=False)
grid_file = GRIDS_PATH.joinpath(R02B04_GLOBAL, GLOBAL_GRIDFILE)
rho = random_field(grid, CellDim, KDim, dtype=float32)
exner = random_field(grid, CellDim, KDim, dtype=float32)
w = random_field(grid, CellDim, KDim,extend={KDim:1}, dtype=float32)

model_state = {"air_density": to_data_array(rho, PROGNOSTIC_CF_ATTRIBUTES["air_density"]),
         "exner_function": to_data_array(exner, PROGNOSTIC_CF_ATTRIBUTES["exner_function"]),
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

def test_io_monitor_create_output_path(path):
    path_name = path.absolute().as_posix() + "/output"
    config = IoConfig(base_name="test_", field_configs=[], output_end_time="2023-01-01T00:00:00", output_path= path_name)
    
    monitor = IoMonitor(config)
    assert monitor.path.exists()
    assert monitor.path.is_dir()

    
def test_io_monitor_write_ugrid_file(path):
    path_name = path.absolute().as_posix() + "/output"
    config = IoConfig(base_name="test_", field_configs=[], output_end_time="2023-01-01T00:00:00",
                      output_path=path_name)
    monitor = IoMonitor(config, grid_file=grid_file)
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
        variables=["upward_air_velocity", "air_density"],
    )
    vertical = VerticalModelParams(heights)

    io_system = FieldGroupMonitor(config, vertical=vertical)
    io_system.store(model_state, datetime.fromisoformat(config.start_time))
    #state["normal_velocity"] = 2.0
    assert False


def test_fieldgroup_monitor_output_time_updates_upon_store():
    config = FieldIoConfig(
        start_time="2022-01-01T00:00:00",
        filename_pattern="{base_name}_{time}.nc",
        output_interval="1 HOUR",
        variables=["normal_velocity", "air_density"],
    )
    state = {"normal_velocity": dict(), "air_density": dict()}
    step_time = datetime.fromisoformat(config.start_time)
    io_system = FieldGroupMonitor(config)
    assert io_system.next_output_time == datetime.fromisoformat(config.start_time)

    io_system.store(state, step_time)
    assert io_system.next_output_time == datetime.fromisoformat("2022-01-01T01:00:00")

    step_time = datetime.fromisoformat("2022-01-01T01:10:00")
    io_system.store(state, step_time)
    assert io_system.next_output_time == datetime.fromisoformat("2022-01-01T01:00:00")
    # TODO (magdalena) how to deal with non time matches? That is if the model time jumps over the output time


def test_initialize_datastore():
    grid = SimpleGrid()
    vertical = VerticalGridSize(grid.num_levels)
    horizontal = grid.config.horizontal_config
    heights = xr.DataArray(random_field(grid, CellDim, KDim, dtype=float32), attrs={"units":"m", "long_name":"sample height levels", "short_name":"height"})
    
    dataset = DatasetStore("test_output.nc", vertical, horizontal, global_attrs={"title":"test", "institution":"EXCLAIM - ETH Zurich"})
    dataset.initialize_dataset()
    
    assert dataset["title"] == "test"
    assert dataset["institution"] == "EXCLAIM - ETH Zurich"
    assert len(dataset.dims) == 6
    assert dataset.dims["full_level"].size == grid.num_levels
    assert dataset.dims["interface_level"].size == grid.num_levels + 1
    assert dataset.dims["cell"].size == grid.num_cells
    assert dataset.dims["vertex"].size == grid.num_vertices
    assert dataset.dims["edge"].size == grid.num_edges
    assert dataset.dims["time"].size == 0
    # TODO assert dims["time"] is unlimited
    
    #assert dataset.variables["times"].ncattrs["units"] == "seconds since 1970-01-01 00:00:00"
    #assert dataset.variables["times"].ncattrs["calendar"] is not None

    

    

   