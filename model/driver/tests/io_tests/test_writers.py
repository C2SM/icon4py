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

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid import base, vertical
from icon4py.model.common.test_utils import helpers
from icon4py.model.driver.io import cf_utils, data
from icon4py.model.driver.io.writers import (
    NETCDFWriter,
    TimeProperties,
    filter_by_standard_name,
)

from . import test_io


@pytest.mark.parametrize("value", ["air_density", "upward_air_velocity"])
def test_filter_by_standard_name(value):
    state = test_io.model_state(test_io.simple_grid)
    assert filter_by_standard_name(state, value) == {value: state[value]}


def test_filter_by_standard_name_key_differs_from_name():
    state = test_io.model_state(test_io.simple_grid)
    assert filter_by_standard_name(state, "virtual_potential_temperature") == {
        "theta_v": state["theta_v"]
    }


def test_filter_by_standard_name_non_existing_name():
    state = test_io.model_state(test_io.simple_grid)
    assert filter_by_standard_name(state, "does_not_exist") == {}


def initialized_writer(
    test_path, random_name, grid=test_io.simple_grid
) -> tuple[NETCDFWriter, base.BaseGrid]:
    vertical_config = grid.config.vertical_config
    num_levels = vertical_config.num_lev
    heights = np.linspace(start=12000.0, stop=0.0, num=num_levels + 1)
    vertical_params = vertical.VerticalModelParams(gtx.as_field((KDim,), heights))
    horizontal = grid.config.horizontal_config
    fname = str(test_path.absolute()) + "/" + random_name + ".nc"
    writer = NETCDFWriter(
        fname,
        vertical_params,
        horizontal,
        TimeProperties(cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR),
        global_attrs={"title": "test", "institution": "EXCLAIM - ETH Zurich"},
    )
    writer.initialize_dataset()
    return writer, grid


def test_initialize_writer_time_var(test_path, random_name):
    dataset, _ = initialized_writer(test_path, random_name)
    time_var = dataset.variables["times"]
    assert time_var.dimensions == ("time",)
    assert time_var.units == "seconds since 1970-01-01 00:00:00"
    assert time_var.calendar == "proleptic_gregorian"
    assert time_var.long_name == "time"
    assert time_var.standard_name == "time"
    assert len(time_var) == 0


def test_initialize_writer_vertical_model_levels(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    vertical = dataset.variables["levels"]
    assert vertical.units == "1"
    assert vertical.dimensions == ("level",)
    assert vertical.long_name == "model full level index"
    assert vertical.standard_name == cf_utils.LEVEL_STANDARD_NAME
    assert vertical.datatype == np.int32
    assert len(vertical) == grid.num_levels
    assert np.all(vertical == np.arange(grid.num_levels))


def test_initialize_writer_interface_levels(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    interface_levels = dataset.variables["interface_levels"]
    assert interface_levels.units == "1"
    assert interface_levels.datatype == np.int32
    assert interface_levels.long_name == "model interface level index"
    assert interface_levels.standard_name == cf_utils.INTERFACE_LEVEL_STANDARD_NAME
    assert len(interface_levels) == grid.num_levels + 1
    assert np.all(interface_levels == np.arange(grid.num_levels + 1))


def test_initialize_writer_heights(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    heights = dataset.variables["height"]
    assert heights.units == "m"
    assert heights.datatype == np.float64
    assert heights.long_name == "height value of half levels for flat topography"
    assert heights.standard_name == cf_utils.INTERFACE_LEVEL_HEIGHT_STANDARD_NAME
    assert len(heights) == grid.num_levels + 1
    assert heights[0] == 12000.0
    assert heights[-1] == 0.0


def test_writer_append_timeslice(test_path, random_name):
    writer, grid = initialized_writer(test_path, random_name)
    time = datetime.now()
    assert len(writer.variables["times"]) == 0
    slice1 = {}
    writer.append(slice1, time)
    assert len(writer.variables["times"]) == 1
    time1 = time + timedelta(hours=1)
    writer.append(slice1, time1)
    assert len(writer.variables["times"]) == 2
    time2 = time1 + timedelta(hours=1)
    writer.append(slice1, time2)
    assert len(writer.variables["times"]) == 3
    time_units = writer.variables["times"].units
    cal = writer.variables["times"].calendar
    assert np.all(
        writer.variables["times"][:]
        == np.array(cf_utils.date2num((time, time1, time2), units=time_units, calendar=cal))
    )


def test_writer_append_timeslice_create_new_var(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    time = datetime.now()
    assert len(dataset.variables["times"]) == 0
    assert "air_density" not in dataset.variables

    state = dict(air_density=test_io.model_state(grid)["air_density"])
    dataset.append(state, time)
    assert len(dataset.variables["times"]) == 1
    assert "air_density" in dataset.variables
    assert dataset.variables["air_density"].dimensions == ("time", "level", "cell")
    assert dataset.variables["air_density"].shape == (
        1,
        grid.num_levels,
        grid.num_cells,
    )
    assert np.allclose(dataset.variables["air_density"][0], state["air_density"].data.T)


def test_writer_append_timeslice_to_existing_var(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    time = datetime.now()
    state = dict(air_density=test_io.model_state(grid)["air_density"])
    dataset.append(state, time)
    assert len(dataset.variables["times"]) == 1
    assert "air_density" in dataset.variables

    new_rho = helpers.random_field(grid, CellDim, KDim, dtype=np.float32)
    state["air_density"] = data.to_data_array(new_rho, data.PROGNOSTIC_CF_ATTRIBUTES["air_density"])
    new_time = time + timedelta(hours=1)
    dataset.append(state, new_time)

    assert len(dataset.variables["times"]) == 2
    assert dataset.variables["air_density"].shape == (
        2,
        grid.num_levels,
        grid.num_cells,
    )
    assert np.allclose(dataset.variables["air_density"][1], new_rho.ndarray.T)


def test_initialize_writer_create_dimensions(
    test_path,
    random_name,
):
    writer, grid = initialized_writer(test_path, random_name)

    assert writer["title"] == "test"
    assert writer["institution"] == "EXCLAIM - ETH Zurich"
    assert len(writer.dims) == 6
    assert writer.dims["level"].size == grid.num_levels
    assert writer.dims["interface_level"].size == grid.num_levels + 1
    assert writer.dims["cell"].size == grid.num_cells
    assert writer.dims["vertex"].size == grid.num_vertices
    assert writer.dims["edge"].size == grid.num_edges
    assert writer.dims["time"].size == 0
    assert writer.dims["time"].isunlimited

    assert writer.variables["times"].units == cf_utils.DEFAULT_TIME_UNIT
    assert writer.variables["times"].calendar == cf_utils.DEFAULT_CALENDAR
