from datetime import datetime, timedelta

import numpy as np
import pytest
from numpy import float32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.common.test_utils.helpers import random_field
from icon4py.model.driver.io.cf_utils import (
    DEFAULT_CALENDAR,
    DEFAULT_TIME_UNIT,
    INTERFACE_LEVEL_NAME,
    LEVEL_NAME,
    date2num,
)
from icon4py.model.driver.io.data import PROGNOSTIC_CF_ATTRIBUTES, to_data_array
from icon4py.model.driver.io.writers import (
    NetcdfWriter,
    filter_by_standard_name,
    to_canonical_dim_order,
)
from model.driver.tests.test_io import model_state, simple_grid, state_values


@pytest.mark.parametrize("input_", state_values())
def test_to_canonical_dim_order(input_):
    input_dims = input_.dims
    output = to_canonical_dim_order(input_)
    assert output.dims == (input_dims[1], input_dims[0])


@pytest.mark.parametrize("value", ["air_density", "upward_air_velocity"])
def test_filter_by_standard_name(value):
    state = model_state(SimpleGrid())
    assert filter_by_standard_name(state, value) == {value: state[value]}


def test_filter_by_standard_name_key_differs_from_name():
    state = model_state(SimpleGrid())
    assert filter_by_standard_name(state, "virtual_potential_temperature") == {
        "theta_v": state["theta_v"]
    }


def test_filter_by_standard_name_non_existing_name():
    state = model_state(SimpleGrid())
    assert filter_by_standard_name(state, "does_not_exist") == {}


def initialized_writer(test_path, random_name, grid=simple_grid):
    vertical = VerticalGridSize(grid.num_levels)
    horizontal = grid.config.horizontal_config
    fname = str(test_path.absolute()) + "/" + random_name + ".nc"
    dataset = NetcdfWriter(
        fname,
        vertical,
        horizontal,
        global_attrs={"title": "test", "institution": "EXCLAIM - ETH Zurich"},
    )
    dataset.initialize_dataset()
    return dataset, grid


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
    assert vertical.long_name == "model full levels"
    assert vertical.standard_name == LEVEL_NAME
    assert vertical.datatype == np.int32
    assert len(vertical) == grid.num_levels
    assert np.all(vertical == np.arange(grid.num_levels))


def test_initialize_writer_interface_levels(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    interface_levels = dataset.variables["interface_levels"]
    assert interface_levels.units == "1"
    assert interface_levels.datatype == np.int32
    assert interface_levels.long_name == "model interface levels"
    assert interface_levels.standard_name == INTERFACE_LEVEL_NAME
    assert len(interface_levels) == grid.num_levels + 1
    assert np.all(interface_levels == np.arange(grid.num_levels + 1))


def test_append_timeslice(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    time = datetime.now()
    assert len(dataset.variables["times"]) == 0
    slice1 = {}
    dataset.append(slice1, time)
    assert len(dataset.variables["times"]) == 1
    time1 = time + timedelta(hours=1)
    dataset.append(slice1, time1)
    assert len(dataset.variables["times"]) == 2
    time2 = time1 + timedelta(hours=1)
    dataset.append(slice1, time2)
    assert len(dataset.variables["times"]) == 3
    time_units = dataset.variables["times"].units
    cal = dataset.variables["times"].calendar
    assert np.all(
        dataset.variables["times"][:]
        == np.array(date2num((time, time1, time2), units=time_units, calendar=cal))
    )


def test_writer_append_timeslice_create_new_var(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    time = datetime.now()
    assert len(dataset.variables["times"]) == 0
    assert "air_density" not in dataset.variables

    state = dict(air_density=model_state(grid)["air_density"])
    dataset.append(state, time)
    assert len(dataset.variables["times"]) == 1
    assert "air_density" in dataset.variables
    assert dataset.variables["air_density"].dimensions == ("time", "level", "cell")
    assert dataset.variables["air_density"].shape == (1, grid.num_levels, grid.num_cells)
    assert np.allclose(dataset.variables["air_density"][0], state["air_density"].data.T)


def test_append_timeslice_existing_var(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    time = datetime.now()
    state = dict(air_density=model_state(grid)["air_density"])
    dataset.append(state, time)
    assert len(dataset.variables["times"]) == 1
    assert "air_density" in dataset.variables

    new_rho = random_field(grid, CellDim, KDim, dtype=float32)
    state["air_density"] = to_data_array(new_rho, PROGNOSTIC_CF_ATTRIBUTES["air_density"])
    new_time = time + timedelta(hours=1)
    dataset.append(state, new_time)

    assert len(dataset.variables["times"]) == 2
    assert dataset.variables["air_density"].shape == (2, grid.num_levels, grid.num_cells)
    assert np.allclose(dataset.variables["air_density"][1], new_rho.ndarray.T)


def test_initialize_writer_create_dimensions(
    test_path,
    random_name,
):
    dataset, grid = initialized_writer(test_path, random_name)

    assert dataset["title"] == "test"
    assert dataset["institution"] == "EXCLAIM - ETH Zurich"
    assert len(dataset.dims) == 6
    assert dataset.dims["level"].size == grid.num_levels
    assert dataset.dims["interface_level"].size == grid.num_levels + 1
    assert dataset.dims["cell"].size == grid.num_cells
    assert dataset.dims["vertex"].size == grid.num_vertices
    assert dataset.dims["edge"].size == grid.num_edges
    assert dataset.dims["time"].size == 0
    assert dataset.dims["time"].isunlimited

    assert dataset.variables["times"].units == DEFAULT_TIME_UNIT
    assert dataset.variables["times"].calendar == DEFAULT_CALENDAR
