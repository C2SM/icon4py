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

import pytest
from gt4py.next.ffront.fbuiltins import float32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.test_utils.datatest_utils import GLOBAL_EXPERIMENT
from icon4py.model.common.test_utils.grid_utils import get_icon_grid_from_gridfile
from icon4py.model.common.test_utils.helpers import random_field
from icon4py.model.driver.io.data import (
    PROGNOSTIC_CF_ATTRIBUTES,
    to_data_array,
)
from icon4py.model.driver.io.io import DatasetFactory, FieldGroupMonitor, FieldIoConfig, to_delta


grid = get_icon_grid_from_gridfile(GLOBAL_EXPERIMENT, on_gpu=False)
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


def test_io_monitor_output_time_updates_on_store():
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
    
    io_system.store(state,step_time)
    assert io_system.next_output_time == datetime.fromisoformat("2022-01-01T01:00:00")
    
    step_time = datetime.fromisoformat("2022-01-01T01:10:00")
    io_system.store(state, step_time)
    assert io_system.next_output_time == datetime.fromisoformat("2022-01-01T01:00:00")
    #TODO (magdalena) how to deal with non time matches? That is if the model time jumps over the output time
    
@pytest.mark.datatest
def test_io_monitor_fields_copied_on_store(grid_savepoint):
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



def test_initialize_dataset_factory():
    grid = SimpleGrid()
    heights = random_field(grid, CellDim, KDim, dtype=float32)
    
    vertical = VerticalModelParams(heights)
    factory = DatasetFactory("test_output.nc", vertical, attrs={"title":"test", "institution":"EXCLAIM - ETH Zurich"})
    dataset = factory.initialize_dataset()
    assert dataset.title == "test"
    assert dataset.institution == "EXCLAIM - ETH Zurich"
    # assert file exists
    # assert dimensions are defined
    # assert time is unlimited