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

from icon4py.model.driver.io.io import FieldGroupMonitor, FieldIoConfig, IoConfig, to_delta


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
    
  
def test_io_monitor_fields_copied_on_store():
    config = IoConfig(
        base_name="icon4py_atm_",
        filename_pattern="{base_name}_{time}.nc",
        start_time="2022-01-01T00:00:00",
        end_time="2022-01-01T05:00:00",
        output_interval="1 HOUR",
        variables=["normal_velocity", "air_density"],
    )
    state = {"normal_velocity": dict(), "air_density": dict()}
    io_system = FieldGroupMonitor(config)
    io_system.store(state, datetime.fromisoformat(config.start_time))
    #state["normal_velocity"] = 2.0
    assert False


