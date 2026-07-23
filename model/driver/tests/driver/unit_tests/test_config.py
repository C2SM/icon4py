# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``driver.config`` (data-free)."""

import dataclasses
import datetime

import pytest

from icon4py.model.driver import config as driver_config, driver_states


def _make_dicts(run_nml: dict) -> tuple[dict, dict]:
    # fortran dumps the whole namelist, so the variables the driver reads are always
    # present. Here they only need a value when the test does not care about it.
    atm_dict = {
        "nonhydrostatic_nml": {"vcfl_threshold": 0.85, "ndyn_substeps": 5},
        "run_nml": {"ltestcase": True, "ltransport": False} | run_nml,
    }
    master_dict = {
        "master_time_control_nml": {
            "experimentstartdate": "2000-01-01T00:00:00Z",
            "experimentstopdate": "2000-01-01T01:00:00Z",
        },
        "master_model_nml": {"model_namelist_filename": "NAMELIST_test_sb_atm"},
    }
    return atm_dict, master_dict


@pytest.mark.parametrize(
    ("duration", "expected_seconds"),
    [
        ("PT300S", 300.0),
        ("PT1H", 3600.0),
        ("PT10M", 600.0),
        ("PT1H30M", 5400.0),
        ("P1DT6H", 108000.0),
        ("PT0.5S", 0.5),
    ],
)
def test_relativetime_from_iso8601_valid(duration: str, expected_seconds: float) -> None:
    assert driver_config.relativetime_from_iso8601(duration) == datetime.timedelta(
        seconds=expected_seconds
    )


@pytest.mark.parametrize("duration", ["", "P", "PT", "P1Y", "P1M", "300", "PT300", "P1DT", "P1WT"])
def test_relativetime_from_iso8601_invalid(duration: str) -> None:
    with pytest.raises(ValueError, match="Invalid ISO 8601 duration"):
        driver_config.relativetime_from_iso8601(duration)


def test_modeltimestep_takes_priority_over_dtime() -> None:
    # trailing whitespace mimics the fixed-width Fortran string
    atm_dict, master_dict = _make_dicts(
        {"dtime": 999.0, "modeltimestep": "PT300S                          "}
    )
    config = driver_config.DriverConfig.from_fortran_dict(
        atm_dict=atm_dict, master_dict=master_dict, profiling_options=None
    )
    assert config.dtime == datetime.timedelta(seconds=300)


def test_empty_modeltimestep_falls_back_to_dtime() -> None:
    atm_dict, master_dict = _make_dicts({"dtime": 120.0, "modeltimestep": "        "})
    config = driver_config.DriverConfig.from_fortran_dict(
        atm_dict=atm_dict, master_dict=master_dict, profiling_options=None
    )
    assert config.dtime == datetime.timedelta(seconds=120)


# ltransport is true for MCH_CH_R04B09, EXCLAIM_APE_AES and Weisman-Klemp, false for
# the dry testcases (JW, GAUSS3D).
@pytest.mark.parametrize("ltransport", [True, False])
def test_do_prep_adv_from_ltransport(ltransport: bool) -> None:
    atm_dict, master_dict = _make_dicts(
        {"dtime": 10.0, "modeltimestep": "  ", "ltransport": ltransport}
    )
    config = driver_config.DriverConfig.from_fortran_dict(
        atm_dict=atm_dict, master_dict=master_dict, profiling_options=None
    )
    assert config.do_prep_adv is ltransport


# The extra diffusion call before the time loop is only made for real data runs, which
# are the ones that are not a testcase. MCH_CH_R04B09 is the only one.
@pytest.mark.parametrize("ltestcase", [True, False])
def test_diffuse_before_time_loop(ltestcase: bool) -> None:
    atm_dict, master_dict = _make_dicts(
        {"dtime": 10.0, "modeltimestep": "  ", "ltestcase": ltestcase}
    )
    config = driver_config.DriverConfig.from_fortran_dict(
        atm_dict=atm_dict, master_dict=master_dict, profiling_options=None
    )
    assert config.diffuse_before_time_loop is (not ltestcase)
    assert config.apply_extra_second_order_divdamp is (not ltestcase)


def _driver_config(
    start_of_timestepping: datetime.datetime | None = None,
) -> driver_config.DriverConfig:
    # the experiment runs from 2000-01-01T00:00:00 to 01:00:00, with a 120 s time step
    atm_dict, master_dict = _make_dicts({"dtime": 120.0, "modeltimestep": "  "})
    config = driver_config.DriverConfig.from_fortran_dict(
        atm_dict=atm_dict, master_dict=master_dict, profiling_options=None
    )
    if start_of_timestepping is None:
        return config
    return dataclasses.replace(config, start_of_timestepping=start_of_timestepping)


def test_time_loop_starts_at_the_beginning_of_the_simulation() -> None:
    model_time = driver_states.ModelTimeVariables(config=_driver_config())

    assert model_time.simulation_current_datetime == model_time.simulation_start_datetime
    assert model_time.is_first_step_in_simulation is True
    assert model_time.elapsed_time_in_seconds == 0.0
    assert model_time.n_time_steps == 30


def test_the_time_loop_cannot_start_before_the_simulation() -> None:
    with pytest.raises(ValueError, match="before the beginning of the simulation"):
        _driver_config(datetime.datetime(1999, 12, 31, tzinfo=datetime.UTC))


def test_restart_starts_the_time_loop_at_start_of_timestepping() -> None:
    start_of_timestepping = datetime.datetime(
        2000, 1, 1, 0, 30, tzinfo=datetime.UTC
    )  # half an hour into the simulation
    model_time = driver_states.ModelTimeVariables(config=_driver_config(start_of_timestepping))

    assert model_time.simulation_current_datetime == start_of_timestepping
    # linit_dyn is false on a restart
    assert model_time.is_first_step_in_simulation is False
    # ICON measures the elapsed time from the beginning of the simulation
    assert model_time.elapsed_time_in_seconds == 1800.0
    assert model_time.n_time_steps == 15
