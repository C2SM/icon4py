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
import pathlib
import textwrap

import pytest

from icon4py.model.common.config import config_io
from icon4py.model.common.io import io as common_io, netcdf_writers
from icon4py.model.driver import config as driver_config, driver_states


def _driver_config(
    start_of_timestepping: datetime.datetime | None = None,
) -> driver_config.DriverConfig:
    # the experiment runs from 2000-01-01T00:00:00 to 01:00:00, with a 120 s time step
    start = datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)
    end = datetime.datetime(2000, 1, 1, 1, 0, 0, tzinfo=datetime.UTC)
    config = driver_config.DriverConfig.make_initial(
        experiment_name="test",
        profiling_options=None,
        dtime=driver_config.relativetime_from_iso8601("PT120S"),
        start_of_simulation=start,
        end_of_simulation=end,
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


def test_driver_config_accepts_distributed_netcdf_on_any_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The driver config never rejects distributed netCDF: the check is rank-aware.

    Single-rank runs write through a serial file handle whatever the installation, so
    the parallel-support check happens when the writer is created in a multi-rank run
    (see ``netcdf_writers.NETCDFWriter``), not at config construction.
    """
    monkeypatch.setattr(netcdf_writers, "missing_parallel_support", lambda: "<serial build>")
    config = dataclasses.replace(
        _driver_config(),
        output_backend=common_io.OutputBackend.NETCDF,
        output_mode=common_io.OutputMode.DISTRIBUTED,
    )
    assert config.output_backend is common_io.OutputBackend.NETCDF
    assert config.output_mode is common_io.OutputMode.DISTRIBUTED


EXPERIMENT_CONFIG_YAML = textwrap.dedent(
    """
    geometry: {}
    metrics: {}
    interpolation: {}
    vertical_grid:
        num_levels: 10
    topography:
        type: jablonowski_williamson
    initial_condition:
        type: jablonowski_williamson
    prescribed_tendencies: {}
    driver:
        experiment_name: foo
        profiling_options:
        dtime: 10 seconds
        start_of_simulation: 2020-01-01T00:00:00
        start_of_timestepping: 2020-01-01T00:00:00
        end_of_simulation:
            type: numsteps
            value: 5
    """
)


def test_read_experiment_config_from_yaml_resolves_relative_data_path(
    tmp_path: pathlib.Path,
) -> None:
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        EXPERIMENT_CONFIG_YAML.replace(
            "prescribed_tendencies: {}", "prescribed_tendencies:\n    data_path: ser_data"
        )
    )
    config = driver_config.read_experiment_config_from_yaml(config_file)
    assert config.prescribed_tendencies.data_path == tmp_path.resolve() / "ser_data"


def test_read_experiment_config_from_yaml_keeps_absolute_data_path(
    tmp_path: pathlib.Path,
) -> None:
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        EXPERIMENT_CONFIG_YAML.replace(
            "prescribed_tendencies: {}", "prescribed_tendencies:\n    data_path: /abs/ser_data"
        )
    )
    config = driver_config.read_experiment_config_from_yaml(config_file)
    assert config.prescribed_tendencies.data_path == pathlib.Path("/abs/ser_data")


def test_io_roundtrip_cls_cls() -> None:
    conf = config_io.read_yaml_str(EXPERIMENT_CONFIG_YAML, driver_config.ExperimentConfig)
    assert conf.driver.experiment_name == "foo"
    assert (
        config_io.read_yaml_str(config_io.write_yaml_str(conf), driver_config.ExperimentConfig)
        == conf
    )


def test_io_roundtrip_str_str() -> None:
    config_str = (pathlib.Path(__file__).parent / "data" / "test_config.yml").read_text()
    roundtrip_str = config_io.write_yaml_str(
        config_io.read_yaml_str(config_str, driver_config.ExperimentConfig)
    )
    assert roundtrip_str == config_str
