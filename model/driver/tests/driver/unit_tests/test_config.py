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
import logging
import pathlib

import cattrs
import pytest

from icon4py.model.common.config import config_io
from icon4py.model.common.io import io as common_io
from icon4py.model.driver import config as driver_config, driver_states


_DATA_DIR = pathlib.Path(__file__).parent / "data"


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


#: The smallest experiment configuration: required sections only.
EXPERIMENT_CONFIG_YAML = (_DATA_DIR / "minimal_config.yml").read_text()


def test_read_experiment_config_from_yaml_resolves_relative_data_path(
    tmp_path: pathlib.Path,
) -> None:
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        EXPERIMENT_CONFIG_YAML.replace(
            "prescribed_tendencies: {}", "prescribed_tendencies:\n  data_path: ser_data"
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
            "prescribed_tendencies: {}", "prescribed_tendencies:\n  data_path: /abs/ser_data"
        )
    )
    config = driver_config.read_experiment_config_from_yaml(config_file)
    assert config.prescribed_tendencies.data_path == pathlib.Path("/abs/ser_data")


def test_read_experiment_config_from_yaml_drops_deprecated_driver_options(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The archived experiment configurations still set the removed driver output options."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        EXPERIMENT_CONFIG_YAML.replace(
            "  experiment_name: foo\n",
            "  experiment_name: foo\n  output_path: output\n  enable_output: false\n"
            "  output_backend: zarr\n  output_mode: distributed\n",
        )
    )
    with caplog.at_level(logging.WARNING, logger=driver_config.__name__):
        config = driver_config.read_experiment_config_from_yaml(config_file)
    assert config.io is None
    assert config.driver.experiment_name == "foo"
    assert "enable_output, output_backend, output_mode, output_path" in caplog.text
    # the strict reader rejects them like any unknown key
    with pytest.raises(cattrs.ClassValidationError):
        config_io.read_yaml_str(config_file.read_text(), driver_config.ExperimentConfig)


def test_io_roundtrip_cls_cls() -> None:
    conf = config_io.read_yaml_str(EXPERIMENT_CONFIG_YAML, driver_config.ExperimentConfig)
    assert conf.driver.experiment_name == "foo"
    assert (
        config_io.read_yaml_str(config_io.write_yaml_str(conf), driver_config.ExperimentConfig)
        == conf
    )


def test_io_roundtrip_str_str() -> None:
    config_str = (_DATA_DIR / "test_config.yml").read_text()
    roundtrip_str = config_io.write_yaml_str(
        config_io.read_yaml_str(config_str, driver_config.ExperimentConfig)
    )
    assert roundtrip_str == config_str


def _config_with_io_section() -> str:
    """The round-trip fixture with the 'io' section of 'io_section.yml' spliced in."""
    base = (_DATA_DIR / "test_config.yml").read_text()
    assert base.count("io:\n") == 1
    return base.replace("io:\n", (_DATA_DIR / "io_section.yml").read_text(), 1)


def test_io_section_roundtrip_str_str() -> None:
    """The YAML form of the 'io' section: the output directory and groups with tagged intervals."""
    config_str = _config_with_io_section()
    conf = config_io.read_yaml_str(config_str, driver_config.ExperimentConfig)
    assert conf.io is not None
    assert conf.io.output_path == pathlib.Path("output")
    assert [group.basename for group in conf.io.field_groups] == ["winds", "precipitation"]
    assert conf.io.field_groups[0].output_interval == 10
    assert conf.io.field_groups[0].variables == ["eastward_wind", "northward_wind"]
    assert conf.io.field_groups[1].backend is common_io.OutputBackend.NETCDF
    assert conf.io.field_groups[1].mode is common_io.OutputMode.GATHER
    assert config_io.write_yaml_str(conf) == config_str


def test_io_section_roundtrip_cls_cls() -> None:
    conf = config_io.read_yaml_str(_config_with_io_section(), driver_config.ExperimentConfig)
    assert (
        config_io.read_yaml_str(config_io.write_yaml_str(conf), driver_config.ExperimentConfig)
        == conf
    )
