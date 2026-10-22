# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from typer.testing import CliRunner

import icon4py.model.common.exceptions as errors
from icon4py.model.common import model_backends
from icon4py.model.common.io import io as common_io
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import config as driver_config, driver, driver_io, driver_utils, main

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


_DATA_DIR = pathlib.Path(__file__).parent / "data"
_TEST_CONFIG_FILE = _DATA_DIR / "test_config.yml"


def _config_file_with_io_section(directory: pathlib.Path) -> pathlib.Path:
    """The test configuration with the 'io' section of 'io_section.yml' spliced in."""
    config_file = directory / "test_config_with_io.yml"
    config_file.write_text(
        _TEST_CONFIG_FILE.read_text().replace(
            "io:\n", (_DATA_DIR / "io_section.yml").read_text(), 1
        )
    )
    return config_file


@pytest.fixture
def backend_name(pytestconfig: pytest.Config) -> str:
    spec = pytestconfig.getoption("backend", model_backends.DEFAULT_BACKEND)
    if spec not in model_backends.BACKENDS:
        pytest.skip(f"'{spec}' is not a backend name the driver CLI accepts")
    return spec


def _run_cli(
    config_file: pathlib.Path, backend_name: str, *options: str
) -> tuple[mock.MagicMock, mock.MagicMock]:
    """Invoke the driver CLI with the grid manager and the driver run mocked away."""
    mock_grid_manager = mock.MagicMock()
    with (
        mock.patch.object(driver_utils, "create_grid_manager", return_value=mock_grid_manager),
        mock.patch.object(driver, "run_driver") as mock_run_driver,
    ):
        result = CliRunner().invoke(
            main.app,
            [
                "--grid-file-path",
                str(config_file.parent / "dummy_grid.nc"),
                "--config-file-path",
                str(config_file),
                "--icon4py-backend",
                backend_name,
                *options,
            ],
        )
    assert result.exit_code == 0, result.output or result.exception
    return mock_grid_manager, mock_run_driver


def _run_config(mock_run_driver: mock.MagicMock) -> driver_config.ExperimentConfig:
    mock_run_driver.assert_called_once()
    config = mock_run_driver.call_args.kwargs["config"]
    assert isinstance(config, driver_config.ExperimentConfig)
    return config


def test_main_cli(backend: gtx_typing.Backend, backend_name: str) -> None:
    mock_grid_manager, mock_run_driver = _run_cli(_TEST_CONFIG_FILE, backend_name)

    run_driver_kwargs = mock_run_driver.call_args.kwargs
    assert run_driver_kwargs["grid_manager"] is mock_grid_manager
    assert data_alloc.backend_name(run_driver_kwargs["backend"]) == data_alloc.backend_name(backend)
    config = _run_config(mock_run_driver)
    assert config.driver.experiment_name == "foo"
    # without output options on the command line the configuration file decides: no
    # 'io' section, no output
    assert config.io is None


def test_main_cli_enable_output_writes_the_default_group(
    backend_name: str, tmp_path: pathlib.Path
) -> None:
    _, mock_run_driver = _run_cli(
        _TEST_CONFIG_FILE,
        backend_name,
        "--enable-output",
        "--output-path",
        str(tmp_path / "out"),
        "--output-backend",
        "netcdf",
        "--output-mode",
        "gather",
    )

    config = _run_config(mock_run_driver)
    assert config.io is not None
    assert config.io.output_path == tmp_path / "out"
    (group,) = config.io.field_groups
    assert group.basename == driver_io.DEFAULT_OUTPUT_BASENAME
    assert group.variables == driver_io.DEFAULT_OUTPUT_VARIABLES
    assert group.backend is common_io.OutputBackend.NETCDF
    assert group.mode is common_io.OutputMode.GATHER


def test_main_cli_output_options_apply_to_every_io_group(
    backend_name: str, tmp_path: pathlib.Path
) -> None:
    _, mock_run_driver = _run_cli(
        _config_file_with_io_section(tmp_path),
        backend_name,
        "--output-backend",
        "netcdf",
        "--output-mode",
        "gather",
    )

    config = _run_config(mock_run_driver)
    assert config.io is not None
    assert [group.basename for group in config.io.field_groups] == ["winds", "precipitation"]
    assert all(
        group.backend is common_io.OutputBackend.NETCDF
        and group.mode is common_io.OutputMode.GATHER
        for group in config.io.field_groups
    )
    # the other settings survive
    assert config.io.output_path == pathlib.Path("output")
    assert [group.output_interval for group in config.io.field_groups] == [10, 1]


def test_main_cli_no_enable_output_drops_the_io_section(
    backend_name: str, tmp_path: pathlib.Path
) -> None:
    _, mock_run_driver = _run_cli(
        _config_file_with_io_section(tmp_path), backend_name, "--no-enable-output"
    )
    assert _run_config(mock_run_driver).io is None


def test_apply_output_overrides_none_keep_the_configuration(tmp_path: pathlib.Path) -> None:
    config = driver_config.read_experiment_config_from_yaml(_config_file_with_io_section(tmp_path))
    overridden = main.apply_output_overrides(
        config, enable_output=None, output_path=None, output_backend=None, output_mode=None
    )
    assert overridden is config


def test_apply_output_overrides_enable_output_keeps_the_io_section(
    tmp_path: pathlib.Path,
) -> None:
    config = driver_config.read_experiment_config_from_yaml(_config_file_with_io_section(tmp_path))
    overridden = main.apply_output_overrides(
        config, enable_output=True, output_path=None, output_backend=None, output_mode=None
    )
    assert overridden.io is config.io


@pytest.mark.parametrize("enable_output", [None, False])
def test_apply_output_overrides_reject_output_options_without_output(
    enable_output: bool | None,
) -> None:
    config = driver_config.read_experiment_config_from_yaml(_TEST_CONFIG_FILE)
    assert config.io is None
    with pytest.raises(errors.InvalidConfigError, match="--output-path, --output-mode require"):
        main.apply_output_overrides(
            config,
            enable_output=enable_output,
            output_path=pathlib.Path("out"),
            output_backend=None,
            output_mode=common_io.OutputMode.GATHER,
        )


def test_apply_output_overrides_name_the_command_line_as_the_cause() -> None:
    # sharding is zarr-only: switching such a group to netCDF from the command line fails
    config = driver_config.read_experiment_config_from_yaml(_TEST_CONFIG_FILE).with_overrides(
        io=common_io.IOConfig(
            field_groups=[
                common_io.FieldGroupIOConfig(
                    basename="sharded",
                    variables=["air_density"],
                    horizontal_chunk_size=4,
                    horizontal_shard_size=8,
                )
            ]
        )
    )
    with pytest.raises(errors.InvalidConfigError, match="command-line output options") as err:
        main.apply_output_overrides(
            config,
            enable_output=None,
            output_path=None,
            output_backend=common_io.OutputBackend.NETCDF,
            output_mode=None,
        )
    assert "'zarr' backend" in str(err.value)
