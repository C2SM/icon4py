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

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import (
    definitions as decomp_defs,
    mpi_decomposition as mpi_decomp,
)
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


def _config_file_with_io_ranks(directory: pathlib.Path, num_io_ranks: int) -> pathlib.Path:
    """The configuration with the 'io' section, dedicating ``num_io_ranks`` ranks to output."""
    config_file = _config_file_with_io_section(directory)
    text = config_file.read_text()
    assert text.count("  num_io_ranks: 0\n") == 1
    config_file.write_text(text.replace("  num_io_ranks: 0\n", f"  num_io_ranks: {num_io_ranks}\n"))
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
    # no dedicated IO ranks: the run's communicator is used as is
    assert run_driver_kwargs["io_ranks"] is None
    config = _run_config(mock_run_driver)
    assert config.driver.experiment_name == "foo"
    # no 'io' section in the file: no output
    assert config.io is None


def test_main_cli_reads_the_io_section(backend_name: str, tmp_path: pathlib.Path) -> None:
    """The output is configured in the file only; its 'io' section reaches the driver as read."""
    _, mock_run_driver = _run_cli(_config_file_with_io_section(tmp_path), backend_name)

    config = _run_config(mock_run_driver)
    assert config.io is not None
    assert config.io.output_path == pathlib.Path("output")
    assert [group.basename for group in config.io.field_groups] == ["winds", "precipitation"]


# --------------------------------------------------------------------------------------
# Dedicated IO ranks
# --------------------------------------------------------------------------------------


def _io_rank_split(is_io_rank: bool) -> mpi_decomp.IORankSplit:
    # neither the role communicator nor the inter-communicator is touched: the grid
    # manager, the driver run and the IO server are mocked away
    return mpi_decomp.IORankSplit(
        is_io_rank=is_io_rank,
        process_props=decomp_defs.SingleNodeProcessProperties(),  # type: ignore[arg-type]  # stand-in for the role communicator
        intercomm=object(),  # type: ignore[arg-type]  # not used
    )


def test_split_io_ranks_without_io_ranks_leaves_the_communicator_alone(
    tmp_path: pathlib.Path,
) -> None:
    process_props = decomp_defs.SingleNodeProcessProperties()
    # no output at all, and output written by the ranks running the model themselves
    without_output = driver_config.read_experiment_config_from_yaml(_TEST_CONFIG_FILE)
    assert without_output.io is None
    assert main.split_io_ranks(without_output, process_props) is None
    with_output = driver_config.read_experiment_config_from_yaml(
        _config_file_with_io_section(tmp_path)
    )
    assert with_output.io is not None and with_output.io.num_io_ranks == 0
    assert main.split_io_ranks(with_output, process_props) is None


def test_main_cli_compute_rank_runs_the_driver_on_its_role_communicator(
    backend_name: str, tmp_path: pathlib.Path
) -> None:
    split = _io_rank_split(is_io_rank=False)
    with mock.patch.object(mpi_decomp, "split_io_ranks", return_value=split) as mock_split:
        _, mock_run_driver = _run_cli(_config_file_with_io_ranks(tmp_path, 2), backend_name)
    mock_split.assert_called_once()
    assert mock_split.call_args.args[1] == 2
    kwargs = mock_run_driver.call_args.kwargs
    assert kwargs["io_ranks"] is split
    # the model (and its grid decomposition) runs on the compute communicator
    assert kwargs["process_props"] is split.process_props
    assert kwargs["config"].io.num_io_ranks == 2


def test_main_cli_io_rank_runs_the_io_server(backend_name: str, tmp_path: pathlib.Path) -> None:
    split = _io_rank_split(is_io_rank=True)
    config_file = _config_file_with_io_ranks(tmp_path, 2)
    with (
        mock.patch.object(mpi_decomp, "split_io_ranks", return_value=split),
        mock.patch.object(driver_io, "run_io_server") as mock_run_io_server,
    ):
        _, mock_run_driver = _run_cli(config_file, backend_name)
    mock_run_driver.assert_not_called()
    mock_run_io_server.assert_called_once()
    kwargs = mock_run_io_server.call_args.kwargs
    assert kwargs["io_ranks"] is split
    assert kwargs["grid_file_path"] == config_file.parent / "dummy_grid.nc"
    # the IO rank reads the same experiment configuration (for the vertical grid)
    assert kwargs["config"].io is not None
    assert [group.basename for group in kwargs["config"].io.field_groups] == [
        "winds",
        "precipitation",
    ]
