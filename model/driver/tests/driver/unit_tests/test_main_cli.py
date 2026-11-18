# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import pathlib
import signal
from collections.abc import Iterator
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from typer.testing import CliRunner, Result

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.io import io_ranks as common_io_ranks
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


def _config_file_with_io_ranks(
    directory: pathlib.Path, io_ranks_type: str, count: int
) -> pathlib.Path:
    """The configuration with the 'io' section, dedicating ranks to output as given."""
    config_file = _config_file_with_io_section(directory)
    text = config_file.read_text()
    default = "  io_ranks:\n"
    assert text.count(default) == 1
    config_file.write_text(
        text.replace(default, f"  io_ranks:\n    type: {io_ranks_type}\n    count: {count}\n")
    )
    return config_file


@pytest.fixture
def backend_name(pytestconfig: pytest.Config) -> str:
    spec = pytestconfig.getoption("backend", model_backends.DEFAULT_BACKEND)
    if spec not in model_backends.BACKENDS:
        pytest.skip(f"'{spec}' is not a backend name the driver CLI accepts")
    return spec


def _invoke_cli(
    config_file: pathlib.Path,
    backend_name: str,
    *options: str,
    run_driver_error: BaseException | None = None,
) -> tuple[Result, mock.MagicMock, mock.MagicMock]:
    """Invoke the driver CLI with the grid manager and the driver run mocked away."""
    mock_grid_manager = mock.MagicMock()
    with (
        mock.patch.object(driver_utils, "create_grid_manager", return_value=mock_grid_manager),
        mock.patch.object(driver, "run_driver", side_effect=run_driver_error) as mock_run_driver,
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
    return result, mock_grid_manager, mock_run_driver


def _run_cli(
    config_file: pathlib.Path, backend_name: str, *options: str
) -> tuple[mock.MagicMock, mock.MagicMock]:
    """Invoke the driver CLI like ``_invoke_cli`` and require success."""
    result, mock_grid_manager, mock_run_driver = _invoke_cli(config_file, backend_name, *options)
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
    # no dedicated IO ranks: every rank runs the model, on the run's communicator
    assert isinstance(run_driver_kwargs["rank_roles"], common_io_ranks.NoIORanks)
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


_SINGLE_NODE_PROPS = decomp_defs.SingleNodeProcessProperties()


def _io_rank_split(
    is_io_rank: bool, run_props: decomp_defs.ProcessProperties = _SINGLE_NODE_PROPS
) -> common_io_ranks.IORankSplit:
    # neither the role communicator nor the inter-communicator is touched: the grid
    # manager, the driver run and the IO server are mocked away
    return common_io_ranks.IORankSplit(
        run_props=run_props,
        assignment=common_io_ranks.assign_io_ranks_per_node(_NODE_IDS, 1),
        is_io_rank=is_io_rank,
        process_props=decomp_defs.SingleNodeProcessProperties(),  # type: ignore[arg-type]  # stand-in for the role communicator
        intercomm=object(),  # type: ignore[arg-type]  # not used
        io_rank=None if is_io_rank else 0,
        served_compute_ranks=(0,) if is_io_rank else (),
    )


#: Node of every rank of the run the split is mocked for: two nodes, the first with one
#: rank more.
_NODE_IDS = [0] * 5 + [1] * 4


@contextlib.contextmanager
def _mocked_split(split: common_io_ranks.IORankSplit) -> Iterator[mock.MagicMock]:
    """Mock the split of a run on ``_NODE_IDS`` away; yields the mock, which returns ``split``."""
    with (
        mock.patch.object(common_io_ranks, "allgather_node_ids", return_value=_NODE_IDS),
        mock.patch.object(common_io_ranks, "split_io_ranks", return_value=split) as mock_split,
    ):
        yield mock_split


def test_main_cli_output_without_io_ranks_leaves_the_communicator_alone(
    backend_name: str, tmp_path: pathlib.Path
) -> None:
    """Output written by the ranks running the model themselves: no split."""
    with _mocked_split(_io_rank_split(is_io_rank=False)) as mock_split:
        _, mock_run_driver = _run_cli(_config_file_with_io_section(tmp_path), backend_name)
    mock_split.assert_not_called()
    kwargs = mock_run_driver.call_args.kwargs
    assert kwargs["config"].io is not None and kwargs["config"].io.io_ranks is None
    assert isinstance(kwargs["rank_roles"], common_io_ranks.NoIORanks)


@pytest.mark.parametrize(
    "io_ranks_type, expected_io_ranks",
    [("per_node", (4, 8)), ("total", (4,))],
    ids=["per-node", "in-total"],
)
def test_main_cli_compute_rank_runs_the_driver_on_its_role_communicator(
    backend_name: str,
    tmp_path: pathlib.Path,
    *,
    io_ranks_type: str,
    expected_io_ranks: tuple[int, ...],
) -> None:
    split = _io_rank_split(is_io_rank=False)
    config_file = _config_file_with_io_ranks(tmp_path, io_ranks_type, 1)
    with _mocked_split(split) as mock_split:
        _, mock_run_driver = _run_cli(config_file, backend_name)
    # the communicator is split by the IO ranks the configuration asks for
    mock_split.assert_called_once()
    assert mock_split.call_args.args[1].io_ranks == expected_io_ranks
    kwargs = mock_run_driver.call_args.kwargs
    # the model runs with the roles of the split, i.e. on the compute communicator
    assert kwargs["rank_roles"] is split
    assert kwargs["config"].io.io_ranks is not None


def test_main_cli_io_rank_runs_the_io_server(backend_name: str, tmp_path: pathlib.Path) -> None:
    split = _io_rank_split(is_io_rank=True)
    config_file = _config_file_with_io_ranks(tmp_path, "per_node", 2)
    with (
        _mocked_split(split),
        mock.patch.object(driver_io, "run_io_server") as mock_run_io_server,
    ):
        _, mock_run_driver = _run_cli(config_file, backend_name)
    mock_run_driver.assert_not_called()
    mock_run_io_server.assert_called_once()
    kwargs = mock_run_io_server.call_args.kwargs
    assert kwargs["rank_roles"] is split
    assert kwargs["grid_file_path"] == config_file.parent / "dummy_grid.nc"
    # the IO rank reads the same experiment configuration (for the vertical grid)
    assert kwargs["config"].io is not None
    assert [group.basename for group in kwargs["config"].io.field_groups] == [
        "winds",
        "precipitation",
    ]


@pytest.mark.parametrize("is_io_rank", [False, True], ids=["compute-rank", "io-rank"])
def test_main_cli_logs_the_layout_of_the_roles_once(
    backend_name: str, tmp_path: pathlib.Path, is_io_rank: bool
) -> None:
    """Rank 0 of the compute ranks logs where the roles run; the IO ranks do not repeat it."""
    split = _io_rank_split(is_io_rank=is_io_rank)
    with (
        _mocked_split(split),
        mock.patch.object(driver_io, "run_io_server"),
        mock.patch.object(common_io_ranks, "log_layout") as mock_log_layout,
    ):
        _run_cli(_config_file_with_io_ranks(tmp_path, "per_node", 1), backend_name)
    if is_io_rank:
        mock_log_layout.assert_not_called()
    else:
        mock_log_layout.assert_called_once_with(split.assignment)


def _process_props_with_mock_comm() -> mock.MagicMock:
    """Process properties of a one-rank run whose communicator records 'Abort' calls."""
    return mock.MagicMock(rank=0, comm_size=1, comm_name="")


@pytest.mark.parametrize("is_io_rank", [False, True], ids=["compute-rank", "io-rank"])
def test_main_cli_failure_with_io_ranks_aborts_the_run(
    backend_name: str, tmp_path: pathlib.Path, is_io_rank: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither role can observe the other's failure: a failing rank aborts the whole run."""
    # the abort arms an alarm that ends a rank whose abort blocks; the stand-in
    # communicator's abort returns, and a real alarm would end pytest
    alarms: list[int] = []
    monkeypatch.setattr(signal, "alarm", alarms.append)
    process_props = _process_props_with_mock_comm()
    failure = RuntimeError("the rank failed")
    with (
        mock.patch.object(decomp_defs, "get_process_properties", return_value=process_props),
        _mocked_split(_io_rank_split(is_io_rank=is_io_rank, run_props=process_props)),
        mock.patch.object(driver_io, "run_io_server", side_effect=failure if is_io_rank else None),
    ):
        result, _, _ = _invoke_cli(
            _config_file_with_io_ranks(tmp_path, "per_node", 1),
            backend_name,
            run_driver_error=None if is_io_rank else failure,
        )
    process_props.comm.Abort.assert_called_once_with(1)
    assert len(alarms) == 1
    assert result.exception is failure


def test_main_cli_failure_without_io_ranks_is_not_aborted(
    backend_name: str, tmp_path: pathlib.Path
) -> None:
    """Without IO ranks a failure propagates as before: every rank raises it itself."""
    process_props = _process_props_with_mock_comm()
    failure = RuntimeError("the model failed")
    with mock.patch.object(decomp_defs, "get_process_properties", return_value=process_props):
        result, _, _ = _invoke_cli(
            _config_file_with_io_section(tmp_path), backend_name, run_driver_error=failure
        )
    process_props.comm.Abort.assert_not_called()
    assert result.exception is failure
