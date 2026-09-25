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
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver, driver_utils, main

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


_TEST_CONFIG_FILE = pathlib.Path(__file__).parent / "data" / "test_config.yml"


@pytest.fixture
def backend_name(pytestconfig: pytest.Config) -> str:
    spec = pytestconfig.getoption("backend", model_backends.DEFAULT_BACKEND)
    if spec not in model_backends.BACKENDS:
        pytest.skip(f"'{spec}' is not a backend name the driver CLI accepts")
    return spec


def test_main_cli(
    tmp_path: pathlib.Path,
    backend: gtx_typing.Backend,
    backend_name: str,
) -> None:
    cli_runner = CliRunner()
    grid_file_path = tmp_path / "dummy_grid.nc"
    mock_grid_manager = mock.MagicMock()

    with (
        mock.patch.object(
            driver_utils, "create_grid_manager", return_value=mock_grid_manager
        ) as mock_create_grid_manager,
        mock.patch.object(driver, "run_driver") as mock_run_driver,
    ):
        result = cli_runner.invoke(
            main.app,
            [
                "--grid-file-path",
                str(grid_file_path),
                "--config-file-path",
                str(_TEST_CONFIG_FILE),
                "--icon4py-backend",
                backend_name,
            ],
        )

        assert result.exit_code == 0, result.output or result.exception
        mock_create_grid_manager.assert_called_once()
        mock_run_driver.assert_called_once()
        run_driver_kwargs = mock_run_driver.call_args.kwargs
        assert run_driver_kwargs["grid_manager"] is mock_grid_manager
        assert data_alloc.backend_name(run_driver_kwargs["backend"]) == data_alloc.backend_name(
            backend
        )
        assert run_driver_kwargs["config"].driver.experiment_name == "foo"
