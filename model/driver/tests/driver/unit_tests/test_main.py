# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import unittest.mock as mock

from typer.testing import CliRunner

from icon4py.model.driver import driver, driver_utils, main


_TEST_CONFIG_FILE = pathlib.Path(__file__).parent / "data" / "test_config.yml"


def test_main_cli_invokes_run_driver(tmp_path: pathlib.Path) -> None:
    cli_runner = CliRunner()
    grid_file_path = tmp_path / "dummy_grid.nc"  # never opened: create_grid_manager is mocked
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
                "embedded",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_create_grid_manager.assert_called_once()
    mock_run_driver.assert_called_once()
    run_driver_kwargs = mock_run_driver.call_args.kwargs
    assert run_driver_kwargs["grid_manager"] is mock_grid_manager
    assert run_driver_kwargs["backend"] is None  # "embedded" backend maps to None
    assert run_driver_kwargs["config"].experiment_name == "foo"
