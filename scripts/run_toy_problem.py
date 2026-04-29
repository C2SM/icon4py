# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib
from typing import Annotated

import typer

from icon4py.model.common import model_backends
from icon4py.model.standalone_driver import driver_states, standalone_driver, driver_utils
from icon4py.model.standalone_driver.testcases import initial_condition
from icon4py.model.testing import grid_utils

log = logging.getLogger(__file__)

def toy_problem(
    icon4py_backend: Annotated[
        str | model_backends.BackendLike,
        typer.Option(
            help=f"Backend for running the driver. Possible options are: {' / '.join([*model_backends.BACKENDS.keys()])}",
        ),
    ],
) -> None:

    grid =
    grid_file_path = grid_utils._download_grid_file(grid)

    icon4py_driver: standalone_driver.Icon4pyDriver = standalone_driver.initialize_driver(
        grid_file_path=grid_file_path,
        log_level="info",
        backend_like=driver_utils.get_backend_from_name(icon4py_backend),
        output_path=pathlib.Path("./output"),
    )

    ds: driver_states.DriverStates = initial_condition.jablonowski_williamson(
        grid=icon4py_driver.grid,
        geometry_field_source=icon4py_driver.static_field_factories.geometry_field_source,
        interpolation_field_source=icon4py_driver.static_field_factories.interpolation_field_source,
        metrics_field_source=icon4py_driver.static_field_factories.metrics_field_source,
        backend=icon4py_driver.backend,
        lowest_layer_thickness=icon4py_driver.vertical_grid_config.lowest_layer_thickness,
        model_top_height=icon4py_driver.vertical_grid_config.model_top_height,
        stretch_factor=icon4py_driver.vertical_grid_config.stretch_factor,
        damping_height=icon4py_driver.vertical_grid_config.rayleigh_damping_height,
        exchange=icon4py_driver.exchange,
    )

if __name__ == "__main__":
    typer.run(toy_problem)
