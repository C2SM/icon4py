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

from icon4py.model.common import model_backends, model_options
from icon4py.model.standalone_driver import config as driver_config, driver_utils, standalone_driver


log = logging.getLogger(__name__)


def main(
    *,
    grid_file_path: Annotated[pathlib.Path, typer.Option(help="Grid file path.")],
    config_file_path: Annotated[pathlib.Path, typer.Option(help="Configuration file path.")],
    output_path: Annotated[
        pathlib.Path | None,
        typer.Option(help="Optional override output path. Normally read from config."),
    ] = None,
    # it may be better to split device from backend,
    # or only asking for cpu or gpu and the best backend for perfornamce is handled inside icon4py,
    # whether to automatically use gpu if cupy is installed can be discussed further
    icon4py_backend: Annotated[
        str | model_backends.BackendLike,
        typer.Option(
            help=f"GT4Py backend for running the entire driver. Possible options are: {' / '.join([*model_backends.BACKENDS.keys()])}",
        ),
    ],
    log_level: Annotated[
        str,
        typer.Option(
            help=f"Logging level of the model. Possible options are {' / '.join([*driver_utils._LOGGING_LEVELS.keys()])}",
        ),
    ] = next(iter(driver_utils._LOGGING_LEVELS.keys())),
    print_distributed_debug_msg: Annotated[
        bool,
        typer.Option(
            help="Print out debug logging message for all ranks (only works when log_level is set to debug).",
        ),
    ] = False,
) -> None:
    """
    CLI entry point that runs the icon4py driver.

    The configuration is read from ``config_file_path``, the driver is
    initialized, an initial condition is generated, and the time integration is
    run.
    """

    backend = model_options.customize_backend(
        program=None, backend=driver_utils.get_backend_from_name(icon4py_backend)
    )
    allocator = model_backends.get_allocator(backend)

    process_props = standalone_driver.setup_environment(
        log_level=log_level,
        print_distributed_debug_msg=print_distributed_debug_msg,
    )

    config = driver_config.read_config(config_file_path)
    if output_path is not None:
        config = config.with_driver_overrides(output_path=output_path)

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )

    standalone_driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )


if __name__ == "__main__":
    typer.run(main)
