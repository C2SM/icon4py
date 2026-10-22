# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import logging
import pathlib
import sys
from typing import Annotated, Any

import typer

from icon4py.model.common import exceptions, model_backends, model_options
from icon4py.model.common.decomposition import (
    definitions as decomposition_defs,
    mpi_decomposition as mpi_decomp,
)
from icon4py.model.common.io import io as common_io
from icon4py.model.driver import config as driver_config, driver, driver_io, driver_utils


log = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)


def apply_output_overrides(
    config: driver_config.ExperimentConfig,
    *,
    enable_output: bool | None,
    output_path: pathlib.Path | None,
    output_backend: common_io.OutputBackend | None,
    output_mode: common_io.OutputMode | None,
) -> driver_config.ExperimentConfig:
    """Apply the output options of the command line; ``None`` keeps the configured value.

    ``--enable-output`` keeps the configured ``io`` section or, without one, adds the
    built-in default group (:func:`driver_io.default_io_config`); ``--no-enable-output``
    drops the section. ``--output-path`` replaces the output directory, ``--output-backend``
    and ``--output-mode`` the format and write strategy of every field group; these three
    need output, i.e. an ``io`` section or ``--enable-output``.
    """
    io_config = config.io
    if enable_output is False:
        io_config = None
    elif enable_output and io_config is None:
        io_config = driver_io.default_io_config()

    if io_config is None:
        given = [
            name
            for name, value in (
                ("--output-path", output_path),
                ("--output-backend", output_backend),
                ("--output-mode", output_mode),
            )
            if value is not None
        ]
        if given:
            raise exceptions.InvalidConfigError(
                f"{', '.join(given)} require(s) output: an 'io' section in the "
                f"configuration file or '--enable-output'."
            )
        return config if config.io is None else config.with_overrides(io=None)

    io_overrides: dict[str, Any] = {}
    if output_path is not None:
        io_overrides["output_path"] = output_path
    group_overrides: dict[str, Any] = {}
    if output_backend is not None:
        group_overrides["backend"] = output_backend
    if output_mode is not None:
        group_overrides["mode"] = output_mode
    if group_overrides:
        try:
            io_overrides["field_groups"] = [
                dataclasses.replace(group, **group_overrides) for group in io_config.field_groups
            ]
        except exceptions.InvalidConfigError as err:
            raise exceptions.InvalidConfigError(
                f"The command-line output options cannot be applied to the configured "
                f"field groups: {err}"
            ) from err
    if io_overrides:
        io_config = dataclasses.replace(io_config, **io_overrides)
    return config if io_config is config.io else config.with_overrides(io=io_config)


@app.command()
def main(
    *,
    grid_file_path: Annotated[pathlib.Path, typer.Option(help="Grid file path.")],
    config_file_path: Annotated[
        pathlib.Path,
        typer.Option(
            help=(
                "YAML configuration file path. Use "
                "'scripts/python/convert_fortran_config_to_yaml.py' to generate a YAML configuration from a "
                "directory of Fortran namelists if conversion from Fortran namelists is needed."
            )
        ),
    ],
    output_path: Annotated[
        pathlib.Path | None,
        typer.Option(help="Override the output directory ('io.output_path')."),
    ] = None,
    # it may be better to split device from backend,
    # or only asking for cpu or gpu and the best backend for perfornamce is handled inside icon4py,
    # whether to automatically use gpu if cupy is installed can be discussed further
    icon4py_backend: Annotated[
        str,
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
    enable_output: Annotated[
        bool | None,
        typer.Option(
            "--enable-output/--no-enable-output",
            help=(
                "Switch the output on (the configured 'io' section or, without one, the "
                "built-in default group) or off."
            ),
        ),
    ] = None,
    output_backend: Annotated[
        common_io.OutputBackend | None,
        typer.Option(help="Override the file format of every output field group."),
    ] = None,
    output_mode: Annotated[
        common_io.OutputMode | None,
        typer.Option(
            help="Override how the ranks of a distributed run write every output field group."
        ),
    ] = None,
) -> None:
    """
    CLI entry point that runs the icon4py driver.

    The configuration is read from the YAML file at ``config_file_path``, the
    driver is initialized, an initial condition is generated, and the time
    integration is run. The output options given on the command line override
    the configuration file; the others keep their configured values.
    """

    process_props = decomposition_defs.get_process_properties(
        decomposition_defs.get_runtype(with_mpi=mpi_decomp.mpi4py is not None)
    )
    driver_utils.configure_logging(
        logging_level=log_level,
        print_distributed_debug_msg=print_distributed_debug_msg,
        process_props=process_props,
    )

    config = apply_output_overrides(
        driver_config.read_experiment_config_from_yaml(config_file_path),
        enable_output=enable_output,
        output_path=output_path,
        output_backend=output_backend,
        output_mode=output_mode,
    )

    backend = model_options.customize_backend(
        program=None,
        backend=driver_utils.get_backend_from_name(icon4py_backend),
        backend_config=config.driver.backend_config,
    )
    allocator = model_backends.get_allocator(backend)

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )

    driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )


if __name__ == "__main__":
    sys.exit(app())
