# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib
import sys
from typing import Annotated

import typer

from icon4py.model.common import model_backends, model_options
from icon4py.model.common.decomposition import (
    definitions as decomposition_defs,
    mpi_decomposition as mpi_decomp,
)
from icon4py.model.driver import config as driver_config, driver, driver_io, driver_utils


log = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)


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
) -> None:
    """
    CLI entry point that runs the icon4py driver.

    The configuration, output included (its ``io`` section), is read from the YAML
    file at ``config_file_path``; the driver is initialized, an initial condition is
    generated, and the time integration is run. With dedicated IO ranks
    (``io.io_ranks_per_node``) the run's communicator is split first: the IO ranks only
    receive and write the compute ranks' output.
    """

    process_props = decomposition_defs.get_process_properties(
        decomposition_defs.get_runtype(with_mpi=mpi_decomp.mpi4py is not None)
    )

    config = driver_config.read_experiment_config_from_yaml(config_file_path)

    io_ranks = split_io_ranks(config, process_props)
    # each role logs with its own rank numbering and communicator name
    role_props = process_props if io_ranks is None else io_ranks.process_props
    driver_utils.configure_logging(
        logging_level=log_level,
        print_distributed_debug_msg=print_distributed_debug_msg,
        process_props=role_props,
    )

    if io_ranks is not None and io_ranks.is_io_rank:
        try:
            driver_io.run_io_server(config=config, grid_file_path=grid_file_path, io_ranks=io_ranks)
        except BaseException:
            # the compute ranks cannot observe an IO-rank failure (they block at the
            # in-flight bound at the latest): abort the whole run instead of hanging
            log.exception(f"IO rank {io_ranks.process_props.rank} failed; aborting the run.")
            process_props.comm.Abort(1)
            raise
        return

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
        process_props=role_props,
    )

    driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=role_props,
        backend=backend,
        io_ranks=io_ranks,
    )


def split_io_ranks(
    config: driver_config.ExperimentConfig, process_props: decomposition_defs.ProcessProperties
) -> mpi_decomp.IORankSplit | None:
    """Split the run's communicator for ``io.io_ranks_per_node``; None without dedicated IO ranks.

    Collective on the communicator.
    """
    if config.io is None or config.io.io_ranks_per_node == 0:
        return None
    return mpi_decomp.split_io_ranks(process_props, config.io.io_ranks_per_node)


if __name__ == "__main__":
    sys.exit(app())
