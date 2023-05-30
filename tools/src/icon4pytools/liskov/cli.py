# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib

import click

from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.pipeline.collection import (
    load_gt4py_stencils,
    parse_fortran_file,
    run_code_generation,
)


logger = setup_logger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """Command line interface for interacting with the ICON-Liskov DSL Preprocessor."""
    if ctx.invoked_subcommand is None:
        click.echo(
            "Need to choose one of the following commands:\nintegrate\nserialise"
        )


@main.command()
@click.option(
    "--profile",
    "-p",
    is_flag=True,
    help="Add nvtx profile statements to integration code.",
)
@click.option(
    "--metadatagen",
    "-m",
    is_flag=True,
    help="Add metadata header with information about program.",
)
@click.argument(
    "input_path",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
@click.argument(
    "output_path",
    type=click.Path(dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
def integrate(input_path, output_path, profile, metadatagen):
    mode = "integration"
    iface = parse_fortran_file(input_path, output_path, mode)
    iface_gt4py = load_gt4py_stencils(iface)
    run_code_generation(
        input_path,
        output_path,
        mode,
        iface_gt4py,
        profile=profile,
        metadatagen=metadatagen,
    )


@main.command()
@click.option(
    "--multinode",
    is_flag=True,
    type=bool,
    help="Specify whether it is a multinode run. Will generate mpi rank information.",
    default=False,
)
@click.argument(
    "input_path",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
@click.argument(
    "output_path",
    type=click.Path(dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
def serialise(input_path, output_path, multinode):
    mode = "serialisation"
    iface = parse_fortran_file(input_path, output_path, mode)
    run_code_generation(input_path, output_path, mode, iface, multinode=multinode)


if __name__ == "__main__":
    main()
