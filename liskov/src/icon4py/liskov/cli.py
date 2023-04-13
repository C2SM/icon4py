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

from icon4py.common.logger import setup_logger
from icon4py.liskov.pipeline.collection import (
    load_gt4py_stencils,
    parse_fortran_file,
    run_integration_code_generation,
    run_serialisation_code_generation,
)


logger = setup_logger(__name__)


@click.command("icon_liskov")
@click.argument(
    "input_filepath",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
@click.argument(
    "output_filepath",
    type=click.Path(dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
@click.option(
    "--profile", "-p", is_flag=True, help="Add nvtx profile statements to stencils."
)
@click.option(
    "--serialise", "-s", is_flag=True, help="Generate pp_ser serialisation statements."
)
@click.option(
    "--metadatagen",
    "-m",
    is_flag=True,
    help="Add metadata header with information about program (requires git).",
)
def main(
    input_filepath: pathlib.Path,
    output_filepath: pathlib.Path,
    profile: bool,
    metadatagen: bool,
    serialise: str,
) -> None:
    """Command line interface for interacting with the ICON-Liskov DSL Preprocessor.

    Usage:
        icon_liskov <input_filepath> <output_filepath> [-p] [-m]

    Options:
        -p --profile Add nvtx profile statements to stencils.
        -m --metadatagen Add metadata header with information about program (requires git).
        -s --serialise Generates pp_ser serialisation statements for all fields in each stencil.

    Arguments:
        input_filepath Path to the input file to process.
        output_filepath Path to the output file to generate.
    """
    if serialise:
        ser_iface = parse_fortran_file(input_filepath, output_filepath, "serialisation")
        run_serialisation_code_generation(
            ser_iface, input_filepath, output_filepath, profile, metadatagen
        )

    else:
        parsed = parse_fortran_file(input_filepath, output_filepath, "integration")
        parsed_checked = load_gt4py_stencils(parsed)
        run_integration_code_generation(
            parsed_checked, input_filepath, output_filepath, profile, metadatagen
        )


if __name__ == "__main__":
    main()
