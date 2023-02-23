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

from icon4py.liskov.logger import setup_logger
from icon4py.liskov.pipeline import (
    load_gt4py_stencils,
    parse_fortran_file,
    run_code_generation,
)


logger = setup_logger(__name__)


@click.command("icon_liskov")
@click.argument(
    "filepath",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--profile", "-p", is_flag=True, help="Add nvtx profile statements to stencils."
)
def main(filepath: pathlib.Path, profile: bool) -> None:
    """Command line interface for interacting with the ICON-Liskov DSL Preprocessor.

    Usage:
        icon_liskov <filepath> [--profile]

    Options:
        -p --profile Add nvtx profile statements to stencils.

    Arguments:
        filepath Path to the input file to process.
    """
    parsed = parse_fortran_file(filepath)
    parsed_checked = load_gt4py_stencils(parsed)
    run_code_generation(parsed_checked, filepath, profile)


if __name__ == "__main__":
    main()
