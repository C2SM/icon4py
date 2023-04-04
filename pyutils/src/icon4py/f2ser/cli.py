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

from icon4py.f2ser.logger import setup_logger
from icon4py.f2ser.deserialise import ParsedGranuleDeserialiser
from icon4py.f2ser.parse import GranuleParser

logger = setup_logger(__name__)

@click.command("icon_f2ser")
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
def main(
    granule_path: pathlib.Path,
    dependencies: Optional[list[pathlib.Path]] = None,
    output_filepath: pathlib.Path,
) -> None:
    """Command line interface for interacting with the ICON-f2ser serialization Preprocessor.

    Usage:
        icon_f2ser <granule_path> <dependencies> <output_filepath>

    Options:

    Arguments:
        granule_path (Path): A path to the Fortran source file to be parsed.
        dependencies (Optional[list[Path]]): A list of paths to any additional Fortran source files that the input file depends on.
        output_filepath (Path): A path to the output Fortran source file to be generated.
    """

    parser = GranuleParser(granule_path, dependencies)
    parsed = parser.parse()
    deserialiser = ParsedGranuleDeserialiser(parsed, directory=".")
    interface = deserialiser.deserialise()
    run_code_generation(interface, output_filepath)

if __name__ == "__main__":
    main()
