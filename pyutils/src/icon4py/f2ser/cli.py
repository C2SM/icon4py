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
    input_filepath: pathlib.Path,
    dep_filepath: pathlib.Path, # TODO: to be removed
    output_filepath: pathlib.Path,
) -> None:
    """Command line interface for interacting with the ICON-f2ser serialization Preprocessor.

    Usage:
        icon_f2ser <input_filepath> <output_filepath>

    Options:

    Arguments:
        input_filepath Path to the input file to process.
        dep_filepath Path to the input file dependencies.
        output_filepath Path to the output file to generate.
    """

    parser = GranuleParser(input_filepath, dep_filepath)
    parsed = parser.parse()
    deserialiser = ParsedGranuleDeserialiser(parsed, directory=".")
    interface = deserialiser.deserialise()
    #run_code_generation(interface, output_filepath)

if __name__ == "__main__":
    main()
