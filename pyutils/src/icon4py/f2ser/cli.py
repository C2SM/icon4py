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
from typing import Optional

import click

from icon4py.f2ser.deserialise import ParsedGranuleDeserialiser
from icon4py.f2ser.parse import GranuleParser
from icon4py.liskov.codegen.serialisation.generate import (
    SerialisationCodeGenerator,
)
from icon4py.liskov.codegen.shared.writer import CodegenWriter


@click.command("icon_f2ser")
@click.argument(
    "granule_path",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--dependencies",
    "-d",
    multiple=True,
    type=click.Path(exists=True),
    help="Optional list of dependency paths.",
)
@click.argument(
    "output_filepath",
    type=click.Path(dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
@click.option(
    "--directory", type=str, help="Directory to serialise variables to.", default="."
)
@click.option(
    "--prefix", type=str, help="Prefix to use for serialised files.", default="f2ser"
)
@click.option(
    "--multinode",
    is_flag=True,
    type=bool,
    help="Specify whether it is a multinode run.",
    default=False,
)
def main(
    granule_path: pathlib.Path,
    dependencies: Optional[list[pathlib.Path]],
    output_filepath: pathlib.Path,
    directory: str,
    prefix: str,
    multinode: bool,
) -> None:
    """Command line interface for interacting with the ICON-f2ser serialization Preprocessor.

    Arguments:
        granule_path (Path): A path to the Fortran source file to be parsed.
        output_filepath (Path): A path to the output Fortran source file to be generated.
        directory (str): The directory to serialise the variables to.
        prefix (str): The prefix to use for each serialised variable.
        multinode (bool): Specify whether this is a multinode run.
    """
    parsed = GranuleParser(granule_path, dependencies)()
    interface = ParsedGranuleDeserialiser(
        parsed, directory=directory, prefix=prefix, multinode=multinode
    )()
    generated = SerialisationCodeGenerator(interface)()
    CodegenWriter(granule_path, output_filepath)(generated)


if __name__ == "__main__":
    main()
