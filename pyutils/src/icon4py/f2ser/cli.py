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
from icon4py.liskov.codegen.serialisation.generate import SerialisationGenerator
from icon4py.liskov.codegen.write import CodegenWriter


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
def main(
    granule_path: pathlib.Path,
    dependencies: Optional[list[pathlib.Path]],
    output_filepath: pathlib.Path,
) -> None:
    """Command line interface for interacting with the ICON-f2ser serialization Preprocessor.

    Arguments:
        granule_path (Path): A path to the Fortran source file to be parsed.
        output_filepath (Path): A path to the output Fortran source file to be generated.
    """
    parsed = GranuleParser(granule_path, dependencies).parse()
    interface = ParsedGranuleDeserialiser(parsed, directory=".").deserialise()
    generator = SerialisationGenerator(interface)
    generated = generator()
    writer = CodegenWriter(granule_path, output_filepath)
    writer(generated)


if __name__ == "__main__":
    main()
