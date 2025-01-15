# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
from typing import Optional

import click

from icon4pytools.f2ser.deserialise import ParsedGranuleDeserialiser
from icon4pytools.f2ser.parse import GranuleParser
from icon4pytools.liskov.codegen.serialisation.generate import SerialisationCodeGenerator
from icon4pytools.liskov.codegen.shared.write import CodegenWriter


@click.command("icon_f2ser")
@click.argument(
    "granule_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
@click.argument(
    "output_filepath",
    type=click.Path(dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
@click.option(
    "--dependencies",
    "-d",
    multiple=True,
    type=click.Path(exists=True),
    help="Optional list of dependency paths.",
)
@click.option(
    "--directory",
    type=str,
    help="Directory to serialise variables to.",
    default=".",
)
@click.option(
    "--prefix",
    type=str,
    help="Prefix to use for serialised files.",
    default="f2ser",
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
        GRANULE_PATH: A path to the Fortran source file to be parsed.
        OUTPUT_FILEPATH: A path to the output Fortran source file to be generated.
    """
    parsed = GranuleParser(granule_path, dependencies)()
    interface = ParsedGranuleDeserialiser(parsed, directory=directory, prefix=prefix)()
    generated = SerialisationCodeGenerator(interface, multinode=multinode)()
    CodegenWriter(granule_path, output_filepath)(generated)


if __name__ == "__main__":
    main()
