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

from icon4py.bindings.utils import format_fortran_code
from icon4py.liskov.codegen import WrapRunFunc, WrapRunFuncGenerator
from icon4py.liskov.collect import DirectivesCollector
from icon4py.liskov.parser import DirectivesParser
from icon4py.liskov.serialise import DirectiveSerialiser


@click.command("icon_liskov")
@click.argument(
    "filepath",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
def main(filepath: pathlib.Path) -> None:
    """Command line interface to interact with the ICON-Liskov DSL Preprocessor.

    Args:
        filepath: path to Fortran file.
    """
    directives_collector = DirectivesCollector(filepath)

    parser = DirectivesParser(directives_collector.directives)

    serialiser = DirectiveSerialiser(parser.parsed_directives)

    print(serialiser.directives)

    # todo: generate code using IntegrationGenerator
    for stencil in serialiser.directives.stencil:
        wrap_run_func = WrapRunFunc(stencil_data=stencil)
        source = WrapRunFuncGenerator.apply(wrap_run_func)
        formatted_source = format_fortran_code(source)
        print(formatted_source)

    # todo: write code using IntegrationWriter
