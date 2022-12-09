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

from icon4py.liskov.codegen.integration import IntegrationGenerator
from icon4py.liskov.collect import DirectivesCollector
from icon4py.liskov.directives import NoDirectivesFound
from icon4py.liskov.parser import DirectivesParser
from icon4py.liskov.serialise import DirectiveSerialiser


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
    """Command line interface to interact with the ICON-Liskov DSL Preprocessor."""
    directives_collector = DirectivesCollector(filepath)

    parser = DirectivesParser(directives_collector.directives)

    if isinstance(parsed_directives := parser.parsed_directives, NoDirectivesFound):
        print(f"No directives found in {filepath}")  # todo: use logger.

    serialiser = DirectiveSerialiser(parsed_directives)

    integrator = IntegrationGenerator(serialiser.directives, profile=profile)

    print(integrator.generated)

    # todo: write code using IntegrationWriter
