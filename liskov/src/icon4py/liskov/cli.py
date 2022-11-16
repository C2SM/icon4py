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
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import click

from icon4py.liskov.input import ExternalInputs


class LiskovInterface(Protocol):
    filepath: Path

    def run(self) -> None:
        ...


@dataclass(frozen=True)
class Liskov:
    """Class which exposes the main interface to the preprocessing tool-chain.

    Args:
        filepath: Path to the file to be preprocessed.
    """

    filepath: Path

    def run(self) -> None:
        """Execute the preprocessing tool-chain."""
        integration_info = ExternalInputs(self.filepath).fetch()
        print(integration_info)
        # todo: generate code using IntegrationGenerator
        # todo: write code using IntegrationWriter
        pass


@click.command("icon_liskov")
@click.argument(
    "filepath",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
def main(filepath: pathlib.Path) -> None:
    """Command line interface to interact with the ICON-Liskov DSL Preprocessor."""
    Liskov(filepath).run()
