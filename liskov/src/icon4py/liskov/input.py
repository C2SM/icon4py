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

from dataclasses import dataclass
from pathlib import Path

from icon4py.liskov.parser import DirectivesParser


@dataclass(frozen=True)
class FieldData:
    inputs: str  # todo: some field class
    outputs: str  # todo: some field class
    associations: str  # todo: association class


@dataclass(frozen=True)
class BoundsData:
    hlower: str | int
    hupper: str | int
    vlower: str | int
    vupper: str | int


@dataclass(frozen=True)
class IntegrationData:
    fields: FieldData
    bounds: BoundsData


class ExternalInputs:
    """With this class we can specify a configurable list of external inputs to icon liskov."""

    def __init__(self, filepath: Path):
        self.directive_parser = DirectivesParser(filepath)

    def fetch(self) -> tuple:
        # todo: create IntegrationData and combine it with parsed directives
        return self.directive_parser.parsed_directives
