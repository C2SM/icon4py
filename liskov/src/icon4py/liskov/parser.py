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
from typing import Protocol


@dataclass
class Directive:
    lnumber: int
    string: str


class DirectivesInput(Protocol):
    ...
    # todo


class IntegrationClassInput(Protocol):
    ...
    # todo


class DirectivesParser:
    _DIRECTIVE_START = "!#DSL"

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.directives = self._scan_file()

    def _scan_file(self) -> list[Directive]:
        """Scan file for preprocessor directives and collect them."""
        directives = []
        with self.filepath.open() as f:
            for lnumber, string in enumerate(f):
                if self._DIRECTIVE_START in string:
                    directives.append(Directive(lnumber, string))
        return directives


class IntegrationClassParser:
    def __call__(self, *args, **kwargs) -> IntegrationClassInput:
        ...
        # todo: parses all integration classes and returns IntegrationInput
