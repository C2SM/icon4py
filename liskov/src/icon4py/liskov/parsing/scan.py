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

from icon4py.liskov.parsing.types import DIRECTIVE_TOKEN, RawDirective


@dataclass(frozen=True)
class Scanned:
    string: str
    lnumber: int


class DirectivesScanner:
    def __init__(self, filepath: Path) -> None:
        """Class which collects all DSL directives as is in a given file.

        Args:
            filepath: Path to file to scan for directives.
        """
        self.filepath = filepath
        self.directives = self._scan_for_directives()

    @staticmethod
    def _process_scanned(collected: list[Scanned]) -> RawDirective:
        directive_string = "".join([c.string for c in collected])
        abs_startln = collected[0].lnumber
        abs_endln = collected[-1].lnumber
        return RawDirective(directive_string, startln=abs_startln, endln=abs_endln)

    def _scan_for_directives(self) -> list[RawDirective]:
        """Scan filepath for directives and returns them along with their line numbers."""
        directives = []
        with self.filepath.open() as f:

            scanned_directives = []
            for lnumber, string in enumerate(f):

                if DIRECTIVE_TOKEN in string:
                    stripped = string.strip()
                    eol = stripped[-1]
                    scanned = Scanned(string, lnumber)
                    scanned_directives.append(scanned)

                    match eol:
                        case ")":
                            directives.append(self._process_scanned(scanned_directives))
                            scanned_directives = []
                        case "&":
                            continue
        return directives
