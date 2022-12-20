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

from icon4py.liskov.parsing.types import DIRECTIVE_IDENT, RawDirective


@dataclass(frozen=True)
class Scanned:
    string: str
    lnumber: int


class DirectivesScanner:
    def __init__(self, filepath: Path) -> None:
        """Class for scanning a file for ICON-Liskov DSL directives.

        Args:
            filepath: Path to file to scan for directives.
        """
        self.filepath = filepath
        self.directives = self._scan_for_directives()

    def _scan_for_directives(self) -> list[RawDirective]:
        """Scan filepath for directives and return them along with their line numbers.

        Returns:
            A list of RawDirective objects containing the scanned directives and their line numbers.
        """
        directives = []
        with self.filepath.open() as f:

            scanned_directives = []
            for lnumber, string in enumerate(f):

                if DIRECTIVE_IDENT in string:
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

    @staticmethod
    def _process_scanned(collected: list[Scanned]) -> RawDirective:
        """Process a list of scanned directives and returns a RawDirective object containing the concatenated directive string and its line numbers."""
        directive_string = "".join([c.string for c in collected])
        abs_startln, abs_endln = collected[0].lnumber, collected[-1].lnumber
        return RawDirective(directive_string, startln=abs_startln, endln=abs_endln)
