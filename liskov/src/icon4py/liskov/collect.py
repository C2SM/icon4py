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

from icon4py.liskov.directives import IDENTIFIER, RawDirective


@dataclass(frozen=True)
class Collected:
    string: str
    lnumber: int


class DirectivesCollector:
    def __init__(self, filepath: Path) -> None:
        """Class which collects all DSL directives as is in a given file.

        Args:
            filepath: Path to file to scan for directives.
        """
        self.filepath = filepath
        self.directives = self._collect_directives()

    @staticmethod
    def _process_collected(collected: list[Collected]) -> RawDirective:
        directive_string = "".join([c.string for c in collected])
        abs_startln = collected[0].lnumber + 1
        abs_endln = collected[-1].lnumber + 1
        return RawDirective(directive_string, startln=abs_startln, endln=abs_endln)

    def _collect_directives(self) -> list[RawDirective]:
        """Scan filepath for directives and returns them along with their line numbers."""
        directives = []
        with self.filepath.open() as f:

            collected_directives = []
            for lnumber, string in enumerate(f):

                if IDENTIFIER in string:
                    stripped = string.strip()
                    eol = stripped[-1]
                    collected = Collected(string, lnumber)
                    collected_directives.append(collected)

                    match eol:
                        case ")":
                            directives.append(
                                self._process_collected(collected_directives)
                            )
                            collected_directives = []
                        case "&":
                            continue
        return directives
