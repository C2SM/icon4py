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
from typing import Any

import icon4pytools.liskov.parsing.types as ts
from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.parsing.exceptions import DirectiveSyntaxError
from icon4pytools.liskov.pipeline.definition import Step


logger = setup_logger(__name__)


@dataclass(frozen=True)
class Scanned:
    string: str
    lnumber: int


class DirectivesScanner(Step):
    def __init__(self, input_filepath: Path) -> None:
        r"""Class for scanning a file for ICON-Liskov DSL directives.

        A directive must start with !$DSL <DIRECTIVE_NAME>( with the
        directive arguments delimited by a ;. The directive if on multiple
        lines must include a & at the end of the line. The directive
        must always be closed by a closing bracket ). A directive can be
        commented out by using a ! before the directive,
        for example, !!$DSL means the directive is disabled.

        Example:
            !$DSL IMPORTS()

            !$DSL START STENCIL(name=single; b=test)

            !$DSL START STENCIL(name=multi; &\n
            !$DSL               b=test)

        Args:
            input_filepath: Path to file to scan for directives.
        """
        self.input_filepath = input_filepath

    def __call__(self, data: Any = None) -> list[ts.RawDirective]:
        """Scan filepath for directives and return them along with their line numbers.

        Returns:
            A list of RawDirective objects containing the scanned directives and their line numbers.
        """
        directives = []
        with self.input_filepath.open() as f:
            scanned_directives = []
            lines = f.readlines()
            for lnumber, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(ts.DIRECTIVE_IDENT):
                    eol = stripped[-1]
                    scanned = Scanned(line, lnumber)
                    scanned_directives.append(scanned)

                    match eol:
                        case ")":
                            directives.append(self._process_scanned(scanned_directives))
                            scanned_directives = []
                        case "&":
                            next_line = self._peek_directive(lines, lnumber)
                            if ts.DIRECTIVE_IDENT not in next_line:
                                raise DirectiveSyntaxError(
                                    f"Error in directive on line number: {lnumber + 1}\n Invalid use of & in single line "
                                    f"directive in file {self.input_filepath} ."
                                )
                            continue
                        case _:
                            raise DirectiveSyntaxError(
                                f"Error in directive on line number: {lnumber + 1}\n Used invalid end of line characterat in file {self.input_filepath} ."
                            )
        logger.info(f"Scanning for directives at {self.input_filepath}")
        return directives

    @staticmethod
    def _process_scanned(collected: list[Scanned]) -> ts.RawDirective:
        """Process a list of scanned directives.

        Returns
            A RawDirective object containing the concatenated directive string and its line numbers.
        """
        directive_string = "".join([c.string for c in collected])
        abs_startln, abs_endln = collected[0].lnumber, collected[-1].lnumber
        return ts.RawDirective(directive_string, startln=abs_startln, endln=abs_endln)

    @staticmethod
    def _peek_directive(lines: list[str], lnumber: int) -> str:
        """Retrieve the next line in the input file.

        This method is used to check if a directive that spans multiple lines is still a valid directive.

        Args:
            lines: List of lines from the input file.
            lnumber: Line number of the current line being processed.

        Returns:
            Next line in the input file.
        """
        return lines[lnumber + 1]
