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
from pathlib import Path
from typing import Match, Optional

from icon4py.liskov.parsing.types import RawDirective, TypedDirective


class UnsupportedDirectiveError(Exception):
    pass


class DirectiveSyntaxError(Exception):
    pass


class RepeatedDirectiveError(Exception):
    pass


class RequiredDirectivesError(Exception):
    pass


class UnbalancedStencilDirectiveError(Exception):
    pass


class MissingBoundsError(Exception):
    pass


class MissingDirectiveArgumentError(Exception):
    pass


class IncompatibleFieldError(Exception):
    pass


class UnknownStencilError(Exception):
    pass


class ParsingExceptionHandler:
    @staticmethod
    def find_unsupported_directives(
        directives: list[RawDirective], typed: list[TypedDirective]
    ) -> None:
        """Check for unsupported directives and raises an exception if any are found."""
        diff = set([d.string for d in directives]).difference(
            set([t.string for t in typed])
        )
        if len(diff) > 0:
            bad_directives = [d.string for d in directives if d.string in list(diff)]
            bad_lines = [
                str(d.startln + 1) for d in directives if d.string in list(diff)
            ]
            raise UnsupportedDirectiveError(
                f"Used unsupported directive(s): {''.join(bad_directives)} on line(s) {','.join(bad_lines)}."
            )


class SyntaxExceptionHandler:
    @staticmethod
    def check_for_matches(
        directive: TypedDirective,
        match: Optional[Match[str]],
        regex: str,
        filepath: Path,
    ) -> None:
        if match is None:
            raise DirectiveSyntaxError(
                f"Error in {filepath} on line {directive.startln + 1}.\n {directive.string} is invalid, "
                f"expected the following regex pattern {directive.directive_type.pattern}({regex}).\n"
            )
