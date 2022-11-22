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

import re

from icon4py.liskov.directives import (
    Create,
    Declare,
    EndStencil,
    StartStencil,
    TypedDirective,
)
from icon4py.liskov.exceptions import ParsingException, SyntaxExceptionHandler
from icon4py.liskov.utils import escape_dollar


class DirectiveSyntaxValidator:
    """Syntax validation method dispatcher for each directive type."""

    def __init__(self):
        self.parentheses_regex = r"\((\w*?)\)"
        self.exception_handler = SyntaxExceptionHandler

    def validate(self, directives: list[TypedDirective]) -> None:
        for d in directives:
            type_name = d.directive_type.__class__.__name__
            getattr(self, type_name)(d)

    def StartStencil(self, directive: TypedDirective):
        regex = rf"{directive.directive_type.pattern}{self.parentheses_regex}"
        self._validate_syntax(directive, regex)

    def EndStencil(self, directive: TypedDirective):
        regex = rf"{directive.directive_type.pattern}{self.parentheses_regex}"
        self._validate_syntax(directive, regex)

    def Create(self, directive: TypedDirective):
        regex = directive.directive_type.pattern
        self._validate_syntax(directive, regex)

    def Declare(self, directive: TypedDirective):
        regex = directive.directive_type.pattern
        self._validate_syntax(directive, regex)

    def _validate_syntax(self, directive, regex):
        escaped = escape_dollar(regex)
        matches = re.fullmatch(escaped, directive.string)
        self.exception_handler.check_for_matches(directive, matches, regex)


class DirectiveSemanticsValidator:
    """Validates semantics of preprocessor directives."""

    def validate(self, directives: list[TypedDirective]) -> None:
        self._validate_directive_uniqueness(directives)
        self._validate_declare_create(directives)
        self._validate_stencil_directives(directives)

    @staticmethod
    def _validate_directive_uniqueness(directives: list[TypedDirective]) -> None:
        """Check that all used directives are unique."""
        unique_directives = set(directives)
        if len(unique_directives) != len(directives):
            raise ParsingException("Found same directive more than once.")

    @staticmethod
    def _validate_declare_create(directives: list[TypedDirective]) -> None:
        """Check that expected directives are used once in the code."""
        expected = [Declare, Create, StartStencil, EndStencil]
        for expected_type in expected:
            if not any(
                [isinstance(d.directive_type, expected_type) for d in directives]
            ):
                raise ParsingException("Did not use Declare or Create directive.")

    @staticmethod
    def _validate_stencil_directives(directives: list[TypedDirective]) -> None:
        """Check that number of start and end stencil directives match."""
        start_stencil_directives = [
            d for d in directives if isinstance(d.directive_type, (StartStencil))
        ]
        end_stencil_directives = [
            d for d in directives if isinstance(d.directive_type, (EndStencil))
        ]
        if len(start_stencil_directives) != len(end_stencil_directives):
            raise ParsingException(
                "Not matching number of start and end stencil directives."
            )
