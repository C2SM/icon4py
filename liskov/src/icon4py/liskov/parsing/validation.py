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
from typing import Protocol

from icon4py.liskov.parsing.exceptions import (
    ParsingException,
    SyntaxExceptionHandler,
)
from icon4py.liskov.parsing.types import (
    Declare,
    EndStencil,
    Imports,
    StartStencil,
    TypedDirective,
)


class Validator(Protocol):
    def validate(self, directives: list[TypedDirective]) -> None:
        ...


class DirectiveSyntaxValidator(Validator):
    """Syntax validation method dispatcher for each directive type."""

    def __init__(self) -> None:
        self.exception_handler = SyntaxExceptionHandler

    def validate(self, directives: list[TypedDirective]) -> None:
        for d in directives:
            to_validate = d.string
            pattern = d.directive_type.pattern
            self._validate_outer(to_validate, pattern, d)
            self._validate_inner(to_validate, pattern, d)

    def _validate_outer(
        self, to_validate: str, pattern: str, d: TypedDirective
    ) -> None:
        regex = f"{pattern}\\((.*)\\)"
        match = re.fullmatch(regex, to_validate)
        self.exception_handler.check_for_matches(d, match, regex)

    def _validate_inner(
        self, to_validate: str, pattern: str, d: TypedDirective
    ) -> None:

        inner = to_validate.replace(f"{pattern}", "")[1:-1].split(";")

        if type(d.directive_type) == Imports:
            regex = r"^(?![\s\S])"
        else:
            regex = r"(.+?)=(.+?)"

        for arg in inner:
            match = re.fullmatch(regex, arg)
            self.exception_handler.check_for_matches(d, match, regex)


class DirectiveSemanticsValidator(Validator):
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
        expected = [Declare, Imports, StartStencil, EndStencil]
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
