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
    RepeatedDirectiveError,
    RequiredDirectivesError,
    SyntaxExceptionHandler,
    UnbalancedStencilDirectiveError,
)
from icon4py.liskov.parsing.types import (
    Create,
    Declare,
    EndStencil,
    Imports,
    StartStencil,
    TypedDirective,
)
from icon4py.liskov.parsing.utils import pretty_print_typed_directive


class Validator(Protocol):
    def validate(self, directives: list[TypedDirective]) -> None:
        ...


class DirectiveSyntaxValidator(Validator):
    """Validates syntax of preprocessor directives."""

    def __init__(self) -> None:
        self.exception_handler = SyntaxExceptionHandler

    def validate(self, directives: list[TypedDirective]) -> None:
        """Validate the syntax of preprocessor directives.

            Checks that each directive's pattern and inner contents, if any, match the expected syntax.
            If a syntax error is detected an appropriate exception using the exception_handler attribute
            is raised.

        Args:
            directives: A list of typed directives to validate.
        """
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

        if type(d.directive_type) in [Imports, Create]:
            # matches an empty string at the beginning of a line
            regex = r"^(?![\s\S])"
        else:
            # match sequences of characters separated by an equal sign
            regex = r"(.+?)=(.+?)"

        for arg in inner:
            match = re.fullmatch(regex, arg)
            self.exception_handler.check_for_matches(d, match, regex)


class DirectiveSemanticsValidator(Validator):
    """Validates semantics of preprocessor directives."""

    def validate(self, directives: list[TypedDirective]) -> None:
        """Validate the semantics of preprocessor directives.

        Checks that all used directives are unique, that all required directives
        are used at least once, and that the number of start and end stencil directives match.

        Args:
            directives: A list of typed directives to validate.
        """
        self._validate_directive_uniqueness(directives)
        self._validate_required_directives(directives)
        self._validate_stencil_directives(directives)

    @staticmethod
    def _validate_directive_uniqueness(directives: list[TypedDirective]) -> None:
        """Check that all used directives are unique."""
        unique_directives = set(directives)
        if len(unique_directives) != len(directives):
            repeated = [d for d in directives if directives.count(d) >= 2]
            pretty_printed = " ".join(
                [pretty_print_typed_directive(d) for d in repeated]
            )
            raise RepeatedDirectiveError(
                f"Found same directive more than once in the following directives:\n {pretty_printed} "
            )

    @staticmethod
    def _validate_required_directives(directives: list[TypedDirective]) -> None:
        """Check that all required directives are used at least once."""
        expected = [Declare, Imports, Create, StartStencil, EndStencil]
        for expected_type in expected:
            if not any(
                [isinstance(d.directive_type, expected_type) for d in directives]
            ):
                raise RequiredDirectivesError(
                    f"Missing required directive of type {expected_type()} in source."
                )

    @staticmethod
    def _validate_stencil_directives(directives: list[TypedDirective]) -> None:
        """Check that number of start and end stencil directives match."""
        start_directives = list(
            filter(lambda d: isinstance(d.directive_type, StartStencil), directives)
        )
        end_directives = list(
            filter(lambda d: isinstance(d.directive_type, EndStencil), directives)
        )
        diff = abs(len(start_directives) - len(end_directives))

        if diff >= 1:
            raise UnbalancedStencilDirectiveError(
                f"Found {diff} unbalanced START or END directives.\n"
            )
