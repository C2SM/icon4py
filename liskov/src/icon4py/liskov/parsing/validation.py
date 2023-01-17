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
from abc import abstractmethod
from pathlib import Path
from typing import Protocol

from icon4py.liskov.parsing.exceptions import (
    RepeatedDirectiveError,
    RequiredDirectivesError,
    SyntaxExceptionHandler,
    UnbalancedStencilDirectiveError,
)
from icon4py.liskov.parsing.types import (
    Declare,
    EndCreate,
    EndStencil,
    Imports,
    StartCreate,
    StartStencil,
    TypedDirective,
)
from icon4py.liskov.parsing.utils import format_typed_directive


class Validator(Protocol):
    filepath: Path

    @abstractmethod
    def validate(self, directives: list[TypedDirective]) -> None:
        ...


class DirectiveSyntaxValidator:
    """Validates syntax of preprocessor directives."""

    def __init__(self, filepath: Path) -> None:
        """Initialise a DirectiveSyntaxValidator.

        Args:
            filepath: Path to file being parsed.
        """
        self.filepath = filepath
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
            self._validate_outer(d.string, d.pattern, d)
            self._validate_inner(d.string, d.pattern, d)

    def _validate_outer(
        self, to_validate: str, pattern: str, d: TypedDirective
    ) -> None:
        regex = f"{pattern}\\((.*)\\)"
        match = re.fullmatch(regex, to_validate)
        self.exception_handler.check_for_matches(d, match, regex, self.filepath)

    def _validate_inner(
        self, to_validate: str, pattern: str, d: TypedDirective
    ) -> None:
        inner = to_validate.replace(f"{pattern}", "")[1:-1].split(";")
        for arg in inner:
            match = re.fullmatch(d.regex, arg)
            self.exception_handler.check_for_matches(d, match, d.regex, self.filepath)


class DirectiveSemanticsValidator:
    """Validates semantics of preprocessor directives."""

    def __init__(self, filepath: Path) -> None:
        """Initialise a DirectiveSyntaxValidator.

        Args:
            filepath: Path to file being parsed.
        """
        self.filepath = filepath

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

    def _validate_directive_uniqueness(self, directives: list[TypedDirective]) -> None:
        """Check that all used directives are unique."""
        repeated = [d for d in directives if directives.count(d) > 1]
        if repeated:
            pretty_printed = " ".join([format_typed_directive(d) for d in repeated])
            raise RepeatedDirectiveError(
                f"Error in {self.filepath}.\n Found same directive more than once in the following directives:\n {pretty_printed}"
            )

    def _validate_required_directives(self, directives: list[TypedDirective]) -> None:
        """Check that all required directives are used at least once."""
        expected = [Declare, Imports, StartCreate, EndCreate, StartStencil, EndStencil]
        for expected_type in expected:
            if not any([isinstance(d, expected_type) for d in directives]):
                raise RequiredDirectivesError(
                    f"Error in {self.filepath}.\n Missing required directive of type {expected_type.pattern} in source."
                )

    @staticmethod
    def extract_arg_from_directive(directive: str, arg: str):
        match = re.search(f"{arg}=([^;)]+)", directive)
        if match:
            return match.group(1)
        else:
            raise ValueError(
                f"Invalid directive string, could not find '{arg}' parameter."
            )

    def _validate_stencil_directives(self, directives: list[TypedDirective]) -> None:
        """Validate that the number of start and end stencil directives match in the input `directives`.

            Also verifies that each unique stencil has a corresponding start and end directive.
            Raise an error if there are unbalanced START or END directives or if any unique stencil does not have corresponding start and end directive.

        Args:
            directives (list[TypedDirective]): List of stencil directives to validate.
        """
        start_directives = [d for d in directives if isinstance(d, StartStencil)]
        end_directives = [d for d in directives if isinstance(d, EndStencil)]

        if len(start_directives) != len(end_directives):
            raise UnbalancedStencilDirectiveError(
                f"Error in {self.filepath}.\n Found {abs(len(start_directives) - len(end_directives))} unbalanced START STENCIL or END STENCIL directives.\n"
            )

        start_stencil_names = {
            self.extract_arg_from_directive(d.string, "name") for d in start_directives
        }
        end_stencil_names = {
            self.extract_arg_from_directive(d.string, "name") for d in end_directives
        }

        if start_stencil_names != end_stencil_names:
            raise UnbalancedStencilDirectiveError(
                f"Error in {self.filepath}.\n Each unique stencil must have a corresponding START STENCIL and END STENCIL directive.\n"
            )
