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
from collections import defaultdict
from pathlib import Path
from typing import Match, Optional, Protocol, Sequence, Type

import icon4pytools.liskov.parsing.types as ts
from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.parsing import parse
from icon4pytools.liskov.parsing.exceptions import (
    DirectiveSyntaxError,
    RepeatedDirectiveError,
    RequiredDirectivesError,
    UnbalancedStencilDirectiveError,
)
from icon4pytools.liskov.parsing.utils import print_parsed_directive, remove_directive_types


logger = setup_logger(__name__)


class Validator(Protocol):
    filepath: Path

    @abstractmethod
    def validate(self, directives: Sequence[ts.ParsedDirective]) -> None:
        ...


def _extract_arg_from_directive(directive: str, arg: str) -> str:
    match = re.search(rf"{arg}\s*=\s*([^\s;)]+)", directive)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Invalid directive string, could not find '{arg}' parameter.")


class DirectiveSyntaxValidator:
    """Validates syntax of preprocessor directives."""

    def __init__(self, filepath: Path) -> None:
        """Initialise a DirectiveSyntaxValidator.

        Args:
            filepath: Path to file being parsed.
        """
        self.filepath = filepath
        self.exception_handler = SyntaxExceptionHandler

    def validate(self, directives: Sequence[ts.ParsedDirective]) -> None:
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

    def _validate_outer(self, to_validate: str, pattern: str, d: ts.ParsedDirective) -> None:
        regex = f"{pattern}\\((.*)\\)"
        match = re.fullmatch(regex, to_validate)
        self.exception_handler.check_for_matches(d, match, regex, self.filepath)

    def _validate_inner(self, to_validate: str, pattern: str, d: ts.ParsedDirective) -> None:
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

    def validate(self, directives: Sequence[ts.ParsedDirective]) -> None:
        """Validate the semantics of preprocessor directives.

        Checks that all used directives are unique, that all required directives
        are used at least once, and that the number of start and end stencil directives match.

        Args:
            directives: A list of typed directives to validate.
        """
        self._validate_directive_uniqueness(directives)
        self._validate_required_directives(directives)
        self._validate_stencil_directives(directives)

    def _validate_directive_uniqueness(self, directives: Sequence[ts.ParsedDirective]) -> None:
        """Check that all used directives are unique.

        Note: Allow repeated START STENCIL, END STENCIL and ENDIF directives.
        """
        repeated = remove_directive_types(
            [d for d in directives if directives.count(d) > 1],
            [
                parse.StartCreate,
                parse.EndCreate,
                parse.StartStencil,
                parse.EndStencil,
                parse.StartFusedStencil,
                parse.EndFusedStencil,
                parse.EndIf,
                parse.EndProfile,
                parse.StartProfile,
                parse.Insert,
            ],
        )
        if repeated:
            pretty_printed = " ".join([print_parsed_directive(d) for d in repeated])
            raise RepeatedDirectiveError(
                f"Error in {self.filepath}.\n Found same directive more than once in the following directives:\n {pretty_printed}"
            )

    def _validate_required_directives(self, directives: Sequence[ts.ParsedDirective]) -> None:
        """Check that all required directives are used at least once."""
        expected = [
            parse.Declare,
            parse.Imports,
            parse.StartStencil,
            parse.EndStencil,
        ]
        for expected_type in expected:
            if not any([isinstance(d, expected_type) for d in directives]):
                raise RequiredDirectivesError(
                    f"Error in {self.filepath}.\n Missing required directive of type {expected_type.pattern} in source."
                )

    def _validate_stencil_directives(self, directives: Sequence[ts.ParsedDirective]) -> None:
        """Validate that the number of start and end stencil directives match in the input `directives`.

            Also verifies that each unique stencil has a corresponding start and end directive.
            Raise an error if there are unbalanced START or END directives or if any unique stencil does not have corresponding start and end directive.

        Args:
            directives (Sequence[ts.ParsedDirective]): List of stencil directives to validate.
        """

        def _identify_unbalanced_directives(
            directives: Sequence[ts.ParsedDirective],
            directive_types: tuple[Type[ts.ParsedDirective], ...],
        ):
            directive_counts: dict[str, int] = defaultdict(int)
            for directive in directives:
                if isinstance(directive, directive_types):
                    directive_name = _extract_arg_from_directive(directive.string, "name")
                    directive_counts[directive_name] += (
                        1 if isinstance(directive, directive_types[0]) else -1
                    )

            unbalanced_directives = [name for name, count in directive_counts.items() if count != 0]
            if unbalanced_directives:
                error_msg = f"Each unique stencil must have a corresponding {directives[0].pattern} and {directives[1].pattern} directive."

                raise UnbalancedStencilDirectiveError(
                    f"Error in {self.filepath}. {error_msg} Errors found in the following stencils: {', '.join(unbalanced_directives)}"
                )

        directive_pairs = [
            (parse.StartStencil, parse.EndStencil),
            (parse.StartFusedStencil, parse.EndFusedStencil),
        ]

        for directive_type in directive_pairs:
            _identify_unbalanced_directives(directives, directive_type)


VALIDATORS: list = [
    DirectiveSyntaxValidator,
    DirectiveSemanticsValidator,
]


class SyntaxExceptionHandler:
    @staticmethod
    def check_for_matches(
        directive: ts.ParsedDirective,
        match: Optional[Match[str]],
        regex: str,
        filepath: Path,
    ) -> None:
        if match is None:
            raise DirectiveSyntaxError(
                f"Error in {filepath} on line {directive.startln + 1}.\n {directive.string} is invalid, "
                f"expected the following regex pattern {directive.pattern}({regex}).\n"
            )
