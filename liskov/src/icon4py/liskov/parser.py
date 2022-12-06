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
import collections

from icon4py.liskov.directives import (
    Create,
    Declare,
    Directive,
    EndStencil,
    NoDirectivesFound,
    RawDirective,
    StartStencil,
    TypedDirective,
)
from icon4py.liskov.exceptions import ParsingExceptionHandler
from icon4py.liskov.types import ParsedType
from icon4py.liskov.validation import (
    DirectiveSemanticsValidator,
    DirectiveSyntaxValidator,
    Validator,
)


class DirectivesParser:
    _SUPPORTED_DIRECTIVES: list[Directive] = [
        StartStencil(),
        EndStencil(),
        Create(),
        Declare(),
    ]

    _VALIDATORS: list[Validator] = [
        DirectiveSyntaxValidator(),
        DirectiveSemanticsValidator(),
    ]

    def __init__(self, directives: list[RawDirective]) -> None:
        """Class which carries out end-to-end parsing of a file with regards to DSL directives.

            Handles parsing and validation of preprocessor directives, returning a dictionary
            which can be used for code generation.

        Args:
            directives: A list of directives collected by the DirectivesCollector.

        Note:
            Directives which are supported by the parser can be modified by editing the self._SUPPORTED_DIRECTIVES class
            attribute.
        """
        self.directives = directives
        self.exception_handler = ParsingExceptionHandler
        self.parsed_directives = self._parse_directives()

    def _parse_directives(self) -> ParsedType:
        """Execute end-to-end parsing of collected directives.

        This includes type deduction, preprocessing of directives, validation, parsing, and serialisation.
        """
        if len(self.directives) != 0:
            typed = self._determine_type(self.directives)
            preprocessed = self._preprocess(typed)
            self._run_validation_passes(preprocessed)
            return dict(directives=preprocessed, content=self._parse(preprocessed))
        return NoDirectivesFound()

    def _run_validation_passes(self, preprocessed: list[TypedDirective]) -> None:
        """Run validation passes on Typed Directives."""
        for v in self._VALIDATORS:
            v.validate(preprocessed)

    def _determine_type(self, directives: list[RawDirective]) -> list[TypedDirective]:
        """Determine type of directive used and whether it is supported."""
        typed = []
        for d in directives:
            for directive_type in self._SUPPORTED_DIRECTIVES:
                if directive_type.pattern in d.string:
                    typed.append(
                        TypedDirective(d.string, d.startln, d.endln, directive_type)
                    )

        self.exception_handler.find_unsupported_directives(directives, typed)
        return typed

    @staticmethod
    def _preprocess(directives: list[TypedDirective]) -> list[TypedDirective]:
        """Apply preprocessing steps to directive strings."""
        preprocessed = []
        for d in directives:
            string = (
                d.string.strip().replace("&", "").replace("\n", "").replace("!$DSL", "")
            )
            string = " ".join(string.split())
            preprocessed.append(
                TypedDirective(string, d.startln, d.endln, d.directive_type)
            )
        return preprocessed

    @staticmethod
    def _parse(directives: list[TypedDirective]) -> dict[str, list]:
        """Parse directive into a dictionary of keys and their corresponding values."""
        parsed_content = collections.defaultdict(list)

        for d in directives:
            content = None
            directive_name = d.directive_type.__str__()

            string = d.string.replace(f"{d.directive_type.pattern}", "")

            if directive_name != "Create":
                args = string[1:-1].split(";")
                content = {a.split("=")[0].strip(): a.split("=")[1] for a in args}

            try:
                parsed_content[directive_name].append(content)
            except Exception as e:
                raise e

        return parsed_content
        # todo: add exception in case parsing fails.
