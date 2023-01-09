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
from pathlib import Path

from icon4py.liskov.logger import setup_logger
from icon4py.liskov.parsing.exceptions import ParsingExceptionHandler
from icon4py.liskov.parsing.types import (
    Create,
    Declare,
    Directive,
    EndStencil,
    Imports,
    NoDirectivesFound,
    ParsedContent,
    ParsedDict,
    RawDirective,
    StartStencil,
    TypedDirective,
)
from icon4py.liskov.parsing.validation import (
    DirectiveSemanticsValidator,
    DirectiveSyntaxValidator,
)


_SUPPORTED_DIRECTIVES: list[Directive] = [
    StartStencil(),
    EndStencil(),
    Imports(),
    Declare(),
    Create(),
]

_VALIDATORS: list = [
    DirectiveSyntaxValidator,
    DirectiveSemanticsValidator,
]

logger = setup_logger(__name__)


class DirectivesParser:
    def __init__(self, directives: list[RawDirective], filepath: Path) -> None:
        """Initialize a DirectivesParser instance.

        This class parses a list of RawDirective objects and returns a dictionary of parsed directives and their associated content.

        Args:
            directives: List of RawDirective objects to parse.
            filepath: Path to file being parsed.

        Attributes:
        exception_handler: Exception handler for handling errors.
        parsed_directives: Dictionary of parsed directives and their associated content, or NoDirectivesFound if no directives were found.
        """
        self.filepath = filepath
        self.directives = directives
        self.exception_handler = ParsingExceptionHandler
        self.parsed_directives = self._parse_directives()

    def _parse_directives(self) -> ParsedDict | NoDirectivesFound:
        """Parse the directives and return a dictionary of parsed directives and their associated content.

        Returns:
            ParsedType: Dictionary of parsed directives and their associated content.
        """
        logger.info(f"Parsing DSL Preprocessor directives at {self.filepath}")
        if len(self.directives) != 0:
            typed = self._determine_type(self.directives)
            preprocessed = self._preprocess(typed)
            self._run_validation_passes(preprocessed)
            return dict(directives=preprocessed, content=self._parse(preprocessed))
        return NoDirectivesFound()

    def _determine_type(self, directives: list[RawDirective]) -> list[TypedDirective]:
        """Determine the type of each directive and return a list of TypedDirective objects."""
        typed = []
        for d in directives:
            for directive_type in _SUPPORTED_DIRECTIVES:
                if directive_type.pattern in d.string:
                    typed.append(
                        TypedDirective(d.string, d.startln, d.endln, directive_type)
                    )

        self.exception_handler.find_unsupported_directives(directives, typed)
        return typed

    @staticmethod
    def _preprocess(directives: list[TypedDirective]) -> list[TypedDirective]:
        """Preprocess the directives by removing unnecessary characters and formatting the directive strings."""
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

    def _run_validation_passes(self, preprocessed: list[TypedDirective]) -> None:
        """Run validation passes on the preprocessed directives."""
        for validator in _VALIDATORS:
            validator(self.filepath).validate(preprocessed)

    @staticmethod
    def _parse(directives: list[TypedDirective]) -> ParsedContent:
        """Parse the directives and return a dictionary of parsed directives and their associated content."""
        parsed_content = collections.defaultdict(list)

        for d in directives:
            directive_name = d.directive_type.__str__()
            pattern = d.directive_type.pattern
            string = d.string.replace(f"{pattern}", "")

            if directive_name in ["Import", "Create"]:
                content = {}
            else:
                args = string[1:-1].split(";")
                content = {a.split("=")[0].strip(): a.split("=")[1] for a in args}

            parsed_content[directive_name].append(content)
        return parsed_content
