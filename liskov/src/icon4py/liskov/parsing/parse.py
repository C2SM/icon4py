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
from copy import deepcopy
from itertools import product
from pathlib import Path

from icon4py.liskov.logger import setup_logger
from icon4py.liskov.parsing.exceptions import ParsingExceptionHandler
from icon4py.liskov.parsing.types import (
    Declare,
    Directive,
    DirectiveType,
    EndCreate,
    EndStencil,
    Imports,
    NoDirectivesFound,
    ParsedContent,
    ParsedDict,
    RawDirective,
    StartCreate,
    StartStencil,
)
from icon4py.liskov.parsing.validation import (
    DirectiveSemanticsValidator,
    DirectiveSyntaxValidator,
)


_SUPPORTED_DIRECTIVES: list[Directive] = [
    StartStencil,
    EndStencil,
    Imports,
    Declare,
    StartCreate,
    EndCreate,
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

    def _determine_type(
        self, raw_directives: list[RawDirective]
    ) -> list[DirectiveType]:
        """Determine the type of each directive and return a list of TypedDirective objects."""
        typed = [
            d(raw.string, raw.startln, raw.endln)
            for raw, d in product(raw_directives, _SUPPORTED_DIRECTIVES)
            if d.pattern in raw.string
        ]
        unsupported = [
            raw
            for raw in raw_directives
            if all(d.pattern not in raw.string for d in _SUPPORTED_DIRECTIVES)
        ]
        self.exception_handler.find_unsupported_directives(unsupported)
        return typed

    @staticmethod
    def _preprocess(directives: list[DirectiveType]) -> list[DirectiveType]:
        """Preprocess the directives by removing unnecessary characters and formatting the directive strings."""
        copy = deepcopy(directives)
        preprocessed = []
        for d in copy:
            new_string = (
                d.string.strip().replace("&", "").replace("\n", "").replace("!$DSL", "")
            )
            new_string = " ".join(new_string.split())
            d.string = new_string
            preprocessed.append(d)
        return preprocessed

    def _run_validation_passes(self, preprocessed: list[DirectiveType]) -> None:
        """Run validation passes on the preprocessed directives."""
        for validator in _VALIDATORS:
            validator(self.filepath).validate(preprocessed)

    @staticmethod
    def _parse(directives: list[DirectiveType]) -> ParsedContent:
        """Parse the directives and return a dictionary of parsed directives and their associated content."""
        parsed_content = collections.defaultdict(list)
        for d in directives:
            name = d.__class__.__name__
            content = d.get_content()
            parsed_content[name].append(content)
        return parsed_content
