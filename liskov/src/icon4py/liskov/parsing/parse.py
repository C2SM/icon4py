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
from typing import Sequence

import icon4py.liskov.parsing.types as ts
from icon4py.liskov.logger import setup_logger
from icon4py.liskov.parsing.validation import VALIDATORS, ParsingExceptionHandler


REPLACE_CHARS = [ts.DIRECTIVE_IDENT, "&", "\n"]

logger = setup_logger(__name__)


class DirectivesParser:
    def __init__(self, directives: Sequence[ts.RawDirective], filepath: Path) -> None:
        """Initialize a DirectivesParser instance.

        This class parses a Sequence of RawDirective objects and returns a dictionary of parsed directives and their associated content.

        Args:
            directives: Sequence of RawDirective objects to parse.
            filepath: Path to file being parsed.

        Attributes:
        exception_handler: Exception handler for handling errors.
        parsed_directives: Dictionary of parsed directives and their associated content.
        """
        self.filepath = filepath
        self.directives = directives
        self.exception_handler = ParsingExceptionHandler
        self.parsed_directives = self._parse_directives()

    def _parse_directives(self) -> ts.ParsedDict:
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
        logger.warning(f"No DSL Preprocessor directives found in {self.filepath}")
        raise SystemExit

    def _determine_type(
        self, raw_directives: Sequence[ts.RawDirective]
    ) -> Sequence[ts.ParsedDirective]:
        """Determine the type of each directive and return a Sequence of TypedDirective objects."""
        typed = [
            directive(raw.string, raw.startln, raw.endln)  # type: ignore
            for raw, directive in product(raw_directives, ts.SUPPORTED_DIRECTIVES)
            if directive.pattern in raw.string
        ]
        self.exception_handler.find_unsupported_directives(raw_directives)
        return typed

    @staticmethod
    def _preprocess(
        directives: Sequence[ts.ParsedDirective],
    ) -> Sequence[ts.ParsedDirective]:
        """Preprocess the directives by removing unnecessary characters and formatting the directive strings."""
        copy = deepcopy(
            directives
        )  # todo: create new instances of ParsedDirectives here?
        for c in copy:
            for r in REPLACE_CHARS:
                c.string = c.string.replace(r, "").strip()
            c.string = " ".join(c.string.strip().split())
        return copy

    def _run_validation_passes(
        self, preprocessed: Sequence[ts.ParsedDirective]
    ) -> None:
        """Run validation passes on the preprocessed directives."""
        for validator in VALIDATORS:
            validator(self.filepath).validate(preprocessed)

    @staticmethod
    def _parse(directives: Sequence[ts.ParsedDirective]) -> ts.ParsedContent:
        """Parse the directives and return a dictionary of parsed directives and their associated content."""
        parsed_content = collections.defaultdict(list)
        for d in directives:
            name = d.__class__.__name__
            content = d.get_content()
            parsed_content[name].append(content)
        return parsed_content
