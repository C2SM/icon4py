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
import sys
from pathlib import Path
from typing import Sequence

import icon4py.liskov.parsing.types as ts
from icon4py.liskov.common import Step
from icon4py.liskov.logger import setup_logger
from icon4py.liskov.parsing.exceptions import UnsupportedDirectiveError
from icon4py.liskov.parsing.validation import VALIDATORS


REPLACE_CHARS = [ts.DIRECTIVE_IDENT, "&", "\n"]

logger = setup_logger(__name__)


class DirectivesParser(Step):
    def __init__(self, filepath: Path) -> None:
        """Initialize a DirectivesParser instance.

        This class parses a Sequence of RawDirective objects and returns a dictionary of parsed directives and their associated content.

        Args:
            directives: Sequence of RawDirective objects to parse.
            filepath: Path to file being parsed.
        """
        self.filepath = filepath

    def __call__(self, directives: list[ts.RawDirective]) -> ts.ParsedDict:
        """Parse the directives and return a dictionary of parsed directives and their associated content.

        Returns:
            ParsedType: Dictionary of parsed directives and their associated content.
        """
        logger.info(f"Parsing DSL Preprocessor directives at {self.filepath}")
        if len(directives) != 0:
            typed = self._determine_type(directives)
            preprocessed = self._preprocess(typed)
            self._run_validation_passes(preprocessed)
            return dict(directives=preprocessed, content=self._parse(preprocessed))
        logger.warning(f"No DSL Preprocessor directives found in {self.filepath}")
        sys.exit()

    @staticmethod
    def _determine_type(
        raw_directives: Sequence[ts.RawDirective],
    ) -> Sequence[ts.ParsedDirective]:
        """Determine the type of each RawDirective and return a Sequence of ParsedDirective objects."""
        typed = []
        for raw in raw_directives:
            found = False
            for directive in ts.SUPPORTED_DIRECTIVES:
                if directive.pattern in raw.string:
                    typed.append(directive(raw.string, raw.startln, raw.endln))  # type: ignore
                    found = True
                    break
            if not found:
                raise UnsupportedDirectiveError(
                    f"Used unsupported directive(s): {raw.string} on line(s) {raw.startln}."
                )
        return typed

    def _preprocess(
        self, directives: Sequence[ts.ParsedDirective]
    ) -> Sequence[ts.ParsedDirective]:
        """Preprocess the directives by removing unnecessary characters and formatting the directive strings."""
        return [
            d.__class__(self._clean_string(d.string), d.startln, d.endln)  # type: ignore
            for d in directives
        ]

    def _run_validation_passes(
        self, preprocessed: Sequence[ts.ParsedDirective]
    ) -> None:
        """Run validation passes on the directives."""
        for validator in VALIDATORS:
            validator(self.filepath).validate(preprocessed)

    @staticmethod
    def _clean_string(string: str) -> str:
        """Remove leading or trailing whitespaces, and words from the REPLACE_CHARS list."""
        return " ".join([c for c in string.strip().split() if c not in REPLACE_CHARS])

    @staticmethod
    def _parse(directives: Sequence[ts.ParsedDirective]) -> ts.ParsedContent:
        """Parse directives and return a dictionary of parsed directives type names and their associated content."""
        parsed_content = collections.defaultdict(list)
        for d in directives:
            content = d.get_content()
            parsed_content[d.type_name].append(content)
        return parsed_content
