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
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Type

import icon4pytools.liskov.parsing.types as ts
from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.parsing.exceptions import UnsupportedDirectiveError
from icon4pytools.liskov.parsing.types import ParsedDirective, RawDirective
from icon4pytools.liskov.parsing.validation import VALIDATORS
from icon4pytools.liskov.pipeline.definition import Step

logger = setup_logger(__name__)


class DirectivesParser(Step):
    def __init__(self, input_filepath: Path, output_filepath: Path) -> None:
        """Initialize a DirectivesParser instance.

        This class parses a Sequence of RawDirective objects and returns a dictionary of parsed directives and their associated content.

        Args:
            directives: Sequence of RawDirective objects to parse.
            input_filepath Path to the input file to process.
            output_filepath Path to the output file to generate.
        """
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def __call__(self, directives: list[ts.RawDirective]) -> ts.ParsedDict:
        """Parse the directives and return a dictionary of parsed directives and their associated content.

        Returns:
            ParsedType: Dictionary of parsed directives and their associated content.
        """
        logger.info(f"Parsing DSL Preprocessor directives at {self.input_filepath}")
        if len(directives) != 0:
            typed = self._determine_type(directives)
            preprocessed = self._preprocess(typed)
            self._run_validation_passes(preprocessed)
            return dict(directives=preprocessed, content=self._parse(preprocessed))
        logger.warning(
            f"No DSL Preprocessor directives found in {self.input_filepath}, copying to {self.output_filepath}"
        )
        shutil.copyfile(self.input_filepath, self.output_filepath)
        sys.exit()

    @staticmethod
    def _determine_type(
        raw_directives: Sequence[ts.RawDirective],
    ) -> Sequence[ts.ParsedDirective]:
        """Determine the type of each RawDirective and return a Sequence of ParsedDirective objects."""
        typed = []
        for raw in raw_directives:
            found = False
            for directive in SUPPORTED_DIRECTIVES:
                if directive.pattern in raw.string:
                    typed.append(directive(raw.string, raw.startln, raw.endln))
                    found = True
                    break
            if not found:
                raise UnsupportedDirectiveError(
                    f"Used unsupported directive(s): {raw.string} on line(s) {raw.startln}."
                )
        return typed

    def _preprocess(self, directives: Sequence[ts.ParsedDirective]) -> Sequence[ts.ParsedDirective]:
        """Preprocess the directives by removing unnecessary characters and formatting the directive strings."""
        return [
            d.__class__(self._clean_string(d.string, d.type_name), d.startln, d.endln)
            for d in directives
        ]

    def _run_validation_passes(self, preprocessed: Sequence[ts.ParsedDirective]) -> None:
        """Run validation passes on the directives."""
        for validator in VALIDATORS:
            validator(self.input_filepath).validate(preprocessed)

    @staticmethod
    def _clean_string(string: str, type_name: str) -> str:
        """Remove leading or trailing whitespaces, and words from the REPLACE_CHARS list."""
        replace_chars = [ts.DIRECTIVE_IDENT]

        # DSL INSERT Statements should be inserted verbatim meaning no string cleaning
        # other than the directive identifier.
        if type_name != "Insert":
            replace_chars += ["&", "\n"]

        return " ".join([c for c in string.strip().split() if c not in replace_chars])

    @staticmethod
    def _parse(directives: Sequence[ts.ParsedDirective]) -> ts.ParsedContent:
        """Parse directives and return a dictionary of parsed directives type names and their associated content."""
        parsed_content = collections.defaultdict(list)
        for d in directives:
            content = d.get_content()
            parsed_content[d.type_name].append(content)
        return parsed_content


class TypedDirective(RawDirective):
    pattern: str

    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedDirective):
            raise NotImplementedError
        return self.string == other.string


@dataclass(eq=False)
class WithArguments(TypedDirective):
    regex: str = field(default=r"(.+?)=(.+?)", init=False)

    def get_content(self) -> dict[str, str]:
        args = self.string.replace(f"{self.pattern}", "")
        delimited = args[1:-1].split(";")
        content = {
            strip_whitespace(a.split("=")[0].strip()): strip_whitespace(a.split("=")[1])
            for a in delimited
        }
        return content


@dataclass(eq=False)
class WithOptionalArguments(TypedDirective):
    regex: str = field(default=r"(?:.+?=.+?|)", init=False)

    def get_content(self) -> Optional[dict[str, str]]:
        args = self.string.replace(f"{self.pattern}", "")[1:-1]
        if len(args) > 0:
            content = dict([args.split("=")])
            return content
        return None


@dataclass(eq=False)
class WithoutArguments(TypedDirective):
    # matches an empty string at the beginning of a line
    regex: str = field(default=r"^(?![\s\S])", init=False)

    def get_content(self) -> dict:
        return {}


@dataclass(eq=False)
class FreeForm(TypedDirective):
    # matches any string inside brackets
    regex: str = field(default=r"(.+?)", init=False)

    def get_content(self) -> str:
        args = self.string.replace(f"{self.pattern}", "")
        return args[1:-1]


def strip_whitespace(string: str) -> str:
    """
    Remove all whitespace characters from the given string.

    Args:
        string: The string to remove whitespace from.

    Returns:
        The input string with all whitespace removed.
    """
    return "".join(string.split())


class StartStencil(WithArguments):
    pattern = "START STENCIL"


class EndStencil(WithArguments):
    pattern = "END STENCIL"


class StartFusedStencil(WithArguments):
    pattern = "START FUSED STENCIL"


class EndFusedStencil(WithArguments):
    pattern = "END FUSED STENCIL"


class Declare(WithArguments):
    pattern = "DECLARE"


class Imports(WithoutArguments):
    pattern = "IMPORTS"


class StartCreate(WithOptionalArguments):
    pattern = "START CREATE"


class EndCreate(WithoutArguments):
    pattern = "END CREATE"


class EndIf(WithoutArguments):
    pattern = "ENDIF"


class StartProfile(WithArguments):
    pattern = "START PROFILE"


class EndProfile(WithoutArguments):
    pattern = "END PROFILE"


class StartDelete(WithoutArguments):
    pattern = "START DELETE"


class EndDelete(WithoutArguments):
    pattern = "END DELETE"


class Insert(FreeForm):
    pattern = "INSERT"


SUPPORTED_DIRECTIVES: Sequence[Type[ParsedDirective]] = [
    StartStencil,
    EndStencil,
    StartFusedStencil,
    EndFusedStencil,
    StartDelete,
    EndDelete,
    Imports,
    Declare,
    StartCreate,
    EndCreate,
    EndIf,
    StartProfile,
    EndProfile,
    Insert,
]
