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
from dataclasses import dataclass
from pathlib import Path
from typing import Match, Pattern

from icon4py.liskov.directives import (
    IDENTIFIER,
    Create,
    Declare,
    DirectiveType,
    EndStencil,
    StartStencil,
)
from icon4py.liskov.exceptions import (
    DirectiveSyntaxError,
    NoDirectivesFound,
    ParsingException,
)


@dataclass(frozen=True)
class Stencil:
    name: str
    startln: int
    endln: int
    filename: Path


@dataclass(frozen=True)
class RawDirective:
    string: str
    lnumber: int


@dataclass(frozen=True)
class TypedDirective(RawDirective):
    directive_type: DirectiveType


@dataclass(frozen=True)
class ParsedDirectives:
    stencils: list[Stencil]
    declare_line: int
    create_line: int


class DirectivesCollector:
    def __init__(self, filepath: Path) -> None:
        """Class which collects all DSL directives in a given file.

        Args:
            filepath: Path to file to scan for directives.
        """
        self.filepath = filepath
        self.directives = self._collect_directives()

    def _collect_directives(self) -> list[RawDirective]:
        """Scan filepath for directives and returns them."""
        directives = []
        with self.filepath.open() as f:
            for lnumber, string in enumerate(f):
                if IDENTIFIER in string:
                    abs_lnumber = lnumber + 1
                    directives.append(RawDirective(string, abs_lnumber))
        return directives


class DirectivesParser:
    _SUPPORTED_DIRECTIVES = [StartStencil, EndStencil, Create, Declare]

    def __init__(self, filepath: Path) -> None:
        """Class which carries out end-to-end parsing of a file with regards to DSL directives.

            Handles collection and validation of preprocessor directives, returning a ParsedDirectives
            object which can be used for code generation.

        Args:
            filepath: Path to file to parse.

        Note:
            Directives which are supported by the parser can be modified by editing the self._SUPPORTED_DIRECTIVES class
            attribute.
        """
        self.filepath = filepath
        self.collector = DirectivesCollector(filepath)
        self.validator = DirectiveSyntaxValidator()
        self.exception_handler = ParsingExceptionHandler
        self.parsed_directives = self._parse_directives()

    def _parse_directives(self) -> ParsedDirectives | NoDirectivesFound:
        """Execute end-to-end parsing of collected directives."""
        if len(self.collector.directives) != 0:
            typed = self._determine_type(self._preprocess(self.collector.directives))
            self._validate_syntax(typed)
            self._validate_semantics(typed)
            stencils = self.extract_stencils(typed)
            return self._build_parsed_directives(stencils, typed)
        return NoDirectivesFound

    def _preprocess(self, directives: list[RawDirective]) -> list[RawDirective]:
        """Apply preprocessing steps to directive strings."""
        preprocessed = []
        for d in directives:
            preprocessed.append(RawDirective(d.string.strip(), d.lnumber))
        return preprocessed

    def _determine_type(self, directives: list[RawDirective]) -> list[TypedDirective]:
        """Determine type of directive used and whether it is supported."""
        typed = []
        for d in directives:
            for directive_type in self._SUPPORTED_DIRECTIVES:
                if directive_type.pattern in d.string:
                    typed.append(TypedDirective(d.string, d.lnumber, directive_type()))

        self.exception_handler.find_unsupported_directives(directives, typed)
        return typed

    def _validate_syntax(self, directives: list[TypedDirective]) -> None:
        """Validate the directive syntax using a validator."""
        for d in directives:
            type_name = d.directive_type.__class__.__name__
            getattr(self.validator, type_name)(d)

    @staticmethod
    def _validate_semantics(directives: list[TypedDirective]) -> None:
        """Validate semantics of certain directives."""
        # todo
        pass

    def extract_stencils(self, directives: list[TypedDirective]) -> list[Stencil]:
        """Extract all stencils from typed and validated directives."""
        stencils = []
        stencil_directives = self._extract_directive(
            directives, (StartStencil, EndStencil)
        )
        it = iter(stencil_directives)
        for s in it:
            start, end = s, next(it)
            string = start.string
            stencil_name = string[string.find("(") + 1 : string.find(")")]
            stencils.append(
                Stencil(
                    name=stencil_name,
                    startln=start.lnumber,
                    endln=end.lnumber,
                    filename=self.filepath,
                )
            )
        return stencils

    def _build_parsed_directives(
        self, stencils: list[Stencil], directives: list[TypedDirective]
    ) -> ParsedDirectives:
        """Build ParsedDirectives object."""
        declare = self._extract_directive(directives, Declare)
        create = self._extract_directive(directives, Create)
        return ParsedDirectives(stencils, declare.lnumber, create.lnumber)

    @staticmethod
    def _extract_directive(
        directives: list[TypedDirective],
        required_type: tuple[DirectiveType] | DirectiveType,
    ) -> list[TypedDirective] | TypedDirective:
        directives = [
            d for d in directives if isinstance(d.directive_type, required_type)
        ]
        if len(directives) == 1:
            return directives[0]
        return directives


class DirectiveSyntaxValidator:
    """Syntax validation method dispatcher for each directive type."""

    def __init__(self):
        self.parentheses_regex = r"\((\w*?)\)"
        self.exception_handler = SyntaxExceptionHandler

    def StartStencil(self, directive: TypedDirective):
        regex = rf"{directive.directive_type.pattern}{self.parentheses_regex}"
        self._validate_syntax(directive, regex)

    def EndStencil(self, directive: TypedDirective):
        regex = rf"{directive.directive_type.pattern}{self.parentheses_regex}"
        self._validate_syntax(directive, regex)

    def Create(self, directive: TypedDirective):
        regex = directive.directive_type.pattern
        self._validate_syntax(directive, regex)

    def Declare(self, directive: TypedDirective):
        regex = directive.directive_type.pattern
        self._validate_syntax(directive, regex)

    def _validate_syntax(self, directive, regex):
        matches = re.fullmatch(regex, directive.string)
        self.exception_handler.check_for_matches(directive, matches, regex)


class ParsingExceptionHandler:
    @staticmethod
    def find_unsupported_directives(
        directives: list[RawDirective], typed: list[TypedDirective]
    ) -> None:
        raw_dirs = set([d.string for d in directives])
        typed_dirs = set([t.string for t in typed])
        diff = raw_dirs.difference(typed_dirs)
        if len(diff) > 0:
            bad_directives = [d.string for d in directives if d.string in list(diff)]
            bad_lines = [str(d.lnumber) for d in directives if d.string in list(diff)]
            raise ParsingException(
                f"Used unsupported directive(s): {''.join(bad_directives)} on lines {''.join(bad_lines)}"
            )


class SyntaxExceptionHandler:
    @staticmethod
    def check_for_matches(
        directive: TypedDirective, matches: Match[str], regex: Pattern[str]
    ) -> None:
        if not matches:
            raise DirectiveSyntaxError(
                f"""DirectiveSyntaxError on line {directive.lnumber}\n
                    {directive.string} is invalid, expected {regex}\n"""
            )


class DirectiveSemanticsValidator:
    # todo: check not more than one declare directive (at least 1)
    # todo: check not more than one create directive (at least 1)
    # todo: number of StencilStart and StencilEnd must be the same
    # todo: stencil names for StencilStart must be unique
    # todo: stencil names for StencilEnd must be unique
    pass


class IntegrationClassParser:
    # todo
    pass
