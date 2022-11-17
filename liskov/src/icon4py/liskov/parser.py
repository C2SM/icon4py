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

from icon4py.liskov.directives import (
    IDENTIFIER,
    Create,
    Declare,
    DirectiveType,
    EndStencil,
    StartStencil,
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


class ParsingException(Exception):
    pass


class DirectiveSyntaxError(Exception):
    pass


class DirectivesCollector:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.directives = self._collect_directives()

    def _collect_directives(self):
        directives = []
        with self.filepath.open() as f:
            for lnumber, string in enumerate(f):
                if IDENTIFIER in string:
                    abs_lnumber = lnumber + 1
                    directives.append(RawDirective(string, abs_lnumber))
        return directives


class NoDirectivesFound:
    pass


class DirectivesParser:
    _SUPPORTED_DIRECTIVES = [StartStencil, EndStencil, Create, Declare]

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.collector = DirectivesCollector(filepath)
        self.parsed_directives = self._parse_directives()

    def _parse_directives(self) -> ParsedDirectives | NoDirectivesFound:
        """Parse collected directives."""
        if len(self.collector.directives) != 0:
            typed = self._determine_type(self._preprocess(self.collector.directives))
            self._validate_syntax(typed)
            self._validate_semantics(typed)
            stencils = self.extract_stencils(typed)
            return self._build_parsed_directives(stencils, typed)
        return NoDirectivesFound

    def _preprocess(self, directives: list[RawDirective]) -> list[RawDirective]:
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

        if len(directives) != len(typed):
            print("Used unsupported directive")  # todo: raise exception
        return typed

    @staticmethod
    def _validate_syntax(directives: list[TypedDirective]) -> None:
        """Validate the directive syntax using a validator."""
        for d in directives:
            type_name = d.directive_type.__class__.__name__
            getattr(DirectiveSyntaxValidator(), type_name)(d)

    @staticmethod
    def _validate_semantics(directives: list[TypedDirective]) -> None:
        """Validate semantics of certain directives."""
        # todo
        pass

    def extract_stencils(self, directives: list[TypedDirective]) -> list[Stencil]:
        """Extract all stencils from typed and validated directives."""
        stencils = []
        stencils_directives = self._extract_directive(
            directives, (StartStencil, EndStencil)
        )
        it = iter(stencils_directives)
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
        declare = self._extract_directive(directives, Declare)[0]
        create = self._extract_directive(directives, Create)[0]
        return ParsedDirectives(stencils, declare.lnumber, create.lnumber)

    def _extract_directive(
        self,
        directives: list[TypedDirective],
        required_type: tuple[DirectiveType] | DirectiveType,
    ):
        return [d for d in directives if isinstance(d.directive_type, required_type)]


class DirectiveSyntaxValidator:
    """Syntax validation method dispatcher for each directive type."""

    # todo: refactor functions

    def __init__(self):
        self.parentheses_regex = r"\(.*?\)"

    def StartStencil(self, directive: TypedDirective):
        # todo: implement check and exception raise
        pattern = directive.directive_type.pattern
        regex = rf"{pattern}\((\w*?)\)"
        matches = re.fullmatch(regex, directive.string)
        if not matches:
            raise DirectiveSyntaxError(
                f"""DirectiveSyntaxError at {directive.lnumber}\n
                                            {directive.string} is invalid, expected {regex}\n
                                        """
            )

    def EndStencil(self, directive: TypedDirective):
        # todo: implement check and exception raise
        pattern = directive.directive_type.pattern
        regex = rf"{pattern}\((\w*?)\)"
        matches = re.fullmatch(regex, directive.string)
        if not matches:
            raise DirectiveSyntaxError(
                f"""DirectiveSyntaxError on line {directive.lnumber}\n
                                            {directive.string} is invalid, expected {regex}\n
                                        """
            )

    def Create(self, directive: TypedDirective):
        # todo: implement check and exception raise
        pattern = directive.directive_type.pattern
        matches = re.fullmatch(rf"{pattern}", directive.string)
        if not matches:
            raise Exception

    def Declare(self, directive: TypedDirective):
        # todo: implement check and exception raise
        pattern = directive.directive_type.pattern
        matches = re.fullmatch(rf"{pattern}", directive.string)
        if not matches:
            raise Exception


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
