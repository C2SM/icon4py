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
from dataclasses import dataclass
from pathlib import Path

from icon4py.liskov.directives import (
    IDENTIFIER,
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
from icon4py.liskov.input import CreateData, DeclareData, StencilData
from icon4py.liskov.validation import (
    DirectiveSemanticsValidator,
    DirectiveSyntaxValidator,
)


@dataclass(frozen=True)
class ParsedDirectives:
    stencil_directive: list[StencilData]
    declare_directive: DeclareData
    create_directive: CreateData


class DirectivesCollector:
    def __init__(self, filepath: Path) -> None:
        """Class which collects all DSL directives in a given file.

        Args:
            filepath: Path to file to scan for directives.
        """
        self.filepath = filepath
        self.directives = self._collect_directives()

    def _process_collected(self, collected):
        directive_string = "\n".join([c[0] for c in collected])
        abs_startln = collected[0][-1] + 1
        abs_endln = collected[-1][-1] + 1
        return RawDirective(directive_string, startln=abs_startln, endln=abs_endln)

    def _collect_directives(self) -> list[RawDirective]:
        """Scan filepath for directives and returns them along with their line numbers."""
        directives = []
        with self.filepath.open() as f:

            collected = []
            for lnumber, string in enumerate(f):

                if IDENTIFIER in string:
                    stripped = string.strip()
                    collected.append([stripped, lnumber])

                    match stripped[-1]:
                        case ")":
                            directives.append(self._process_collected(collected))
                            collected = []
                        case "&":
                            continue
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
        self.exception_handler = ParsingExceptionHandler
        self.parsed_directives = self._parse_directives()

    def _parse_directives(self) -> ParsedDirectives | NoDirectivesFound:
        """Execute end-to-end parsing of collected directives."""
        if len(self.collector.directives) != 0:
            typed = self._determine_type(self._preprocess(self.collector.directives))
            DirectiveSyntaxValidator().validate(typed)
            DirectiveSemanticsValidator().validate(typed)
            stencils = self.extract_stencils(typed)
            return self._build_parsed_directives(stencils, typed)
        return NoDirectivesFound()

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

    def extract_stencils(self, directives: list[TypedDirective]) -> list[StencilData]:
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
                StencilData(
                    name=stencil_name,
                    startln=start.lnumber,
                    endln=end.lnumber,
                    filename=self.filepath,
                )
            )
        return stencils

    def _build_parsed_directives(
        self, stencils: list[StencilData], directives: list[TypedDirective]
    ) -> ParsedDirectives:
        """Build ParsedDirectives object."""
        declare = self._extract_directive(directives, Declare)
        create = self._extract_directive(directives, Create)
        return ParsedDirectives(stencils, declare.lnumber, create.lnumber)

    @staticmethod
    def _extract_directive(
        directives: list[TypedDirective],
        required_type: tuple[Directive] | Directive,
    ) -> list[TypedDirective] | TypedDirective:
        directives = [
            d for d in directives if isinstance(d.directive_type, required_type)
        ]
        if len(directives) == 1:
            return directives[0]
        return directives
