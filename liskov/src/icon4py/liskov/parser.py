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


class DirectivesParser:
    _SUPPORTED_DIRECTIVES = [StartStencil, EndStencil, Create, Declare]

    def __init__(self, directives: list[RawDirective]) -> None:
        """Class which carries out end-to-end parsing of a file with regards to DSL directives.

            Handles parsing and validation of preprocessor directives, returning a ParsedDirectives
            object which can be used for code generation.

        Args:
            directives: A list of directives collected by the DirectivesCollector.

        Note:
            Directives which are supported by the parser can be modified by editing the self._SUPPORTED_DIRECTIVES class
            attribute.
        """
        self.directives = directives
        self.exception_handler = ParsingExceptionHandler
        self.parsed_directives = self._parse_directives()

    def _parse_directives(self) -> ParsedDirectives | NoDirectivesFound:
        """Execute end-to-end parsing of collected directives."""
        if len(self.directives) != 0:
            typed = self._determine_type(self.directives)

            # run validation passes
            preprocessed = self._preprocess(typed)
            DirectiveSyntaxValidator().validate(preprocessed)
            DirectiveSemanticsValidator().validate(preprocessed)

            stencils = self.extract_stencils(preprocessed)
            return self._build_parsed_directives(stencils, preprocessed)
        return NoDirectivesFound()

    def _preprocess(self, directives: list[TypedDirective]) -> list[TypedDirective]:
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

    def _determine_type(self, directives: list[RawDirective]) -> list[TypedDirective]:
        """Determine type of directive used and whether it is supported."""
        typed = []
        for d in directives:
            for directive_type in self._SUPPORTED_DIRECTIVES:
                if directive_type.pattern in d.string:
                    typed.append(
                        TypedDirective(d.string, d.startln, d.endln, directive_type())
                    )

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
