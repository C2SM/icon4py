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
import copy
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
from icon4py.liskov.input import (
    BoundsData,
    CreateData,
    DeclareData,
    FieldAssociationData,
    StencilData,
)
from icon4py.liskov.validation import (
    DirectiveSemanticsValidator,
    DirectiveSyntaxValidator,
)


@dataclass(frozen=True)
class ParsedDirectives:
    stencils: list[StencilData]
    declare: DeclareData
    create: CreateData


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
            # run type deduction
            typed = self._determine_type(self.directives)

            # run validation passes
            preprocessed = self._preprocess(typed)
            DirectiveSyntaxValidator().validate(preprocessed)
            DirectiveSemanticsValidator().validate(preprocessed)

            # extract directive data
            stencils = self._extract_stencils(preprocessed)
            declare = self._extract_declare(preprocessed)
            create = self._extract_create(preprocessed)

            return ParsedDirectives(stencils, declare, create)
        return NoDirectivesFound()

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

    def _extract_stencils(self, directives: list[TypedDirective]) -> list[StencilData]:
        """Extract all stencils from typed and validated directives."""
        stencils = []
        stencil_directives = self._extract_directive(directives, StartStencil)
        for d in stencil_directives:

            named_args = self._parse_directive_string(d)

            # extract fields
            field_args = copy.copy(named_args)
            entries_to_remove = (
                "name",
                "horizontal_lower",
                "horizontal_upper",
                "vertical_lower",
                "vertical_upper",
            )
            list(map(field_args.pop, entries_to_remove))

            fields = [
                FieldAssociationData(
                    variable_name=varname, variable_association=association
                )
                for varname, association in field_args.items()
            ]
            bounds = BoundsData(
                hlower=named_args["horizontal_lower"],
                hupper=named_args["horizontal_upper"],
                vlower=named_args["vertical_lower"],
                vupper=named_args["vertical_upper"],
            )
            try:
                stencils.append(
                    StencilData(
                        name=named_args["name"],
                        fields=fields,
                        bounds=bounds,
                        startln=d.startln,
                        endln=d.endln,
                    )
                )
            except Exception as e:
                raise e
        return stencils

    @staticmethod
    def _parse_directive_string(d: TypedDirective):
        # clean string (remove outer parentheses and pattern)
        string = d.string.replace(f"{d.directive_type.pattern}", "")
        args = string[1:-1].split(";")
        named_args = {a.split("=")[0].strip(): a.split("=")[1] for a in args}
        return named_args

    def _extract_declare(self, directives: list[TypedDirective]) -> DeclareData:
        declare = self._extract_directive(directives, Declare)[0]
        declarations = self._parse_directive_string(declare)
        return DeclareData(declare.startln, declare.endln, declarations)

    def _extract_create(self, directives: list[TypedDirective]) -> DeclareData:
        create = self._extract_directive(directives, Create)[0]
        string = create.string.replace(f"{create.directive_type.pattern}", "")
        args = string[1:-1].split(";")
        variables = [s.strip() for s in args]
        return CreateData(create.startln, create.endln, variables)

    @staticmethod
    def _extract_directive(
        directives: list[TypedDirective],
        required_type: tuple[Directive] | Directive,
    ) -> list[TypedDirective] | TypedDirective:
        directives = [
            d for d in directives if isinstance(d.directive_type, required_type)
        ]
        return directives
