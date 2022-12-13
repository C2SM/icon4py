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

from icon4py.liskov.codegen.f90 import (
    CreateStatement,
    CreateStatementGenerator,
    DeclareStatement,
    DeclareStatementGenerator,
    ImportsStatement,
    ImportsStatementGenerator,
    OutputFieldCopy,
    OutputFieldCopyGenerator,
    WrapRunFunc,
    WrapRunFuncGenerator,
    generate_fortran_code,
)
from icon4py.liskov.codegen.interface import SerialisedDirectives


@dataclass
class GeneratedCode:
    source: str
    startln: int
    endln: int


class IntegrationGenerator:
    def __init__(self, directives: SerialisedDirectives, profile: bool):
        self.generated: list[GeneratedCode] = []
        self.profile = profile
        self.directives = directives
        self._generate_code()

    def _generate_declare(self) -> None:
        declare_source = generate_fortran_code(
            DeclareStatement,
            DeclareStatementGenerator,
            declare_data=self.directives.declare,
        )
        declare_code = GeneratedCode(
            source=declare_source,
            startln=self.directives.declare.startln,
            endln=self.directives.declare.endln,
        )
        self.generated.append(declare_code)

    def _generate_stencil(self) -> None:
        for i, stencil in enumerate(self.directives.start):
            # generate output field copies
            output_field_copy_source = generate_fortran_code(
                OutputFieldCopy,
                OutputFieldCopyGenerator,
                stencil_data=stencil,
                profile=self.profile,
            )
            output_field_copy_code = GeneratedCode(
                source=output_field_copy_source,
                startln=self.directives.start[i].startln,
                endln=self.directives.start[i].endln,
            )
            self.generated.append(output_field_copy_code)

            # generate wrap run call
            wrap_run_source = generate_fortran_code(
                WrapRunFunc,
                WrapRunFuncGenerator,
                stencil_data=stencil,
                profile=self.profile,
            )
            wrap_run_code = GeneratedCode(
                source=wrap_run_source,
                startln=self.directives.end[i].startln,
                endln=self.directives.end[i].endln,
            )
            self.generated.append(wrap_run_code)

    def _generate_imports(self) -> None:
        imports_source = generate_fortran_code(
            ImportsStatement,
            ImportsStatementGenerator,
            stencils=self.directives.start,
        )
        imports_code = GeneratedCode(
            source=imports_source,
            startln=self.directives.imports.startln,
            endln=self.directives.imports.endln,
        )
        self.generated.append(imports_code)

    def _generate_create(self) -> None:
        create_source = generate_fortran_code(
            CreateStatement, CreateStatementGenerator, stencils=self.directives.start
        )
        create_code = GeneratedCode(
            source=create_source,
            startln=self.directives.create.startln,
            endln=self.directives.create.endln,
        )
        self.generated.append(create_code)

    def _generate_code(self) -> None:
        """Generate all f90 code snippets for integration."""
        self._generate_create()
        self._generate_imports()
        self._generate_declare()
        self._generate_stencil()
