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
from icon4py.liskov.serialise import SerialisedDirectives


@dataclass(frozen=True)
class GeneratedCode:
    gen: str
    target_ln: int


class IntegrationGenerator:
    def __init__(self, directives: SerialisedDirectives, profile: bool):
        self.generated = []
        self.profile = profile
        self.directives = directives
        self._generate_code()

    def _generate_declare(self):
        declare_source = generate_fortran_code(
            DeclareStatement,
            DeclareStatementGenerator,
            declare_data=self.directives.declare,
        )
        declare_code = GeneratedCode(
            gen=declare_source, target_ln=self.directives.declare.startln
        )
        self.generated.append(declare_code)

    def _generate_stencil(self):
        for i, stencil in enumerate(self.directives.start):
            # generate output field copies
            output_field_copy_source = generate_fortran_code(
                OutputFieldCopy,
                OutputFieldCopyGenerator,
                stencil_data=stencil,
                profile=self.profile,
            )
            output_field_copy_code = GeneratedCode(
                gen=output_field_copy_source, target_ln=self.directives.start[i].startln
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
                gen=wrap_run_source, target_ln=self.directives.end[i].startln
            )
            self.generated.append(wrap_run_code)

    def _generate_imports(self):
        names = [stencil.name for stencil in self.directives.start]
        imports_source = generate_fortran_code(
            ImportsStatement,
            ImportsStatementGenerator,
            names=names,
        )
        imports_code = GeneratedCode(
            gen=imports_source, target_ln=self.directives.imports.startln
        )
        self.generated.append(imports_code)

    def _generate_code(self):
        """Generate all f90 code snippets for integration."""
        self._generate_imports()
        self._generate_declare()
        self._generate_stencil()
