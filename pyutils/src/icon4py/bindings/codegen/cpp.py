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

from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import Node, TemplatedGenerator, format_source

from icon4py.bindings.types import Field
from icon4py.bindings.utils import write_string


@dataclass
class CppImpl:
    stencil_name: str
    fields: list[Field]

    def write(self, outpath: Path):
        header = self._generate_header()
        source = format_source("cpp", CppImplGenerator.apply(header), style="LLVM")
        write_string(source, outpath, f"{self.stencil_name}.cpp")

    def _generate_header(self):
        pass


class IncludeStatements(Node):
    pass


class UsingDeclarations(Node):
    pass


class UtilityFunctions(Node):
    pass


class StencilClass(Node):
    pass


class RunFunc(Node):
    pass


class VerifyFunc(Node):
    pass


class RunAndVerifyFunc(Node):
    pass


class SetupFunc(Node):
    pass


class FreeFunc(Node):
    pass


class CppFile(Node):
    includeStatements: IncludeStatements
    usingDeclarations: UsingDeclarations
    utilityFunctions: UtilityFunctions
    stencilClass: StencilClass
    runFunc: RunFunc
    verifyFunc: VerifyFunc
    runAndVerifyFunc: RunAndVerifyFunc
    setupFunc: SetupFunc
    freeFunc: FreeFunc


class CppImplGenerator(TemplatedGenerator):
    CppFile = as_jinja(
        """\
        {{ includeStatements }}
        {{ usingDeclarations }}
        {{ utilityFunctions }}
        {{ stencilClass }}
        extern "C" {
        {{ runFunc }}
        {{ verifyFunc }}
        {{ runAndVerifyFunc }}
        {{ setupFunc }}
        {{ freeFunc }}
        }
        """
    )
