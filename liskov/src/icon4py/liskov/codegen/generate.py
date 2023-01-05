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
from typing import Sequence, Type

import eve
from eve.codegen import TemplatedGenerator

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
from icon4py.liskov.codegen.interface import CodeGenInput, SerialisedDirectives
from icon4py.liskov.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class GeneratedCode:
    """A class for storing generated f90 code and its line number information."""

    source: str
    startln: int
    endln: int


class IntegrationGenerator:
    def __init__(self, directives: SerialisedDirectives, profile: bool):
        self.directives = directives
        self.generated: list[GeneratedCode] = []
        self._generate(profile)

    def _generate(self, profile: bool) -> None:
        """Generate all f90 code for integration.

        Args:
            profile: A boolean indicating whether to include profiling calls in the generated code.
        """
        # self._generate_create()
        self._generate_imports()
        self._generate_declare()
        self._generate_stencil(profile)

    def _add_generated_code(
        self,
        parent_node: Type[eve.Node],
        code_generator: Type[TemplatedGenerator],
        startln: int,
        endln: int,
        **kwargs: CodeGenInput | Sequence[CodeGenInput] | bool,
    ) -> None:
        """Add a GeneratedCode object to the `generated` attribute with the given source code and line number information.

        Args:
            parent_node: The parent node of the code to be generated.
            code_generator: The code generator to use for generating the code.
            startln: The start line number of the generated code.
            endln: The end line number of the generated code.
            **kwargs: Additional keyword arguments to be passed to the code generator.
        """
        source = generate_fortran_code(parent_node, code_generator, **kwargs)
        code = GeneratedCode(source=source, startln=startln, endln=endln)
        self.generated.append(code)

    def _generate_declare(self) -> None:
        """Generate f90 code for declaration statements."""
        logger.info("Generating DECLARE statement.")
        self._add_generated_code(
            DeclareStatement,
            DeclareStatementGenerator,
            self.directives.declare.startln,
            self.directives.declare.endln,
            declare_data=self.directives.declare,
        )

    def _generate_stencil(self, profile: bool) -> None:
        """Generate f90 integration code surrounding a stencil.

        Args:
            profile: A boolean indicating whether to include profiling calls in the generated code.
        """
        for i, stencil in enumerate(self.directives.start):
            logger.info(f"Generating START statement for {stencil.name}")
            self._add_generated_code(
                OutputFieldCopy,
                OutputFieldCopyGenerator,
                self.directives.start[i].startln,
                self.directives.start[i].endln,
                stencil_data=stencil,
                profile=profile,
            )
            logger.info(f"Generating END statement for {stencil.name}")
            self._add_generated_code(
                WrapRunFunc,
                WrapRunFuncGenerator,
                self.directives.end[i].startln,
                self.directives.end[i].endln,
                stencil_data=stencil,
                profile=profile,
            )

    def _generate_imports(self) -> None:
        """Generate f90 code for import statements."""
        logger.info("Generating IMPORT statement.")
        self._add_generated_code(
            ImportsStatement,
            ImportsStatementGenerator,
            self.directives.imports.startln,
            self.directives.imports.endln,
            stencils=self.directives.start,
        )

    def _generate_create(self) -> None:
        """Generate f90 code for OpenACC DATA CREATE statements."""
        logger.info("Generating DATA CREATE statement.")
        self._add_generated_code(
            CreateStatement,
            CreateStatementGenerator,
            self.directives.create.startln,
            self.directives.create.endln,
            stencils=self.directives.start,
        )
