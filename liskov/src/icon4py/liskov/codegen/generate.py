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
from typing import Optional, Sequence, Type

import eve
from eve.codegen import TemplatedGenerator
from typing_extensions import Any

from icon4py.liskov.codegen.f90 import (
    DeclareStatement,
    DeclareStatementGenerator,
    EndCreateStatement,
    EndCreateStatementGenerator,
    EndIfStatement,
    EndIfStatementGenerator,
    EndStencilStatement,
    EndStencilStatementGenerator,
    ImportsStatement,
    ImportsStatementGenerator,
    StartCreateStatement,
    StartCreateStatementGenerator,
    StartStencilStatement,
    StartStencilStatementGenerator,
    generate_fortran_code,
)
from icon4py.liskov.codegen.interface import (
    CodeGenInput,
    DeserialisedDirectives,
    UnusedDirective,
)
from icon4py.liskov.common import Step
from icon4py.liskov.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class GeneratedCode:
    """A class for storing generated f90 code and its line number information."""

    source: str
    startln: int
    endln: int


class IntegrationGenerator(Step):
    def __init__(self, directives: DeserialisedDirectives, profile: bool):
        self.profile = profile
        self.directives = directives
        self.generated: list[GeneratedCode] = []

    def __call__(self, data: Any = None) -> list[GeneratedCode]:
        """Generate all f90 code for integration.

        Args:
            profile: A boolean indicating whether to include profiling calls in the generated code.
        """
        self._generate_create()
        self._generate_imports()
        self._generate_declare()
        self._generate_stencil()
        self._generate_endif()
        return self.generated

    def _generate(
        self,
        parent_node: Type[eve.Node],
        code_generator: Type[TemplatedGenerator],
        startln: int,
        endln: int,
        **kwargs: CodeGenInput | Sequence[CodeGenInput] | Optional[bool],
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
        self._generate(
            DeclareStatement,
            DeclareStatementGenerator,
            self.directives.Declare.startln,
            self.directives.Declare.endln,
            declare_data=self.directives.Declare,
        )

    def _generate_stencil(self) -> None:
        """Generate f90 integration code surrounding a stencil.

        Args:
            profile: A boolean indicating whether to include profiling calls in the generated code.
        """
        for i, stencil in enumerate(self.directives.StartStencil):
            logger.info(f"Generating START statement for {stencil.name}")
            self._generate(
                StartStencilStatement,
                StartStencilStatementGenerator,
                self.directives.StartStencil[i].startln,
                self.directives.StartStencil[i].endln,
                stencil_data=stencil,
                profile=self.profile,
            )
            logger.info(f"Generating END statement for {stencil.name}")
            self._generate(
                EndStencilStatement,
                EndStencilStatementGenerator,
                self.directives.EndStencil[i].startln,
                self.directives.EndStencil[i].endln,
                stencil_data=stencil,
                profile=self.profile,
                noendif=self.directives.EndStencil[i].noendif,
            )

    def _generate_imports(self) -> None:
        """Generate f90 code for import statements."""
        logger.info("Generating IMPORT statement.")
        self._generate(
            ImportsStatement,
            ImportsStatementGenerator,
            self.directives.Imports.startln,
            self.directives.Imports.endln,
            stencils=self.directives.StartStencil,
        )

    def _generate_create(self) -> None:
        """Generate f90 code for OpenACC DATA CREATE statements."""
        logger.info("Generating DATA CREATE statement.")
        self._generate(
            StartCreateStatement,
            StartCreateStatementGenerator,
            self.directives.StartCreate.startln,
            self.directives.StartCreate.endln,
            stencils=self.directives.StartStencil,
        )

        self._generate(
            EndCreateStatement,
            EndCreateStatementGenerator,
            self.directives.EndCreate.startln,
            self.directives.EndCreate.endln,
        )

    def _generate_endif(self) -> None:
        """Generate f90 code for endif statements."""
        if self.directives.EndIf != UnusedDirective:
            for endif in self.directives.EndIf:
                logger.info("Generating ENDIF statement.")
                self._generate(
                    EndIfStatement,
                    EndIfStatementGenerator,
                    endif.startln,
                    endif.endln,
                )
