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

import gt4py.eve as eve
from gt4py.eve.codegen import TemplatedGenerator
from typing_extensions import Any

from icon4py.liskov.codegen.f90 import (
    DeclareStatement,
    DeclareStatementGenerator,
    EndCreateStatement,
    EndCreateStatementGenerator,
    EndIfStatement,
    EndIfStatementGenerator,
    EndProfileStatement,
    EndProfileStatementGenerator,
    EndStencilStatement,
    EndStencilStatementGenerator,
    ImportsStatement,
    ImportsStatementGenerator,
    InsertStatement,
    InsertStatementGenerator,
    MetadataStatement,
    MetadataStatementGenerator,
    StartCreateStatement,
    StartCreateStatementGenerator,
    StartProfileStatement,
    StartProfileStatementGenerator,
    StartStencilStatement,
    StartStencilStatementGenerator,
    generate_fortran_code,
)
from icon4py.liskov.codegen.interface import (
    CodeGenInput,
    DeserialisedDirectives,
    StartStencilData,
    UnusedDirective,
)
from icon4py.liskov.common import Step
from icon4py.liskov.external.metadata import CodeMetadata
from icon4py.liskov.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class GeneratedCode:
    """A class for storing generated f90 code and its line number information."""

    source: str
    startln: int
    endln: int


class IntegrationGenerator(Step):
    def __init__(
        self,
        directives: DeserialisedDirectives,
        profile: bool,
        metadata_gen: bool = False,
    ):
        self.profile = profile
        self.directives = directives
        self.generated: list[GeneratedCode] = []
        self.metadata_gen = metadata_gen

    def __call__(self, data: Any = None) -> list[GeneratedCode]:
        """Generate all f90 code for integration.

        Args:
            profile: A boolean indicating whether to include profiling calls in the generated code.
        """
        self._generate_metadata()
        self._generate_create()
        self._generate_imports()
        self._generate_declare()
        self._generate_start_stencil()
        self._generate_end_stencil()
        self._generate_endif()
        self._generate_profile()
        self._generate_insert()
        return self.generated

    def _generate(
        self,
        parent_node: Type[eve.Node],
        code_generator: Type[TemplatedGenerator],
        startln: int,
        endln: int,
        **kwargs: CodeGenInput | Sequence[CodeGenInput] | Optional[bool] | Any,
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

    def _generate_metadata(self) -> None:
        """Generate metadata about the current liskov execution."""
        if self.metadata_gen:
            logger.info("Generating icon-liskov metadata.")
            self._generate(
                MetadataStatement,
                MetadataStatementGenerator,
                0,
                0,
                metadata=CodeMetadata(),
            )

    def _generate_declare(self) -> None:
        """Generate f90 code for declaration statements."""
        for i, declare in enumerate(self.directives.Declare):
            logger.info("Generating DECLARE statement.")
            self._generate(
                DeclareStatement,
                DeclareStatementGenerator,
                self.directives.Declare[i].startln,
                self.directives.Declare[i].endln,
                declare_data=declare,
            )

    def _generate_start_stencil(self) -> None:
        """Generate f90 integration code surrounding a stencil.

        Args:
            profile: A boolean indicating whether to include profiling calls in the generated code.
        """
        i = 0

        while i < len(self.directives.StartStencil):
            stencil = self.directives.StartStencil[i]
            logger.info(f"Generating START statement for {stencil.name}")

            try:
                next_stencil = self.directives.StartStencil[i + 1]
            except IndexError:
                pass

            if stencil.mergecopy and next_stencil.mergecopy:
                stencil = StartStencilData(
                    startln=stencil.startln,
                    endln=next_stencil.endln,
                    name=stencil.name + "_" + next_stencil.name,
                    fields=stencil.fields + next_stencil.fields,
                    bounds=stencil.bounds,
                    acc_present=stencil.acc_present,
                    mergecopy=stencil.mergecopy,
                    copies=stencil.copies,
                )
                i += 2

                self._generate(
                    StartStencilStatement,
                    StartStencilStatementGenerator,
                    stencil.startln,
                    next_stencil.endln,
                    stencil_data=stencil,
                    profile=self.profile,
                )
            else:
                self._generate(
                    StartStencilStatement,
                    StartStencilStatementGenerator,
                    self.directives.StartStencil[i].startln,
                    self.directives.StartStencil[i].endln,
                    stencil_data=stencil,
                    profile=self.profile,
                )
                i += 1

    def _generate_end_stencil(self) -> None:
        """Generate f90 integration code surrounding a stencil.

        Args:
            profile: A boolean indicating whether to include profiling calls in the generated code.
        """
        for i, stencil in enumerate(self.directives.StartStencil):
            logger.info(f"Generating END statement for {stencil.name}")
            self._generate(
                EndStencilStatement,
                EndStencilStatementGenerator,
                self.directives.EndStencil[i].startln,
                self.directives.EndStencil[i].endln,
                stencil_data=stencil,
                profile=self.profile,
                noendif=self.directives.EndStencil[i].noendif,
                noprofile=self.directives.EndStencil[i].noprofile,
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
            extra_fields=self.directives.StartCreate.extra_fields,
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
            for endif in self.directives.EndIf:  # type: ignore
                logger.info("Generating ENDIF statement.")
                self._generate(
                    EndIfStatement,
                    EndIfStatementGenerator,
                    endif.startln,
                    endif.endln,
                )

    def _generate_profile(self) -> None:
        """Generate additional nvtx profiling statements."""
        if self.profile:
            if self.directives.StartProfile != UnusedDirective:
                for start in self.directives.StartProfile:  # type: ignore
                    logger.info("Generating nvtx start statement.")
                    self._generate(
                        StartProfileStatement,
                        StartProfileStatementGenerator,
                        start.startln,
                        start.endln,
                        name=start.name,
                    )

            if self.directives.EndProfile != UnusedDirective:
                for end in self.directives.EndProfile:  # type: ignore
                    logger.info("Generating nvtx end statement.")
                    self._generate(
                        EndProfileStatement,
                        EndProfileStatementGenerator,
                        end.startln,
                        end.endln,
                    )

    def _generate_insert(self) -> None:
        """Generate free form statement from insert directive."""
        if self.directives.Insert != UnusedDirective:
            for insert in self.directives.Insert:  # type: ignore
                logger.info("Generating free form statement.")
                self._generate(
                    InsertStatement,
                    InsertStatementGenerator,
                    insert.startln,
                    insert.endln,
                    content=insert.content,
                )
