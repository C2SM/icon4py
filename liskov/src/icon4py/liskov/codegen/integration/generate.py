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

from typing import Any

from icon4py.common.logger import setup_logger
from icon4py.liskov.codegen.integration.interface import (
    IntegrationCodeInterface,
    StartStencilData,
    UnusedDirective,
)
from icon4py.liskov.codegen.integration.template import (
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
)
from icon4py.liskov.codegen.shared.generator import CodeGenerator
from icon4py.liskov.codegen.shared.types import GeneratedCode
from icon4py.liskov.external.metadata import CodeMetadata


logger = setup_logger(__name__)


class IntegrationCodeGenerator(CodeGenerator):
    def __init__(
        self,
        interface: IntegrationCodeInterface,
        profile: bool = False,
        metadatagen: bool = False,
    ):
        super().__init__()
        self.profile = profile
        self.interface = interface
        self.metadatagen = metadatagen

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

    def _generate_metadata(self) -> None:
        """Generate metadata about the current liskov execution."""
        if self.metadatagen:
            logger.info("Generating icon-liskov metadata.")
            self._generate(
                MetadataStatement,
                MetadataStatementGenerator,
                startln=0,
                metadata=CodeMetadata(),
            )

    def _generate_declare(self) -> None:
        """Generate f90 code for declaration statements."""
        for i, declare in enumerate(self.interface.Declare):
            logger.info("Generating DECLARE statement.")
            self._generate(
                DeclareStatement,
                DeclareStatementGenerator,
                self.interface.Declare[i].startln,
                declare_data=declare,
            )

    def _generate_start_stencil(self) -> None:
        """Generate f90 integration code surrounding a stencil.

        Args:
            profile: A boolean indicating whether to include profiling calls in the generated code.
        """
        i = 0

        while i < len(self.interface.StartStencil):
            stencil = self.interface.StartStencil[i]
            logger.info(f"Generating START statement for {stencil.name}")

            try:
                next_stencil = self.interface.StartStencil[i + 1]
            except IndexError:
                pass

            if stencil.mergecopy and next_stencil.mergecopy:
                stencil = StartStencilData(
                    startln=stencil.startln,
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
                    stencil_data=stencil,
                    profile=self.profile,
                )
            else:
                self._generate(
                    StartStencilStatement,
                    StartStencilStatementGenerator,
                    self.interface.StartStencil[i].startln,
                    stencil_data=stencil,
                    profile=self.profile,
                )
                i += 1

    def _generate_end_stencil(self) -> None:
        """Generate f90 integration code surrounding a stencil.

        Args:
            profile: A boolean indicating whether to include profiling calls in the generated code.
        """
        for i, stencil in enumerate(self.interface.StartStencil):
            logger.info(f"Generating END statement for {stencil.name}")
            self._generate(
                EndStencilStatement,
                EndStencilStatementGenerator,
                self.interface.EndStencil[i].startln,
                stencil_data=stencil,
                profile=self.profile,
                noendif=self.interface.EndStencil[i].noendif,
                noprofile=self.interface.EndStencil[i].noprofile,
            )

    def _generate_imports(self) -> None:
        """Generate f90 code for import statements."""
        logger.info("Generating IMPORT statement.")
        self._generate(
            ImportsStatement,
            ImportsStatementGenerator,
            self.interface.Imports.startln,
            stencils=self.interface.StartStencil,
        )

    def _generate_create(self) -> None:
        """Generate f90 code for OpenACC DATA CREATE statements."""
        logger.info("Generating DATA CREATE statement.")
        self._generate(
            StartCreateStatement,
            StartCreateStatementGenerator,
            self.interface.StartCreate.startln,
            stencils=self.interface.StartStencil,
            extra_fields=self.interface.StartCreate.extra_fields,
        )

        self._generate(
            EndCreateStatement,
            EndCreateStatementGenerator,
            self.interface.EndCreate.startln,
        )

    def _generate_endif(self) -> None:
        """Generate f90 code for endif statements."""
        if self.interface.EndIf != UnusedDirective:
            for endif in self.interface.EndIf:  # type: ignore
                logger.info("Generating ENDIF statement.")
                self._generate(EndIfStatement, EndIfStatementGenerator, endif.startln)

    def _generate_profile(self) -> None:
        """Generate additional nvtx profiling statements."""
        if self.profile:
            if self.interface.StartProfile != UnusedDirective:
                for start in self.interface.StartProfile:  # type: ignore
                    logger.info("Generating nvtx start statement.")
                    self._generate(
                        StartProfileStatement,
                        StartProfileStatementGenerator,
                        start.startln,
                        name=start.name,
                    )

            if self.interface.EndProfile != UnusedDirective:
                for end in self.interface.EndProfile:  # type: ignore
                    logger.info("Generating nvtx end statement.")
                    self._generate(
                        EndProfileStatement, EndProfileStatementGenerator, end.startln
                    )

    def _generate_insert(self) -> None:
        """Generate free form statement from insert directive."""
        if self.interface.Insert != UnusedDirective:
            for insert in self.interface.Insert:  # type: ignore
                logger.info("Generating free form statement.")
                self._generate(
                    InsertStatement,
                    InsertStatementGenerator,
                    insert.startln,
                    content=insert.content,
                )
