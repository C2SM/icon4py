# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Sequence, TypeGuard, Union

from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.codegen.integration.interface import (
    IntegrationCodeInterface,
    StartStencilData,
    UnusedDirective,
)
from icon4pytools.liskov.codegen.integration.template import (
    DeclareStatement,
    DeclareStatementGenerator,
    EndCreateStatement,
    EndCreateStatementGenerator,
    EndDeleteStatement,
    EndDeleteStatementGenerator,
    EndFusedStencilStatement,
    EndFusedStencilStatementGenerator,
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
    StartDeleteStatement,
    StartDeleteStatementGenerator,
    StartFusedStencilStatement,
    StartFusedStencilStatementGenerator,
    StartProfileStatement,
    StartProfileStatementGenerator,
    StartStencilStatement,
    StartStencilStatementGenerator,
)
from icon4pytools.liskov.codegen.shared.generate import CodeGenerator
from icon4pytools.liskov.codegen.shared.types import GeneratedCode
from icon4pytools.liskov.external.metadata import CodeMetadata


logger = setup_logger(__name__)


def _is_sequence(value: Union[Sequence[Any], UnusedDirective]) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence)


class IntegrationCodeGenerator(CodeGenerator):
    def __init__(
        self,
        interface: IntegrationCodeInterface,
        profile: bool = False,
        verification: bool = False,
        metadatagen: bool = False,
    ):
        super().__init__()
        self.profile = profile
        self.interface = interface
        self.verification = verification
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
        self._generate_start_fused_stencil()
        self._generate_end_fused_stencil()
        self._generate_delete()
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
                verification=self.verification,
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
                    optional_module=stencil.optional_module,
                )
                i += 2

                self._generate(
                    StartStencilStatement,
                    StartStencilStatementGenerator,
                    stencil.startln,
                    stencil_data=stencil,
                    profile=self.profile,
                    verification=self.verification,
                )
            else:
                self._generate(
                    StartStencilStatement,
                    StartStencilStatementGenerator,
                    self.interface.StartStencil[i].startln,
                    stencil_data=stencil,
                    profile=self.profile,
                    verification=self.verification,
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
                verification=self.verification,
                noendif=self.interface.EndStencil[i].noendif,
                noprofile=self.interface.EndStencil[i].noprofile,
                noaccenddata=self.interface.EndStencil[i].noaccenddata,
            )

    def _generate_start_fused_stencil(self) -> None:
        """Generate f90 integration code surrounding a fused stencil."""
        if self.interface.StartFusedStencil != UnusedDirective:
            for stencil in self.interface.StartFusedStencil:
                logger.info(f"Generating START FUSED statement for {stencil.name}")
                self._generate(
                    StartFusedStencilStatement,
                    StartFusedStencilStatementGenerator,
                    stencil.startln,
                    stencil_data=stencil,
                    verification=self.verification,
                )

    def _generate_end_fused_stencil(self) -> None:
        """Generate f90 integration code surrounding a fused stencil."""
        if self.interface.EndFusedStencil != UnusedDirective:
            for i, stencil in enumerate(self.interface.StartFusedStencil):
                logger.info(f"Generating END Fused statement for {stencil.name}")
                self._generate(
                    EndFusedStencilStatement,
                    EndFusedStencilStatementGenerator,
                    self.interface.EndFusedStencil[i].startln,
                    stencil_data=stencil,
                    verification=self.verification,
                )

    def _generate_delete(self) -> None:
        """Generate f90 integration code for delete section."""
        if isinstance(self.interface.StartDelete, Sequence) and isinstance(
            self.interface.EndDelete, Sequence
        ):
            logger.info("Generating DELETE statement.")
            for start, end in zip(
                self.interface.StartDelete, self.interface.EndDelete, strict=True
            ):
                self._generate(
                    StartDeleteStatement,
                    StartDeleteStatementGenerator,
                    start.startln,
                )
                self._generate(
                    EndDeleteStatement,
                    EndDeleteStatementGenerator,
                    end.startln,
                )

    def _generate_imports(self) -> None:
        """Generate f90 code for import statements."""
        logger.info("Generating IMPORT statement.")
        self._generate(
            ImportsStatement,
            ImportsStatementGenerator,
            self.interface.Imports.startln,
            stencils=[*self.interface.StartStencil, *self.interface.StartFusedStencil],
            verification=self.verification,
        )

    def _generate_create(self) -> None:
        """Generate f90 code for OpenACC DATA CREATE statements."""
        if _is_sequence(self.interface.StartCreate):
            for startcreate in self.interface.StartCreate:
                logger.info("Generating DATA CREATE statement.")
                self._generate(
                    StartCreateStatement,
                    StartCreateStatementGenerator,
                    startcreate.startln,
                    extra_fields=startcreate.extra_fields,
                )

        if _is_sequence(self.interface.EndCreate):
            for endcreate in self.interface.EndCreate:
                self._generate(
                    EndCreateStatement,
                    EndCreateStatementGenerator,
                    endcreate.startln,
                )

    def _generate_endif(self) -> None:
        """Generate f90 code for endif statements."""
        if _is_sequence(self.interface.EndIf):
            for endif in self.interface.EndIf:
                logger.info("Generating ENDIF statement.")
                self._generate(EndIfStatement, EndIfStatementGenerator, endif.startln)

    def _generate_profile(self) -> None:
        """Generate additional nvtx profiling statements."""
        if self.profile:
            if _is_sequence(self.interface.StartProfile):
                for start in self.interface.StartProfile:
                    logger.info("Generating nvtx start statement.")
                    self._generate(
                        StartProfileStatement,
                        StartProfileStatementGenerator,
                        start.startln,
                        name=start.name,
                    )

            if _is_sequence(self.interface.EndProfile):
                for end in self.interface.EndProfile:
                    logger.info("Generating nvtx end statement.")
                    self._generate(
                        EndProfileStatement,
                        EndProfileStatementGenerator,
                        end.startln,
                    )

    def _generate_insert(self) -> None:
        """Generate free form statement from insert directive."""
        if _is_sequence(self.interface.Insert):
            for insert in self.interface.Insert:
                logger.info("Generating free form statement.")
                self._generate(
                    InsertStatement,
                    InsertStatementGenerator,
                    insert.startln,
                    content=insert.content,
                )
