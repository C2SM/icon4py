# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.codegen.serialisation.interface import SerialisationCodeInterface
from icon4pytools.liskov.codegen.serialisation.template import (
    ImportStatement,
    ImportStatementGenerator,
    SavepointStatement,
    SavepointStatementGenerator,
)
from icon4pytools.liskov.codegen.shared.generate import CodeGenerator
from icon4pytools.liskov.codegen.shared.types import GeneratedCode


logger = setup_logger(__name__)


class SerialisationCodeGenerator(CodeGenerator):
    def __init__(self, interface: SerialisationCodeInterface, multinode: bool = False):
        super().__init__()
        self.interface = interface
        self.multinode = multinode

    def __call__(self, data: Any = None) -> list[GeneratedCode]:
        """Generate all f90 code for integration."""
        self._generate_import()
        self._generate_savepoints()
        return self.generated

    def _generate_import(self) -> None:
        if self.multinode:
            self._generate(
                ImportStatement,
                ImportStatementGenerator,
                self.interface.Import.startln,
            )

    def _generate_savepoints(self) -> None:
        init_complete = False
        for i, savepoint in enumerate(self.interface.Savepoint):
            logger.info("Generating pp_ser savepoint statement.")
            if init_complete:
                self._generate(
                    SavepointStatement,
                    SavepointStatementGenerator,
                    self.interface.Savepoint[i].startln,
                    savepoint=savepoint,
                    multinode=self.multinode,
                )
            else:
                self._generate(
                    SavepointStatement,
                    SavepointStatementGenerator,
                    self.interface.Savepoint[i].startln,
                    savepoint=savepoint,
                    init=self.interface.Init,
                    multinode=self.multinode,
                )
                init_complete = True
