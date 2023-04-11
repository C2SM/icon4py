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

from icon4py.liskov.codegen.common import CodeGenerator
from icon4py.liskov.codegen.serialisation.interface import SerialisationInterface
from icon4py.liskov.codegen.serialisation.template import (
    InitStatement,
    InitStatementGenerator,
    SavepointStatement,
    SavepointStatementGenerator,
)
from icon4py.liskov.codegen.types import GeneratedCode
from icon4py.liskov.logger import setup_logger


logger = setup_logger(__name__)


class SerialisationGenerator(CodeGenerator):
    def __init__(self, ser_iface: SerialisationInterface):
        super().__init__()
        self.ser_iface = ser_iface

    def __call__(self, data: Any = None) -> list[GeneratedCode]:
        """Generate all f90 code for integration."""
        self._generate_init()
        self._generate_savepoints()
        return self.generated

    def _generate_init(self) -> None:
        logger.info("Generating pp_ser initialisation statement.")
        self._generate(
            InitStatement,
            InitStatementGenerator,
            self.ser_iface.Init.startln,
            directory=self.ser_iface.Init.directory,
            prefix=self.ser_iface.Init.prefix,
        )

    def _generate_savepoints(self) -> None:
        for i, savepoint in enumerate(self.ser_iface.Savepoint):
            logger.info("Generating pp_ser savepoint statement.")
            self._generate(
                SavepointStatement,
                SavepointStatementGenerator,
                self.ser_iface.Savepoint[i].startln,
                savepoint=savepoint,
            )
