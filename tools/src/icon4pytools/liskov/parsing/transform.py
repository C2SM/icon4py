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

import icon4pytools.liskov.parsing.types as ts
from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.codegen.integration.interface import (
    EndDeleteData,
    IntegrationCodeInterface,
    StartDeleteData,
)
from icon4pytools.liskov.pipeline.definition import Step


logger = setup_logger(__name__)


class TransformFuseStencils(Step):
    def __init__(self, parsed: IntegrationCodeInterface, fused: bool) -> None:
        self.parsed = parsed
        self.fused = fused

    def __call__(self, data: Any = None) -> ts.ParsedDict:
        """Parse the directives and return a dictionary of parsed directives and their associated content.

        Returns:
            ParsedType: Dictionary of parsed directives and their associated content.
        """
        if self.fused:
            logger.info("Transforming single stencils.")

            removeStartStencil = []
            removeEndStencil = []

            for startFused, endFused in zip(
                self.parsed.StartFusedStencil, self.parsed.EndFusedStencil, strict=True
            ):
                for startSingle, endSingle in zip(
                    self.parsed.StartStencil, self.parsed.EndStencil, strict=True
                ):
                    if (
                        startFused.startln < startSingle.startln
                        and startSingle.startln < endFused.startln
                        and startFused.startln < endSingle.startln
                        and endSingle.startln < endFused.startln
                    ):
                        try:
                            self.parsed.StartDelete.append(StartDeleteData(startSingle))
                        except AttributeError:
                            self.parsed.StartDelete = [StartDeleteData(startSingle)]
                        try:
                            self.parsed.EndDelete.append(EndDeleteData(endSingle))
                        except AttributeError:
                            self.parsed.EndDelete = [EndDeleteData(endSingle)]
                        removeStartStencil.append(startSingle)
                        removeEndStencil.append(endSingle)

            self.parsed.StartStencil = [
                x for x in self.parsed.StartStencil if x not in removeStartStencil
            ]
            self.parsed.EndStencil = [
                x for x in self.parsed.EndStencil if x not in removeEndStencil
            ]

        else:
            logger.info("Removing fused stencils.")

            self.parsed.StartFusedStencil = []
            self.parsed.EndFusedStencil = []

        return self.parsed
