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

import importlib
from inspect import getmembers
from typing import Any, ClassVar, Sequence

from gt4py.next.ffront.decorator import Program

from icon4pytools.common import ICON4PY_MODEL_QUALIFIED_NAME
from icon4pytools.common.logger import setup_logger
from icon4pytools.icon4pygen.metadata import get_stencil_info
from icon4pytools.liskov.codegen.integration.interface import (
    BaseStartStencilData,
    IntegrationCodeInterface,
)
from icon4pytools.liskov.external.exceptions import IncompatibleFieldError, UnknownStencilError
from icon4pytools.liskov.pipeline.definition import Step


logger = setup_logger(__name__)


class UpdateFieldsWithGt4PyStencils(Step):
    _STENCIL_PACKAGES: ClassVar[list[str]] = [
        "atmosphere.dycore",
        "atmosphere.advection",
        "atmosphere.diffusion.stencils",
        "common.interpolation.stencils",
    ]

    def __init__(self, parsed: IntegrationCodeInterface):
        self.parsed = parsed

    def __call__(self, data: Any = None) -> IntegrationCodeInterface:
        logger.info("Updating parsed fields with data from icon4py stencils...")

        self._set_in_out_field(self.parsed.StartStencil)
        self._set_in_out_field(self.parsed.StartFusedStencil)

        return self.parsed

    def _set_in_out_field(self, startStencil: Sequence[BaseStartStencilData]) -> None:
        for s in startStencil:
            gt4py_program = self._collect_icon4py_stencil(s.name)
            gt4py_stencil_info = get_stencil_info(gt4py_program)
            gt4py_fields = gt4py_stencil_info.fields
            for f in s.fields:
                try:
                    field_info = gt4py_fields[f.variable]
                except KeyError as err:
                    error_msg = f"Used field variable name ({f.variable}) that is incompatible with the expected field names defined in {s.name} in icon4py."
                    raise IncompatibleFieldError(error_msg) from err
                f.out = field_info.out
                f.inp = field_info.inp

    def _collect_icon4py_stencil(self, stencil_name: str) -> Program:
        """Collect and return the ICON4PY stencil program with the given name."""
        err_counter = 0
        for pkg in self._STENCIL_PACKAGES:
            try:
                module_name = f"{ICON4PY_MODEL_QUALIFIED_NAME}.{pkg}.{stencil_name}"
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                err_counter += 1

        if err_counter == len(self._STENCIL_PACKAGES):
            raise UnknownStencilError(f"Did not find module: {stencil_name}")

        module_members = getmembers(module)
        found_stencil = [elt for elt in module_members if elt[0] == stencil_name]

        if len(found_stencil) == 0:
            raise UnknownStencilError(
                f"Did not find module member: {stencil_name} in module: {module.__name__}"
            )

        return found_stencil[0][1]
