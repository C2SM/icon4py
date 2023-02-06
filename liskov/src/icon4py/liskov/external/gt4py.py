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

from functional.ffront.decorator import Program
from typing_extensions import Any

from icon4py.liskov.codegen.interface import DeserialisedDirectives
from icon4py.liskov.common import Step
from icon4py.liskov.logger import setup_logger
from icon4py.liskov.parsing.exceptions import (
    IncompatibleFieldError,
    UnknownStencilError,
)
from icon4py.pyutils.metadata import get_field_infos


logger = setup_logger(__name__)


class UpdateFieldsWithGt4PyStencils(Step):
    _STENCIL_PACKAGES = ["atm_dyn_iconam", "advection"]

    def __init__(self, parsed: DeserialisedDirectives):
        self.parsed = parsed

    def __call__(self, data: Any = None) -> DeserialisedDirectives:
        logger.info("Updating parsed fields with data from icon4py stencils...")

        for s in self.parsed.StartStencil:
            gt4py_program = self._collect_icon4py_stencil(s.name)
            gt4py_field_info = get_field_infos(gt4py_program)
            for f in s.fields:
                try:
                    field_info = gt4py_field_info[f.variable]
                except KeyError:
                    raise IncompatibleFieldError(
                        f"Used field variable name that is incompatible with the expected field names defined in {s.name} in icon4py."
                    )
                f.out = field_info.out
                f.inp = field_info.inp
                # f.type = field_info.field.type # todo: is this needed?
        return self.parsed

    def _collect_icon4py_stencil(self, stencil_name: str) -> Program:
        """Collect and return the ICON4PY stencil program with the given name."""
        err_counter = 0
        for pkg in self._STENCIL_PACKAGES:

            try:
                module_name = f"icon4py.{pkg}.{stencil_name}"
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                err_counter += 1

        if err_counter == len(self._STENCIL_PACKAGES):
            raise UnknownStencilError(f"Did not find module: {stencil_name}")

        module_members = getmembers(module)
        found_stencil = [elt for elt in module_members if elt[0] == stencil_name]

        if len(found_stencil) == 0:
            raise UnknownStencilError(
                f"Did not find member: {stencil_name} in module: {module.__name__}"
            )

        return found_stencil[0][1]
