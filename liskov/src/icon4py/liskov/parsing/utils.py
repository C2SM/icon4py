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
from typing import Type

from functional.ffront.decorator import Program

from icon4py.liskov.parsing.types import DirectiveType, TypedDirective


def pretty_print_typed_directive(directive: TypedDirective):
    return f"Directive: {directive.string}, start line: {directive.startln}, end line: {directive.endln}\n"


def extract_directive(
    directives: list[TypedDirective],
    required_type: Type[DirectiveType],
) -> list[TypedDirective]:
    directives = [d for d in directives if type(d.directive_type) == required_type]
    return directives


class StencilCollector:
    _STENCIL_PACKAGES = ["atm_dyn_iconam", "advection"]

    def __init__(self, name: str):
        self.name = name

    @property
    def fvprog(self) -> Program:
        return self._collect_stencil_program()[1]

    def _collect_stencil_program(self) -> tuple[str, Program]:
        err_counter = 0
        for pkg in self._STENCIL_PACKAGES:

            try:
                module_name = f"icon4py.{pkg}.{self.name}"
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                err_counter += 1

        if err_counter == len(self._STENCIL_PACKAGES):
            raise Exception(f"Did not find module: {self.name}")

        module_members = getmembers(module)
        found_stencil = [elt for elt in module_members if elt[0] == self.name]

        if len(found_stencil) == 0:
            raise Exception(
                f"Did not find member: {self.name} in module: {module_name}"
            )
        # todo: More specific exception

        return found_stencil[0]
