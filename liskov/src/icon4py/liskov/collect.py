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
from dataclasses import dataclass
from inspect import getmembers
from pathlib import Path

from functional.ffront.decorator import Program

from icon4py.liskov.directives import IDENTIFIER, RawDirective


@dataclass(frozen=True)
class Collected:
    string: str
    lnumber: int


class DirectivesCollector:
    def __init__(self, filepath: Path) -> None:
        """Class which collects all DSL directives as is in a given file.

        Args:
            filepath: Path to file to scan for directives.
        """
        self.filepath = filepath
        self.directives = self._collect_directives()

    @staticmethod
    def _process_collected(collected: list[Collected]) -> RawDirective:
        directive_string = "".join([c.string for c in collected])
        abs_startln = collected[0].lnumber + 1
        abs_endln = collected[-1].lnumber + 1
        return RawDirective(directive_string, startln=abs_startln, endln=abs_endln)

    def _collect_directives(self) -> list[RawDirective]:
        """Scan filepath for directives and returns them along with their line numbers."""
        directives = []
        with self.filepath.open() as f:

            collected_directives = []
            for lnumber, string in enumerate(f):

                if IDENTIFIER in string:
                    stripped = string.strip()
                    eol = stripped[-1]
                    collected = Collected(string, lnumber)
                    collected_directives.append(collected)

                    match eol:
                        case ")":
                            directives.append(
                                self._process_collected(collected_directives)
                            )
                            collected_directives = []
                        case "&":
                            continue
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

        return found_stencil[0]
