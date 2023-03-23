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

from copy import deepcopy
from enum import Enum
from pathlib import Path

from numpy.f2py.crackfortran import crackfortran

from icon4py.serialisation.interface import SerialisationInterface


class SubroutineType(Enum):
    RUN = "_run"
    INIT = "_init"


class GranuleParser:
    def __init__(self, path: Path):
        self.path = path
        self.cracked = crackfortran(self.path)[0]

    def parse(self) -> SerialisationInterface:
        subroutines = self._extract_subroutines()
        variables = {
            name: self._extract_intent_vars(routine)
            for name, routine in subroutines.items()
        }
        intrinsic_type_vars = self._remove_derived_type_vars(variables)  # noqa

        # todo: handle derived type vars
        # todo: find post declarations line number (required for input field serialisation codegen)
        # todo: find pre end subroutine line number (required for output field serialisation codegen)
        # todo: construct SerialisationInterface (object to pass to code generator)
        return SerialisationInterface  # temporary

    @staticmethod
    def _remove_derived_type_vars(routine_vars: dict):
        copy = deepcopy(routine_vars)
        for subroutine in copy:
            for intent in copy[subroutine]:
                copy[subroutine][intent] = {
                    var_name: var_dict
                    for var_name, var_dict in copy[subroutine][intent].items()
                    if var_dict["typespec"] != "type"
                }
        return copy

    def _extract_subroutines(self):
        subroutines = {}
        for elt in self.cracked["body"]:
            name = elt["name"]
            if SubroutineType.RUN.value in name:
                subroutines[name] = elt
            elif SubroutineType.INIT.value in name:
                subroutines[name] = elt
        return subroutines

    @staticmethod
    def _extract_intent_vars(subroutine: dict):
        intents = ["in", "inout", "out"]
        result = {}
        for var in subroutine["vars"]:
            var_intent = subroutine["vars"][var]["intent"]
            common_intents = list(set(intents).intersection(var_intent))
            for intent in common_intents:
                if intent not in result:
                    result[intent] = {}
                result[intent][var] = subroutine["vars"][var]
        return result

    # todo
    def _find_post_variable_declaration(self) -> int:
        pass

    # todo
    def _find_end_of_subroutine(self) -> int:
        pass

    # todo
    def _handle_derived_types(self):
        pass
