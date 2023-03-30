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
import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from numpy.f2py.crackfortran import crackfortran

from icon4py.f2ser.exceptions import MissingDerivedTypeError, ParsingError


CodeGenLines = list[int]
ParsedGranule = dict[str, dict[str, dict[str, any] | CodeGenLines]]


def crack(path: Path) -> dict:
    return crackfortran(path)[0]


class SubroutineType(Enum):
    RUN = "_run"
    INIT = "_init"


@dataclass
class CodegenContext:
    last_intent_ln: int
    end_subroutine_ln: int


class GranuleParser:
    """Parses a Fortran source file and extracts information about its subroutines and variables.

    Attributes:
        granule (Path): A path to the Fortran source file to be parsed.
        dependencies (Optional[list[Path]]): A list of paths to any additional Fortran source files that the input file depends on.

    Methods:
        parse(): Parses the input file and returns a dictionary with information about its subroutines and variables.

    Example usage:
        parser = GranuleParser(Path("my_file.f90"), dependencies=[Path("common.f90"), Path("constants.f90")])
        parsed_types = parser.parse()
    """

    def __init__(
        self, granule: Path, dependencies: Optional[list[Path]] = None
    ) -> None:
        self.granule = granule
        self.dependencies = dependencies

    def parse(self) -> ParsedGranule:
        parsed = crack(self.granule)

        subroutines = self._extract_subroutines(parsed)

        variables_grouped_by_intent = {
            name: self._extract_intent_vars(routine)
            for name, routine in subroutines.items()
        }

        intrinsic_type_vars, derived_type_vars = self._parse_types(
            variables_grouped_by_intent
        )

        combined_type_vars = self._combine_types(derived_type_vars, intrinsic_type_vars)

        vars_with_lines = self._update_with_codegen_lines(combined_type_vars)

        return vars_with_lines

    def _extract_subroutines(self, parsed: dict) -> dict:
        subroutines: dict = {}
        for elt in parsed["body"]:
            name = elt["name"]
            if SubroutineType.RUN.value in name:
                subroutines[name] = elt
            elif SubroutineType.INIT.value in name:
                subroutines[name] = elt

        if len(subroutines) != 2:
            raise ParsingError(
                f"Did not find _init and _run subroutines in {self.granule}"
            )

        return subroutines

    @staticmethod
    def _extract_intent_vars(subroutine: dict) -> dict:
        intents = ["in", "inout", "out"]
        result: dict = {}
        for var in subroutine["vars"]:
            var_intent = subroutine["vars"][var]["intent"]
            common_intents = list(set(intents).intersection(var_intent))
            for intent in common_intents:
                if intent not in result:
                    result[intent] = {}
                result[intent][var] = subroutine["vars"][var]
        return result

    def _parse_types(self, parsed: dict) -> tuple[dict, dict]:
        intrinsic_types: dict = {}
        derived_types: dict = {}

        for subroutine, subroutine_vars in parsed.items():
            intrinsic_types[subroutine] = {}
            derived_types[subroutine] = {}

            for intent, intent_vars in subroutine_vars.items():
                intrinsic_vars = {}
                derived_vars = {}

                for var_name, var_dict in intent_vars.items():
                    if var_dict["typespec"] != "type":
                        intrinsic_vars[var_name] = var_dict
                    else:
                        derived_vars[var_name] = var_dict

                intrinsic_types[subroutine][intent] = intrinsic_vars
                derived_types[subroutine][intent] = derived_vars

        return intrinsic_types, self._parse_derived_types(derived_types)

    def _parse_derived_types(self, derived_types: dict) -> dict:
        # Create a dictionary that maps the typename to the typedef for each derived type
        derived_type_defs = {}
        for dep in self.dependencies:
            parsed = crack(dep)
            for block in parsed["body"]:
                if block["block"] == "type":
                    derived_type_defs[block["name"]] = block["vars"]

        # Iterate over the derived types and add the typedef for each derived type
        for _, subroutine_vars in derived_types.items():
            for _, intent_vars in subroutine_vars.items():
                for _, var in intent_vars.items():
                    if var["typespec"] == "type":
                        typename = var["typename"]
                        if typename in derived_type_defs:
                            var["typedef"] = derived_type_defs[typename]
                        else:
                            raise MissingDerivedTypeError(
                                f"Could not find type definition for TYPE: {typename} in dependency files: {self.dependencies}"
                            )

        return self._decompose_derived_types(derived_types)

    @staticmethod
    def _decompose_derived_types(derived_types: dict) -> dict:
        decomposed_vars: dict = {}
        for subroutine, subroutine_vars in derived_types.items():
            decomposed_vars[subroutine] = {}
            for intent, intent_vars in subroutine_vars.items():
                decomposed_vars[subroutine][intent] = {}
                for var_name, var_dict in intent_vars.items():
                    if "typedef" in var_dict:
                        typedef = var_dict["typedef"]
                        del var_dict["typedef"]
                        for subtype_name, subtype_spec in typedef.items():
                            new_type_name = f"{var_name}_{subtype_name}"
                            new_var_dict = var_dict.copy()
                            new_var_dict.update(subtype_spec)
                            decomposed_vars[subroutine][intent][
                                new_type_name
                            ] = new_var_dict
                    else:
                        decomposed_vars[subroutine][intent][var_name] = var_dict

        return decomposed_vars

    @staticmethod
    def _combine_types(derived_type_vars: dict, intrinsic_type_vars: dict) -> dict:
        combined = deepcopy(intrinsic_type_vars)
        for subroutine_name in combined:
            for intent in combined[subroutine_name]:
                new_vars = derived_type_vars[subroutine_name][intent]
                combined[subroutine_name][intent].update(new_vars)
        return combined

    def _update_with_codegen_lines(self, parsed_types: dict) -> dict:
        with_lines = deepcopy(parsed_types)
        for subroutine in with_lines:
            ctx = self.get_line_numbers(subroutine)
            for intent in with_lines[subroutine]:
                if intent == "in":
                    ln = ctx.last_intent_ln
                elif intent == "inout":
                    continue
                elif intent == "out":
                    ln = ctx.end_subroutine_ln
                else:
                    raise ValueError(f"Unrecognized intent: {intent}")
                with_lines[subroutine][intent]["codegen_line"] = ln
        return with_lines

    def get_line_numbers(self, subroutine_name: str) -> CodegenContext:
        with open(self.granule, "r") as f:
            code = f.read()

        # Find the line number where the subroutine is defined
        start_subroutine_pattern = r"SUBROUTINE\s+" + subroutine_name + r"\s*\("
        end_subroutine_pattern = r"END\s+SUBROUTINE\s+" + subroutine_name + r"\s*"
        start_match = re.search(start_subroutine_pattern, code)
        end_match = re.search(end_subroutine_pattern, code)
        if start_match is None or end_match is None:
            return None
        start_subroutine_ln = code[: start_match.start()].count("\n") + 1
        end_subroutine_ln = code[: end_match.start()].count("\n") + 1

        # Find the last intent statement line number in the subroutine
        intent_pattern = r"\bINTENT\b"
        intent_pattern_lines = [
            i
            for i, line in enumerate(
                code.splitlines()[start_subroutine_ln:end_subroutine_ln]
            )
            if re.search(intent_pattern, line)
        ]
        if not intent_pattern_lines:
            raise ParsingError(f"No INTENT declarations found in {self.granule}")
        last_intent_ln = intent_pattern_lines[-1] + start_subroutine_ln + 1

        return CodegenContext(last_intent_ln, end_subroutine_ln)
