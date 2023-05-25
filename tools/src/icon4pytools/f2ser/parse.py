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

from icon4pytools.f2ser.exceptions import MissingDerivedTypeError, ParsingError
from numpy.f2py.crackfortran import crackfortran


def crack(path: Path) -> dict:
    return crackfortran(path)[0]


class SubroutineType(Enum):
    RUN = "_run"
    INIT = "_init"


@dataclass
class CodegenContext:
    first_declaration_ln: int
    last_declaration_ln: int
    end_subroutine_ln: int


ParsedSubroutines = dict[str, dict[str, dict[str, any] | CodegenContext]]


@dataclass
class ParsedGranule:
    subroutines: ParsedSubroutines
    last_import_ln: int


class GranuleParser:
    """Parses a Fortran source file and extracts information about its subroutines and variables.

    Attributes:
        granule (Path): A path to the Fortran source file to be parsed.
        dependencies (Optional[list[Path]]): A list of paths to any additional Fortran source files that the input file depends on.

    Example usage:
        parser = GranuleParser(Path("my_file.f90"), dependencies=[Path("common.f90"), Path("constants.f90")])
        parsed_types = parser()
    """

    def __init__(
        self, granule: Path, dependencies: Optional[list[Path]] = None
    ) -> None:
        self.granule_path = granule
        self.dependencies = dependencies

    def __call__(self) -> ParsedGranule:
        """Parse the granule and return the parsed data."""
        subroutines = self.parse_subroutines()
        last_import_ln = self.find_last_fortran_use_statement()
        return ParsedGranule(subroutines=subroutines, last_import_ln=last_import_ln)

    def parse_subroutines(self):
        subroutines = self._extract_subroutines(crack(self.granule_path))
        variables_grouped_by_intent = {
            name: self._extract_intent_vars(routine)
            for name, routine in subroutines.items()
        }
        intrinsic_type_vars, derived_type_vars = self._parse_types(
            variables_grouped_by_intent
        )
        combined_type_vars = self._combine_types(derived_type_vars, intrinsic_type_vars)
        with_lines = self._update_with_codegen_lines(combined_type_vars)
        return with_lines

    def _extract_subroutines(self, parsed: dict[str, any]) -> dict[str, any]:
        """Extract the _init and _run subroutines from the parsed granule.

        Args:
            parsed: A dictionary representing the parsed granule.

        Returns:
            A dictionary containing the extracted _init and _run subroutines.
        """
        subroutines = {}
        for elt in parsed["body"]:
            name = elt["name"]
            if SubroutineType.RUN.value in name or SubroutineType.INIT.value in name:
                subroutines[name] = elt

        if len(subroutines) != 2:
            raise ParsingError(
                f"Did not find _init and _run subroutines in {self.granule_path}"
            )

        return subroutines

    @staticmethod
    def _extract_intent_vars(subroutine: dict) -> dict:
        """Extract variables grouped by their intent.

        Args:
            subroutine (dict): A dictionary representing the subroutine.

        Returns:
            A dictionary representing variables grouped by their intent.
        """
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
        """Parse the intrinsic and derived type variables of each subroutine and intent from a parsed granule dictionary.

        Args:
            parsed (dict): A dictionary containing the parsed information of a granule.

        Returns:
            tuple[dict, dict]: A tuple containing two dictionaries. The first one maps each subroutine and intent to a dictionary of intrinsic type variables. The second one maps each subroutine and intent to a dictionary of derived type variables.
        """
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
        """Parse the derived types defined in the input dictionary by adding their type definitions.

        Args:
            derived_types (dict): A dictionary containing the derived types.

        Returns:
            dict: A dictionary containing the parsed derived types with their type definitions.

        Raises:
            MissingDerivedTypeError: If the type definition for a derived type could not be found in any of the dependency files.
        """
        derived_type_defs = {}
        if self.dependencies:
            for dep in self.dependencies:
                parsed = crack(dep)
                for block in parsed["body"]:
                    if block["block"] == "type":
                        derived_type_defs[block["name"]] = block["vars"]

        for _, subroutine_vars in derived_types.items():
            for _, intent_vars in subroutine_vars.items():
                for _, var in intent_vars.items():
                    if var["typespec"] == "type":
                        typename = var["typename"]
                        if typename in derived_type_defs:
                            var["typedef"] = derived_type_defs[typename]
                            var["decomposed"] = True
                        else:
                            raise MissingDerivedTypeError(
                                f"Could not find type definition for TYPE: {typename} in dependency files: {self.dependencies}"
                            )

        return self._decompose_derived_types(derived_types)

    @staticmethod
    def _decompose_derived_types(derived_types: dict) -> dict:
        """Decompose derived types into individual subtypes.

        Args:
            derived_types (dict): A dictionary containing the derived types to be decomposed.

        Returns:
            dict: A dictionary containing the decomposed derived types, with each subtype represented by a separate entry.
        """
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
                            new_var_dict["ptr_var"] = subtype_name
                    else:
                        decomposed_vars[subroutine][intent][var_name] = var_dict

        return decomposed_vars

    @staticmethod
    def _combine_types(derived_type_vars: dict, intrinsic_type_vars: dict) -> dict:
        """Combine intrinsic and derived type variables and returns a dictionary with the combined result.

        Args:
            derived_type_vars (dict): A dictionary with derived type variables.
            intrinsic_type_vars (dict): A dictionary with intrinsic type variables.

        Returns:
            dict: A dictionary with the combined intrinsic and derived type variables.
        """
        combined = deepcopy(intrinsic_type_vars)
        for subroutine_name in combined:
            for intent in combined[subroutine_name]:
                new_vars = derived_type_vars[subroutine_name][intent]
                combined[subroutine_name][intent].update(new_vars)
        return combined

    def _update_with_codegen_lines(self, parsed_types: dict) -> dict:
        """Update the parsed_types dictionary with the line numbers for codegen.

        Args:
            parsed_types (dict): A dictionary containing the parsed intrinsic and derived types.

        Returns:
            dict: A dictionary containing the parsed intrinsic and derived types with line numbers for codegen.
        """
        with_lines = deepcopy(parsed_types)
        for subroutine in with_lines:
            for intent in with_lines[subroutine]:
                with_lines[subroutine][intent][
                    "codegen_ctx"
                ] = self.get_subroutine_lines(subroutine)
        return with_lines

    def find_last_fortran_use_statement(self):
        with open(self.granule_path) as f:
            file_contents = f.readlines()

        # Reverse the order of the lines so we can search from the end
        file_contents.reverse()

        # Look for the last USE statement
        use_ln = None
        for i, line in enumerate(file_contents):
            if line.strip().lower().startswith("use"):
                use_ln = len(file_contents) - i
                if i > 0 and file_contents[i - 1].strip().lower() == "#endif":
                    # If the USE statement is preceded by an #endif statement, return the line number after the #endif statement
                    return use_ln + 1
                else:
                    return use_ln
        return None

    def get_subroutine_lines(self, subroutine_name: str) -> CodegenContext:
        """Return CodegenContext object containing line numbers of the last declaration statement and the code before the end of the given subroutine.

        Args:
            subroutine_name (str): Name of the subroutine to look for in the code.

        Returns:
            CodegenContext: Object containing the line number of the last declaration statement and the line number of the last line of the code before the end of the given subroutine.
        """
        with open(self.granule_path) as f:
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
        declaration_pattern = r".*::\s*(\w+\b)"
        declaration_pattern_lines = [
            i
            for i, line in enumerate(
                code.splitlines()[start_subroutine_ln:end_subroutine_ln]
            )
            if re.search(declaration_pattern, line)
        ]
        if not declaration_pattern_lines:
            raise ParsingError(f"No declarations found in {self.granule_path}")
        last_declaration_ln = declaration_pattern_lines[-1] + start_subroutine_ln + 1
        first_declaration_ln = declaration_pattern_lines[0] + start_subroutine_ln

        pre_end_subroutine_ln = (
            end_subroutine_ln - 1
        )  # we want to generate the code before the end of the subroutine

        return CodegenContext(
            first_declaration_ln, last_declaration_ln, pre_end_subroutine_ln
        )
