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

from icon4py.f2ser.parse import ParsedGranule
from icon4py.liskov.codegen.serialisation.interface import (
    FieldSerialisationData,
    InitData,
    SavepointData,
    SerialisationInterface,
)


class ParsedGranuleDeserialiser:
    def __init__(self, parsed: ParsedGranule, directory: str):
        self.parsed = parsed
        self.directory = directory
        self.data = {"savepoint": [], "init": ...}

    def deserialise(self) -> SerialisationInterface:
        """
        Deserialises the parsed granule and returns a serialisation interface.

        Returns:
            A `SerialisationInterface` object representing the deserialised data.
        """
        self._merge_out_inout_fields()
        self._make_savepoints()
        self._make_init_data()
        return SerialisationInterface(**self.data)

    def _make_savepoints(self) -> None:
        """Create savepoints for each subroutine and intent in the parsed granule."""
        for subroutine_name, intent_dict in self.parsed.items():
            for intent, var_dict in intent_dict.items():
                savepoint_name = self._create_savepoint_name(subroutine_name, intent)
                self._create_savepoint(savepoint_name, var_dict)

    @staticmethod
    def _create_savepoint_name(subroutine_name: str, intent: str) -> str:
        return f"{subroutine_name}_{intent}"

    def _create_savepoint(self, savepoint_name: str, var_dict: dict) -> None:
        """Create a savepoint for the given variables.

        Args:
            savepoint_name: The name of the savepoint.
            var_dict: A dictionary representing the variables to be saved.
        """
        fields = []
        metadata = None  # todo: decide how to handle metadata
        field_vals = {k: v for k, v in var_dict.items() if isinstance(v, dict)}

        for var_name, var_data in field_vals.items():
            association = self._get_variable_association(var_data, var_name)
            field = FieldSerialisationData(variable=var_name, association=association)
            fields.append(field)

        self.data["savepoint"].append(
            SavepointData(
                name=savepoint_name,
                startln=var_dict["codegen_line"],
                fields=fields,
                metadata=metadata,
            )
        )

    @staticmethod
    def _get_variable_association(var_data: dict, var_name: str) -> str:
        """
        Generate a string representing the association of a variable with its dimensions.

        Parameters:
            var_data (dict): A dictionary containing information about the variable, including its dimensions.
            var_name (str): The name of the variable.

        Returns:
            str: A string representing the association of the variable with its dimensions, formatted as
                "var_name(dim1,dim2,...)" if the variable has dimensions, or simply "var_name" otherwise.
        """
        # todo: handle other dimension cases e.g. verts_end_index(min_rlvert:)
        dimension = var_data.get("dimension", None)

        if dimension is not None:
            dim_string = ",".join(dimension)
            association = f"{var_name}({dim_string})"
        else:
            association = var_name
        return association

    def _make_init_data(self) -> None:
        lns = []
        for _, intent_dict in self.parsed.items():
            for intent, var_dict in intent_dict.items():
                if intent == "in":
                    lns.append(var_dict["codegen_line"])
        startln = min(lns)
        self.data["init"] = InitData(startln=startln, directory=self.directory)

    def _merge_out_inout_fields(self):
        for _, intent_dict in self.parsed.items():
            if "inout" in intent_dict:
                intent_dict["in"].update(intent_dict["inout"])
                intent_dict["out"].update(intent_dict["inout"])
                del intent_dict["inout"]
