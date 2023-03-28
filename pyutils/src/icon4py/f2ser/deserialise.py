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

from icon4py.f2ser.interface import (
    FieldSerialisationData,
    SavepointData,
    SerialisationInterface,
)
from icon4py.f2ser.parse import ParsedGranule


class ParsedGranuleDeserialiser:
    def __init__(self, parsed: ParsedGranule):
        self.parsed = parsed
        self.savepoints: list[SavepointData] = []

    def deserialise(self) -> SerialisationInterface:
        """
        Deserialises the parsed granule and returns a serialisation interface.

        Returns:
            A `SerialisationInterface` object representing the deserialised data.
        """
        self._create_savepoints()
        init_data = self._get_init_data()
        return SerialisationInterface(init=init_data, savepoint=self.savepoints)

    def _create_savepoints(self) -> None:
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

        for ln in var_dict["codegen_lines"]:
            self.savepoints.append(
                SavepointData(
                    name=savepoint_name,
                    startln=ln,
                    endln=ln,
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
        # todo: handle other dimension cases e.g. (min_rl:)
        dimension = var_data.get("dimension", None)

        if dimension is not None:
            dim_string = ",".join(dimension)
            association = f"{var_name}({dim_string})"
        else:
            association = var_name
        return association

    @staticmethod
    def _get_init_data() -> None:
        # todo: implement the logic for getting init data
        return None
