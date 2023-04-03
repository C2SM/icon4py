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
        self.data = {"Savepoint": [], "Init": ...}

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
                self._create_savepoint(subroutine_name, intent, var_dict)

    def _create_savepoint(
        self, subroutine_name: str, intent: str, var_dict: dict
    ) -> None:
        """Create a savepoint for the given variables.

        Args:
            savepoint_name: The name of the savepoint.
            var_dict: A dictionary representing the variables to be saved.
        """
        fields = []
        metadata = None  # todo: decide how to handle metadata
        field_vals = {k: v for k, v in var_dict.items() if isinstance(v, dict)}

        for var_name, var_data in field_vals.items():
            association = self._create_association(var_data, var_name)
            if var_data.get("decomposed"):
                fields.append(
                    FieldSerialisationData(
                        variable=var_name,
                        association=association,
                        decomposed=var_data["decomposed"],
                        dimension=var_data["dimension"],
                        typespec=var_data["typespec"],
                        typename=var_data["typename"],
                        ptr_var=var_data["ptr_var"],
                    )
                )
            else:
                fields.append(
                    FieldSerialisationData(variable=var_name, association=association)
                )

        self.data["Savepoint"].append(
            SavepointData(
                subroutine=subroutine_name,
                intent=intent,
                startln=var_dict["codegen_line"],
                fields=fields,
                metadata=metadata,
            )
        )

    @staticmethod
    def get_slice_expression(var_name, dimension):
        idx = dimension.split()[-1].lstrip("-+")
        return f"{var_name}({idx}:)"

    def _create_association(self, var_data, var_name):
        dimension = var_data.get("dimension", None)
        if dimension is not None:
            if ":" not in dimension:
                return self.get_slice_expression(var_name, dimension[0])
            else:
                dim_string = ",".join(dimension)
                return f"{var_name}({dim_string})"
        else:
            return var_name

    def _make_init_data(self) -> None:
        lns = []
        for _, intent_dict in self.parsed.items():
            for intent, var_dict in intent_dict.items():
                if intent == "in":
                    lns.append(var_dict["codegen_line"])
        startln = min(lns)
        self.data["Init"] = InitData(startln=startln, directory=self.directory)

    def _merge_out_inout_fields(self):
        for _, intent_dict in self.parsed.items():
            if "inout" in intent_dict:
                intent_dict["in"].update(intent_dict["inout"])
                intent_dict["out"].update(intent_dict["inout"])
                del intent_dict["inout"]
