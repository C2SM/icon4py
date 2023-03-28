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

    def deserialise(self):

        savepoints = []

        for subroutine_name, field_dict in self.parsed.items():
            for intent, variables in field_dict.items():
                savepoint_name = f"{subroutine_name}_{intent}"
                savepoints.append(self._make_savepoint(savepoint_name, variables))

        init_data = None  # todo: make init data

        return SerialisationInterface(init=init_data, savepoint=savepoints)

    def _make_savepoint(self, savepoint_name: str, variables: dict):
        savepoints = []
        fields = []
        metadata = None  # todo: decide how to handle metadata

        field_vals = {k: v for k, v in variables.items() if isinstance(v, dict)}

        for var_name, var_data in field_vals.items():

            dimension = var_data.get("dimension", None)

            if dimension is not None:
                dim_string = ",".join(dimension)
                association = f"{var_name}({dim_string})"
            else:
                association = var_name

            field = FieldSerialisationData(variable=var_name, association=association)
            fields.append(field)

        for ln in vars["codegen_lines"]:
            savepoints.append(
                SavepointData(
                    name=savepoint_name,
                    startln=ln,
                    endln=ln,
                    fields=fields,
                    metadata=metadata,
                )
            )

        return savepoints
