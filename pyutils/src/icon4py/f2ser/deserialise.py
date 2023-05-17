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

from icon4py.f2ser.parse import CodegenContext, ParsedGranule
from icon4py.liskov.codegen.serialisation.interface import (
    FieldSerialisationData,
    ImportData,
    InitData,
    SavepointData,
    SerialisationCodeInterface,
)


class ParsedGranuleDeserialiser:
    def __init__(
        self,
        parsed: ParsedGranule,
        directory: str = ".",
        prefix: str = "f2ser",
        multinode: bool = False,
    ):
        self.parsed = parsed
        self.directory = directory
        self.prefix = prefix
        self.multinode = multinode
        self.data = {"Savepoint": [], "Init": ..., "Import": ...}

    def __call__(self) -> SerialisationCodeInterface:
        """Deserialise the parsed granule and returns a serialisation interface.

        Returns:
            A `SerialisationInterface` object representing the deserialised data.
        """
        self._merge_out_inout_fields()
        self._make_savepoints()
        self._make_init_data()
        self._make_imports()
        return SerialisationCodeInterface(**self.data)

    def _make_savepoints(self) -> None:
        """Create savepoints for each subroutine and intent in the parsed granule.

        Returns:
            None.
        """
        for subroutine_name, intent_dict in self.parsed.subroutines.items():
            for intent, var_dict in intent_dict.items():
                self._create_savepoint(subroutine_name, intent, var_dict)

    def _create_savepoint(
        self, subroutine_name: str, intent: str, var_dict: dict
    ) -> None:
        """Create a savepoint for the given variables.

        Args:
            subroutine_name: The name of the subroutine.
            intent: The intent of the fields to be serialised.
            var_dict: A dictionary representing the variables to be saved.

        Returns:
            None.
        """
        field_vals = {k: v for k, v in var_dict.items() if isinstance(v, dict)}
        fields = [
            FieldSerialisationData(
                variable=var_name,
                association=self._create_association(var_data, var_name),
                decomposed=var_data["decomposed"]
                if var_data.get("decomposed")
                else False,
                dimension=var_data.get("dimension"),
                typespec=var_data.get("typespec"),
                typename=var_data.get("typename"),
                ptr_var=var_data.get("ptr_var"),
            )
            for var_name, var_data in field_vals.items()
        ]

        self.data["Savepoint"].append(
            SavepointData(
                subroutine=subroutine_name,
                intent=intent,
                startln=self._get_codegen_line(var_dict["codegen_ctx"], intent),
                fields=fields,
                metadata=None,  # todo: currently not using metadata
            )
        )

    @staticmethod
    def get_slice_expression(var_name: str, dimension: str) -> str:
        """Return a string representing a slice expression for a given variable name and dimension.

        Args:
            var_name (str): The name of the variable.
            dimension (str): The dimension of the variable.

        Returns:
            str: A string representing a slice expression.
        """
        idx = dimension.split()[-1].lstrip("-+")
        return f"{var_name}({idx}:)"

    def _create_association(self, var_data: dict, var_name: str) -> str:
        """Create an association between a variable and its data.

        Args:
            var_data (dict): A dictionary containing information about the variable.
            var_name (str): The name of the variable.

        Returns:
            str: A string representing the association between the variable and its data.
        """
        dimension = var_data.get("dimension")
        if dimension is not None:
            return (
                self.get_slice_expression(var_name, dimension[0])
                if ":" not in dimension
                else f"{var_name}({','.join(dimension)})"
            )
        return var_name

    def _make_init_data(self) -> None:
        """Create an `InitData` object and sets it to the `Init` key in the `data` dictionary.

        Returns:
            None.
        """
        first_intent_in_subroutine = [
            var_dict
            for intent_dict in self.parsed.subroutines.values()
            for intent, var_dict in intent_dict.items()
            if intent == "in"
        ][0]
        startln = self._get_codegen_line(
            first_intent_in_subroutine["codegen_ctx"], "init"
        )
        self.data["Init"] = InitData(
            startln=startln,
            directory=self.directory,
            prefix=self.prefix,
            multinode=self.multinode,
        )

    def _merge_out_inout_fields(self):
        """Merge the `inout` fields into the `in` and `out` fields in the `parsed` dictionary.

        Returns:
            None.
        """
        for _, intent_dict in self.parsed.subroutines.items():
            if "inout" in intent_dict:
                intent_dict["in"].update(intent_dict["inout"])
                intent_dict["out"].update(intent_dict["inout"])
                del intent_dict["inout"]

    @staticmethod
    def _get_codegen_line(ctx: CodegenContext, intent: str):
        if intent == "in":
            return ctx.last_declaration_ln
        elif intent == "out":
            return ctx.end_subroutine_ln
        elif intent == "init":
            return ctx.first_declaration_ln
        else:
            raise ValueError(f"Unrecognized intent: {intent}")

    def _make_imports(self):
        if self.multinode:
            self.data["Import"] = ImportData(startln=self.parsed.last_import_ln)
