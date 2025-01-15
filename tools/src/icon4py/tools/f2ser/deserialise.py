# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4pytools.f2ser.parse import CodegenContext, ParsedGranule
from icon4pytools.liskov.codegen.serialisation.interface import (
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
    ):
        self.parsed = parsed
        self.directory = directory
        self.prefix = prefix

    def __call__(self) -> SerialisationCodeInterface:
        """Deserialise the parsed granule and returns a serialisation interface."""
        self._merge_out_inout_fields()
        savepoints = self._make_savepoints()
        init_data = self._make_init_data()
        import_data = self._make_imports()
        return SerialisationCodeInterface(Import=import_data, Init=init_data, Savepoint=savepoints)

    def _make_savepoints(self) -> list[SavepointData]:
        """Create savepoints for each subroutine and intent in the parsed granule."""
        savepoints: list[SavepointData] = []

        for subroutine_name, intent_dict in self.parsed.subroutines.items():
            for intent, var_dict in intent_dict.items():
                savepoints.append(self._create_savepoint(subroutine_name, intent, var_dict))

        return savepoints

    def _create_savepoint(self, subroutine_name: str, intent: str, var_dict: dict) -> SavepointData:
        """Create a savepoint for the given variables.

        Args:
            subroutine_name: The name of the subroutine.
            intent: The intent of the fields to be serialised.
            var_dict: A dictionary representing the variables to be saved.
        """
        field_vals = {k: v for k, v in var_dict.items() if isinstance(v, dict)}
        fields = [
            FieldSerialisationData(
                variable=var_name,
                association=self._create_association(var_data, var_name),
                decomposed=var_data["decomposed"] if var_data.get("decomposed") else False,
                dimension=var_data.get("dimension"),
                typespec=var_data.get("typespec"),
                typename=var_data.get("typename"),
                ptr_var=var_data.get("ptr_var"),
            )
            for var_name, var_data in field_vals.items()
        ]

        return SavepointData(
            subroutine=subroutine_name,
            intent=intent,
            startln=self._get_codegen_line(var_dict["codegen_ctx"], intent),
            fields=fields,
            metadata=None,
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

    def _make_init_data(self) -> InitData:
        """Create an `InitData` object and sets it to the `Init` key in the `data` dictionary."""
        first_intent_in_subroutine = next(
            (
                var_dict
                for intent_dict in self.parsed.subroutines.values()
                for intent, var_dict in intent_dict.items()
                if intent == "in"
            )
        )

        startln = self._get_codegen_line(first_intent_in_subroutine["codegen_ctx"], "init")

        return InitData(
            startln=startln,
            directory=self.directory,
            prefix=self.prefix,
        )

    def _merge_out_inout_fields(self) -> None:
        """Merge the `inout` fields into the `in` and `out` fields in the `parsed` dictionary."""
        for _, intent_dict in self.parsed.subroutines.items():
            if "inout" in intent_dict:
                intent_dict["in"].update(intent_dict["inout"])
                intent_dict["out"].update(intent_dict["inout"])
                del intent_dict["inout"]

    @staticmethod
    def _get_codegen_line(ctx: CodegenContext, intent: str) -> int:
        if intent == "in":
            return ctx.last_declaration_ln
        elif intent == "out":
            return ctx.end_subroutine_ln
        elif intent == "init":
            return ctx.first_declaration_ln
        else:
            raise ValueError(f"Unrecognized intent: {intent}")

    def _make_imports(self) -> ImportData:
        return ImportData(startln=self.parsed.last_import_ln)
