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

import copy
from typing import Callable, Protocol

from icon4py.liskov.codegen.interface import (
    BoundsData,
    CodeGenInput,
    CreateData,
    DeclareData,
    EndStencilData,
    FieldAssociationData,
    ImportsData,
    SerialisedDirectives,
    StartStencilData,
)
from icon4py.liskov.parsing.types import (
    Create,
    Declare,
    EndStencil,
    Imports,
    ParsedType,
    StartStencil,
)
from icon4py.liskov.parsing.utils import StencilCollector, extract_directive
from icon4py.pyutils.metadata import get_field_infos


class DirectiveInputFactory(Protocol):
    def __call__(self, parsed: dict) -> list[CodeGenInput] | CodeGenInput:
        ...


class CreateDataFactory:
    def __call__(self, parsed: dict) -> CreateData:
        extracted = extract_directive(parsed["directives"], Create)[0]
        return CreateData(startln=extracted.startln, endln=extracted.endln)


class ImportsDataFactory:
    def __call__(self, parsed: dict) -> ImportsData:
        extracted = extract_directive(parsed["directives"], Imports)[0]
        return ImportsData(startln=extracted.startln, endln=extracted.endln)


class DeclareDataFactory:
    def __call__(self, parsed: dict) -> DeclareData:
        extracted = extract_directive(parsed["directives"], Declare)[0]
        declarations = parsed["content"]["Declare"]
        return DeclareData(
            startln=extracted.startln, endln=extracted.endln, declarations=declarations
        )


class StartStencilDataFactory:
    TOLERANCE_ARGS = ["abs_tol", "rel_tol"]

    def __call__(self, parsed: dict) -> list[StartStencilData]:
        serialised = []
        directives = extract_directive(parsed["directives"], StartStencil)
        for i, directive in enumerate(directives):
            named_args = parsed["content"]["Start"][i]
            stencil_name = named_args["name"]
            fields = self._get_field_associations(named_args)
            fields_w_tolerance = self._update_field_tolerances(named_args, fields)
            bounds = self._get_bounds(named_args)
            try:
                serialised.append(
                    StartStencilData(
                        name=stencil_name,
                        fields=fields_w_tolerance,
                        bounds=bounds,
                        startln=directive.startln,
                        endln=directive.endln,
                    )
                )
            except Exception as e:
                raise e
        return serialised

    @staticmethod
    def _get_bounds(named_args: dict) -> BoundsData:
        """Extract stencil bounds from directive arguments."""
        try:
            bounds = BoundsData(
                hlower=named_args["horizontal_lower"],
                hupper=named_args["horizontal_upper"],
                vlower=named_args["vertical_lower"],
                vupper=named_args["vertical_upper"],
            )
        except Exception as e:
            # todo: more specific exception
            raise e
        return bounds

    def _get_field_associations(
        self, named_args: dict[str, str]
    ) -> list[FieldAssociationData]:
        """Extract all fields from directive arguments and create corresponding field association data.

        For each directive, the corresponding gt4py stencil is parsed, which is used to infer the
        input and output intent of each field.
        """
        try:
            field_args = copy.copy(named_args)
            entries_to_remove = (
                "name",
                "horizontal_lower",
                "horizontal_upper",
                "vertical_lower",
                "vertical_upper",
            )
            list(map(field_args.pop, entries_to_remove))
        except Exception as e:
            # todo: more specific exception
            raise e

        stencil_collector = StencilCollector(named_args["name"])
        gt4py_stencil_info = get_field_infos(stencil_collector.fvprog)

        # todo: handle KeyError with exception
        fields = []
        for field_name, association in field_args.items():

            if any([field_name.endswith(tol) for tol in self.TOLERANCE_ARGS]):
                continue

            gt4py_field_info = gt4py_stencil_info[field_name]

            field_association_data = FieldAssociationData(
                variable=field_name,
                association=association,
                inp=gt4py_field_info.inp,
                out=gt4py_field_info.out,
            )

            fields.append(field_association_data)
        return fields

    def _update_field_tolerances(
        self, named_args: dict, fields: list[FieldAssociationData]
    ) -> list[FieldAssociationData]:
        """Set relative and absolute tolerance for a given field if set in the directives."""
        for field_name, association in named_args.items():
            for tol in self.TOLERANCE_ARGS:

                _tol = f"_{tol}"

                if field_name.endswith(_tol):
                    name = field_name.replace(_tol, "")

                    for f in fields:
                        if f.variable == name:
                            setattr(f, tol, association)
        return fields


class EndStencilDataFactory:
    def __call__(self, parsed: dict) -> list[EndStencilData]:
        serialised = []
        extracted = extract_directive(parsed["directives"], EndStencil)
        for i, directive in enumerate(extracted):
            named_args = parsed["content"]["End"][i]
            stencil_name = named_args["name"]
            serialised.append(
                EndStencilData(
                    name=stencil_name, startln=directive.startln, endln=directive.endln
                )
            )
        return serialised


class DirectiveSerialiser:
    def __init__(self, parsed: ParsedType):
        self.directives = self.serialise(parsed)

    _FACTORIES: dict[str, Callable] = {
        "create": CreateDataFactory(),
        "imports": ImportsDataFactory(),
        "declare": DeclareDataFactory(),
        "start": StartStencilDataFactory(),
        "end": EndStencilDataFactory(),
    }

    def serialise(self, directives: ParsedType) -> SerialisedDirectives:
        serialised = dict()

        for key, func in self._FACTORIES.items():
            ser = func(directives)
            serialised[key] = ser

        return SerialisedDirectives(**serialised)
