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

import icon4py.liskov.parsing.types as ts
from icon4py.liskov.codegen.interface import (
    BoundsData,
    CodeGenInput,
    DeclareData,
    DeserialisedDirectives,
    EndCreateData,
    EndStencilData,
    FieldAssociationData,
    ImportsData,
    StartCreateData,
    StartStencilData,
)
from icon4py.liskov.logger import setup_logger
from icon4py.liskov.parsing.exceptions import (
    IncompatibleFieldError,
    MissingBoundsError,
    MissingDirectiveArgumentError,
)
from icon4py.liskov.parsing.utils import StencilCollector, extract_directive
from icon4py.pyutils.metadata import get_field_infos


logger = setup_logger(__name__)


class DirectiveInputFactory(Protocol):
    def __call__(self, parsed: ts.ParsedDict) -> list[CodeGenInput] | CodeGenInput:
        ...


class StartCreateDataFactory:
    def __call__(self, parsed: ts.ParsedDict) -> StartCreateData:
        extracted = extract_directive(parsed["directives"], ts.StartCreate)[0]
        return StartCreateData(startln=extracted.startln, endln=extracted.endln)


class EndCreateDataFactory:
    def __call__(self, parsed: ts.ParsedDict) -> EndCreateData:
        extracted = extract_directive(parsed["directives"], ts.EndCreate)[0]
        return EndCreateData(startln=extracted.startln, endln=extracted.endln)


class ImportsDataFactory:
    def __call__(self, parsed: ts.ParsedDict) -> ImportsData:
        extracted = extract_directive(parsed["directives"], ts.Imports)[0]
        return ImportsData(startln=extracted.startln, endln=extracted.endln)


class DeclareDataFactory:
    def __call__(self, parsed: ts.ParsedDict) -> DeclareData:
        extracted = extract_directive(parsed["directives"], ts.Declare)[0]
        declarations = parsed["content"]["Declare"]
        return DeclareData(
            startln=extracted.startln, endln=extracted.endln, declarations=declarations
        )


class EndStencilDataFactory:
    def __call__(self, parsed: ts.ParsedDict) -> list[EndStencilData]:
        deserialised = []
        extracted = extract_directive(parsed["directives"], ts.EndStencil)
        for i, directive in enumerate(extracted):
            named_args = parsed["content"]["EndStencil"][i]
            stencil_name = _extract_stencil_name(named_args, directive)
            deserialised.append(
                EndStencilData(
                    name=stencil_name, startln=directive.startln, endln=directive.endln
                )
            )
        return deserialised


def _extract_stencil_name(named_args: dict, directive: ts.ParsedDirective) -> str:
    """Extract stencil name from directive arguments."""
    try:
        stencil_name = named_args["name"]
    except KeyError as e:
        raise MissingDirectiveArgumentError(
            f"Missing argument {e} in {directive.type_name} directive on line {directive.startln}."
        )
    return stencil_name


class StartStencilDataFactory:
    TOLERANCE_ARGS = ["abs_tol", "rel_tol"]

    def __call__(self, parsed: ts.ParsedDict) -> list[StartStencilData]:
        """Create and return a list of StartStencilData objects from the parsed directives.

        Args:
            parsed (ParsedDict): Dictionary of parsed directives and their associated content.

        Returns:
            List[StartStencilData]: List of StartStencilData objects created from the parsed directives.
        """
        deserialised = []
        directives = extract_directive(parsed["directives"], ts.StartStencil)
        for i, directive in enumerate(directives):
            named_args = parsed["content"]["StartStencil"][i]
            stencil_name = _extract_stencil_name(named_args, directive)
            bounds = self._get_bounds(named_args)
            fields = self._get_field_associations(named_args)
            fields_w_tolerance = self._update_field_tolerances(named_args, fields)

            deserialised.append(
                StartStencilData(
                    name=stencil_name,
                    fields=fields_w_tolerance,
                    bounds=bounds,
                    startln=directive.startln,
                    endln=directive.endln,
                )
            )
        return deserialised

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
        except Exception:
            raise MissingBoundsError(
                f"Missing or invalid bounds provided in stencil: {named_args['name']}"
            )
        return bounds

    def _get_field_associations(
        self, named_args: dict[str, str]
    ) -> list[FieldAssociationData]:
        """Extract all fields from directive arguments and create corresponding field association data."""
        field_args = self._create_field_args(named_args)
        fields = self._combine_field_info(field_args, named_args)
        return fields

    @staticmethod
    def _create_field_args(named_args: dict[str, str]) -> dict[str, str]:
        """Create a dictionary of field names and their associations from named_args.

        Raises:
            MissingDirectiveArgumentError: If a required argument is missing in the named_args.
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
            raise MissingDirectiveArgumentError(
                f"Missing argument {e} in a StartStencil directive."
            )
        return field_args

    def _combine_field_info(
        self, field_args: dict[str, str], named_args: dict[str, str]
    ) -> list[FieldAssociationData]:
        """Combine directive field info with field info extracted from the corresponding icon4py stencil.

        Raises:
            IncompatibleFieldError: If a used field variable name is incompatible with the expected field
                names defined in the corresponding icon4py stencil.
        """
        stencil_collector = StencilCollector(named_args["name"])
        gt4py_stencil_info = get_field_infos(stencil_collector.fvprog)
        fields = []
        for field_name, association in field_args.items():

            # skipped as handled by _update_field_tolerances
            if any([field_name.endswith(tol) for tol in self.TOLERANCE_ARGS]):
                continue

            try:
                gt4py_field_info = gt4py_stencil_info[field_name]
            except KeyError:
                raise IncompatibleFieldError(
                    f"Used field variable name that is incompatible with the expected field names defined in {named_args['name']} in icon4py."
                )

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


class DirectiveDeserialiser:
    def __init__(self, parsed: ts.ParsedDict) -> None:
        self.directives = self.deserialise(parsed)

    _FACTORIES: dict[str, Callable] = {
        "StartCreate": StartCreateDataFactory(),
        "EndCreate": EndCreateDataFactory(),
        "Imports": ImportsDataFactory(),
        "Declare": DeclareDataFactory(),
        "StartStencil": StartStencilDataFactory(),
        "EndStencil": EndStencilDataFactory(),
    }

    def deserialise(self, directives: ts.ParsedDict) -> DeserialisedDirectives:
        """Deserialise the provided parsed directives to a DeserialisedDirectives object.

        Args:
            directives: The parsed directives to deserialise.

        Returns:
            A DeserialisedDirectives object containing the deserialised directives.

        Note:
            The method uses the `_FACTORIES` class attribute to create the appropriate
            factory object for each directive type, and uses these objects to deserialise
            the parsed directives. The DeserialisedDirectives class is a dataclass
            containing the deserialised versions of the different directives.
        """
        logger.info("Deserialising directives ...")
        deserialised = dict()

        for key, func in self._FACTORIES.items():
            ser = func(directives)
            deserialised[key] = ser

        return DeserialisedDirectives(**deserialised)
