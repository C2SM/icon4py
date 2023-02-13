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

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Type

import icon4py.liskov.parsing.types as ts
from icon4py.liskov.codegen.interface import (
    BoundsData,
    CodeGenInput,
    DeclareData,
    DeserialisedDirectives,
    EndCreateData,
    EndIfData,
    EndProfileData,
    EndStencilData,
    FieldAssociationData,
    ImportsData,
    StartCreateData,
    StartProfileData,
    StartStencilData,
    UnusedDirective,
)
from icon4py.liskov.common import Step
from icon4py.liskov.logger import setup_logger
from icon4py.liskov.parsing.exceptions import (
    MissingBoundsError,
    MissingDirectiveArgumentError,
)
from icon4py.liskov.parsing.utils import extract_directive, string_to_bool


TOLERANCE_ARGS = ["abs_tol", "rel_tol"]

logger = setup_logger(__name__)


def _extract_stencil_name(named_args: dict, directive: ts.ParsedDirective) -> str:
    """Extract stencil name from directive arguments."""
    try:
        stencil_name = named_args["name"]
    except KeyError as e:
        raise MissingDirectiveArgumentError(
            f"Missing argument {e} in {directive.type_name} directive on line {directive.startln}."
        )
    return stencil_name


def _extract_boolean_kwarg(args: dict, arg_name: str) -> Optional[bool]:
    """Extract a boolean kwarg from the parsed dictionary. Kwargs are false by default."""
    if a := args.get(arg_name):
        return string_to_bool(a)
    return False


class DirectiveInputFactory(Protocol):
    def __call__(
        self, parsed: ts.ParsedDict
    ) -> list[CodeGenInput] | CodeGenInput | Type[UnusedDirective]:
        ...


@dataclass
class DataFactoryBase:
    directive_cls: ts.ParsedDirective
    dtype: Type[CodeGenInput]


@dataclass
class OptionalMultiUseDataFactory(DataFactoryBase):
    def __call__(
        self, parsed: ts.ParsedDict, **kwargs
    ) -> Type[UnusedDirective] | list[CodeGenInput]:
        extracted = extract_directive(parsed["directives"], self.directive_cls)
        if len(extracted) < 1:
            return UnusedDirective
        else:
            deserialised = []
            for directive in extracted:
                deserialised.append(
                    self.dtype(
                        startln=directive.startln, endln=directive.endln, **kwargs
                    )
                )
            return deserialised


@dataclass
class RequiredSingleUseDataFactory(DataFactoryBase):
    def __call__(self, parsed: ts.ParsedDict) -> CodeGenInput:
        extracted = extract_directive(parsed["directives"], self.directive_cls)[0]
        return self.dtype(startln=extracted.startln, endln=extracted.endln)


@dataclass
class StartCreateDataFactory(RequiredSingleUseDataFactory):
    directive_cls: ts.ParsedDict = ts.StartCreate
    dtype: Type[CodeGenInput] = StartCreateData


@dataclass
class EndCreateDataFactory(RequiredSingleUseDataFactory):
    directive_cls: ts.ParsedDict = ts.EndCreate
    dtype: Type[CodeGenInput] = EndCreateData


@dataclass
class ImportsDataFactory(RequiredSingleUseDataFactory):
    directive_cls: ts.ParsedDict = ts.Imports
    dtype: Type[CodeGenInput] = ImportsData


@dataclass
class EndIfDataFactory(OptionalMultiUseDataFactory):
    directive_cls: ts.ParsedDict = ts.EndIf
    dtype: Type[CodeGenInput] = EndIfData


@dataclass
class EndProfileDataFactory(OptionalMultiUseDataFactory):
    directive_cls: ts.ParsedDict = ts.EndProfile
    dtype: Type[CodeGenInput] = EndProfileData


@dataclass
class DeclareDataFactory(DataFactoryBase):
    directive_cls: ts.ParsedDict = ts.Declare
    dtype: Type[CodeGenInput] = DeclareData

    @staticmethod
    def get_field_dimensions(declarations: dict) -> dict[str, int]:
        return {k: len(v.split(",")) for k, v in declarations.items()}

    def __call__(self, parsed: ts.ParsedDict) -> DeclareData:
        extracted = extract_directive(parsed["directives"], self.directive_cls)[0]
        declarations = parsed["content"]["Declare"]
        return self.dtype(
            startln=extracted.startln, endln=extracted.endln, declarations=declarations
        )


@dataclass
class StartProfileDataFactory(DataFactoryBase):
    directive_cls: ts.ParsedDict = ts.StartProfile
    dtype: Type[CodeGenInput] = StartProfileData

    def __call__(self, parsed: ts.ParsedDict) -> list[StartProfileData]:
        deserialised = []
        extracted = extract_directive(parsed["directives"], self.directive_cls)
        for i, directive in enumerate(extracted):
            named_args = parsed["content"]["StartProfile"][i]
            stencil_name = _extract_stencil_name(named_args, directive)
            deserialised.append(
                self.dtype(
                    name=stencil_name,
                    startln=directive.startln,
                    endln=directive.endln,
                )
            )
        return deserialised


@dataclass
class EndStencilDataFactory(DataFactoryBase):
    directive_cls: ts.ParsedDict = ts.EndStencil
    dtype: Type[CodeGenInput] = EndStencilData

    def __call__(self, parsed: ts.ParsedDict) -> list[EndStencilData]:
        deserialised = []
        extracted = extract_directive(parsed["directives"], self.directive_cls)
        for i, directive in enumerate(extracted):
            named_args = parsed["content"]["EndStencil"][i]
            stencil_name = _extract_stencil_name(named_args, directive)
            noendif = _extract_boolean_kwarg(named_args, "noendif")
            noprofile = _extract_boolean_kwarg(named_args, "noprofile")
            deserialised.append(
                self.dtype(
                    name=stencil_name,
                    startln=directive.startln,
                    endln=directive.endln,
                    noendif=noendif,
                    noprofile=noprofile,
                )
            )
        return deserialised


@dataclass
class StartStencilDataFactory(DataFactoryBase):
    directive_cls: ts.ParsedDict = ts.StartStencil
    dtype: Type[CodeGenInput] = StartStencilData

    def __call__(self, parsed: ts.ParsedDict) -> list[StartStencilData]:
        """Create and return a list of StartStencilData objects from the parsed directives.

        Args:
            parsed (ParsedDict): Dictionary of parsed directives and their associated content.

        Returns:
            List[StartStencilData]: List of StartStencilData objects created from the parsed directives.
        """
        deserialised = []
        field_dimensions = DeclareDataFactory.get_field_dimensions(
            parsed["content"]["Declare"][0]
        )
        directives = extract_directive(parsed["directives"], self.directive_cls)
        for i, directive in enumerate(directives):
            named_args = parsed["content"]["StartStencil"][i]
            stencil_name = _extract_stencil_name(named_args, directive)
            bounds = self._make_bounds(named_args)
            fields = self._make_fields(named_args, field_dimensions)
            fields_w_tolerance = self._update_tolerances(named_args, fields)

            deserialised.append(
                self.dtype(
                    name=stencil_name,
                    fields=fields_w_tolerance,
                    bounds=bounds,
                    startln=directive.startln,
                    endln=directive.endln,
                )
            )
        return deserialised

    @staticmethod
    def _make_bounds(named_args: dict) -> BoundsData:
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

    def _make_fields(
        self, named_args: dict[str, str], dimensions: dict
    ) -> list[FieldAssociationData]:
        """Extract all fields from directive arguments and create corresponding field association data."""
        field_args = self._create_field_args(named_args)
        fields = self._make_field_associations(field_args, dimensions)
        return fields

    @staticmethod
    def _create_field_args(named_args: dict[str, str]) -> dict[str, str]:
        """Create a dictionary of field names and their associations from named_args.

        Raises:
            MissingDirectiveArgumentError: If a required argument is missing in the named_args.
        """
        field_args = named_args.copy()
        required_args = (
            "name",
            "horizontal_lower",
            "horizontal_upper",
            "vertical_lower",
            "vertical_upper",
        )

        for arg in required_args:
            if arg not in field_args:
                raise MissingDirectiveArgumentError(
                    f"Missing required argument '{arg}' in a StartStencil directive."
                )
            else:
                field_args.pop(arg)

        return field_args

    @staticmethod
    def _make_field_associations(
        field_args: dict[str, str], dimensions: dict
    ) -> list[FieldAssociationData]:
        """Create a list of FieldAssociation objects."""
        fields = []
        for field_name, association in field_args.items():

            # skipped as handled by _update_field_tolerances
            if any([field_name.endswith(tol) for tol in TOLERANCE_ARGS]):
                continue

            field_association_data = FieldAssociationData(
                variable=field_name,
                association=association,
                dims=dimensions.get(field_name),
            )
            # todo: improve error handling
            fields.append(field_association_data)
        return fields

    @staticmethod
    def _update_tolerances(
        named_args: dict, fields: list[FieldAssociationData]
    ) -> list[FieldAssociationData]:
        """Set relative and absolute tolerance for a given field if set in the directives."""
        for field_name, association in named_args.items():
            for tol in TOLERANCE_ARGS:

                _tol = f"_{tol}"

                if field_name.endswith(_tol):
                    name = field_name.replace(_tol, "")

                    for f in fields:
                        if f.variable == name:
                            setattr(f, tol, association)
        return fields


class DirectiveDeserialiser(Step):
    _FACTORIES: dict[str, Callable] = {
        "StartCreate": StartCreateDataFactory(),
        "EndCreate": EndCreateDataFactory(),
        "Imports": ImportsDataFactory(),
        "Declare": DeclareDataFactory(),
        "StartStencil": StartStencilDataFactory(),
        "EndStencil": EndStencilDataFactory(),
        "EndIf": EndIfDataFactory(),
        "StartProfile": StartProfileDataFactory(),
        "EndProfile": EndProfileDataFactory(),
    }

    def __call__(self, directives: ts.ParsedDict) -> DeserialisedDirectives:
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
