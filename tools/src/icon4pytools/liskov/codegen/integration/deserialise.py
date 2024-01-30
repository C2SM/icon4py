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

from typing import Any, Optional, Protocol, Sequence, Type, cast

import icon4pytools.liskov.parsing.parse
import icon4pytools.liskov.parsing.types as ts
from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.codegen.integration.interface import (
    BoundsData,
    DeclareData,
    EndCreateData,
    EndDeleteData,
    EndFusedStencilData,
    EndIfData,
    EndProfileData,
    EndStencilData,
    FieldAssociationData,
    ImportsData,
    InsertData,
    IntegrationCodeInterface,
    StartCreateData,
    StartDeleteData,
    StartFusedStencilData,
    StartProfileData,
    StartStencilData,
    UnusedDirective,
)
from icon4pytools.liskov.codegen.shared.deserialise import Deserialiser
from icon4pytools.liskov.codegen.shared.types import CodeGenInput
from icon4pytools.liskov.parsing.exceptions import (
    DirectiveSyntaxError,
    MissingBoundsError,
    MissingDirectiveArgumentError,
)
from icon4pytools.liskov.parsing.utils import (
    extract_directive,
    flatten_list_of_dicts,
    string_to_bool,
)


TOLERANCE_ARGS = ["abs_tol", "rel_tol"]
DEFAULT_DECLARE_IDENT_TYPE = "REAL(wp)"
DEFAULT_DECLARE_SUFFIX = "before"

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


def _extract_boolean_kwarg(
    directive: ts.ParsedDirective, args: dict, arg_name: str
) -> Optional[bool]:
    """Extract a boolean kwarg from the parsed dictionary. Kwargs are false by default."""
    if a := args.get(arg_name):
        try:
            return string_to_bool(a)
        except Exception:
            raise DirectiveSyntaxError(
                f"Expected boolean string as value to keyword argument {arg_name} on line {directive.startln}. Got {a}"
            )
    return False


def pop_item_from_dict(dictionary: dict, key: str, default_value: Optional[str]) -> str:
    return dictionary.pop(key, default_value)


class DirectiveInputFactory(Protocol):
    def __call__(
        self, parsed: ts.ParsedDict
    ) -> list[CodeGenInput] | CodeGenInput | Type[UnusedDirective]:
        ...


class DataFactoryBase:
    directive_cls: Type[ts.ParsedDirective]
    dtype: Type[CodeGenInput]


class OptionalMultiUseDataFactory(DataFactoryBase):
    def __call__(
        self, parsed: ts.ParsedDict, **kwargs: Any
    ) -> Type[UnusedDirective] | list[CodeGenInput]:
        extracted = extract_directive(parsed["directives"], self.directive_cls)
        if len(extracted) < 1:
            return UnusedDirective
        else:
            deserialised = []
            for directive in extracted:
                deserialised.append(self.dtype(startln=directive.startln, **kwargs))
            return deserialised


class RequiredSingleUseDataFactory(DataFactoryBase):
    def __call__(self, parsed: ts.ParsedDict) -> CodeGenInput:
        extracted = extract_directive(parsed["directives"], self.directive_cls)[0]
        return self.dtype(startln=extracted.startln)


class EndCreateDataFactory(OptionalMultiUseDataFactory):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.EndCreate
    dtype: Type[EndCreateData] = EndCreateData


class ImportsDataFactory(RequiredSingleUseDataFactory):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.Imports
    dtype: Type[ImportsData] = ImportsData


class EndIfDataFactory(OptionalMultiUseDataFactory):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.EndIf
    dtype: Type[EndIfData] = EndIfData


class EndProfileDataFactory(OptionalMultiUseDataFactory):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.EndProfile
    dtype: Type[EndProfileData] = EndProfileData


class EndDeleteDataFactory(OptionalMultiUseDataFactory):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.EndDelete
    dtype: Type[EndDeleteData] = EndDeleteData


class StartDeleteDataFactory(OptionalMultiUseDataFactory):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.StartDelete
    dtype: Type[StartDeleteData] = StartDeleteData


class StartCreateDataFactory(DataFactoryBase):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.StartCreate
    dtype: Type[StartCreateData] = StartCreateData

    def __call__(self, parsed: ts.ParsedDict) -> list[StartCreateData] | Type[UnusedDirective]:
        deserialised = []
        extracted = extract_directive(parsed["directives"], self.directive_cls)

        if len(extracted) < 1:
            return UnusedDirective

        for i, directive in enumerate(extracted):
            named_args = parsed["content"]["StartCreate"][i]

            extra_fields = None
            if named_args:
                extra_fields = named_args["extra_fields"].split(",")

            deserialised.append(
                self.dtype(
                    startln=directive.startln,
                    extra_fields=extra_fields,
                )
            )

        return deserialised


class DeclareDataFactory(DataFactoryBase):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.Declare
    dtype: Type[DeclareData] = DeclareData

    @staticmethod
    def get_field_dimensions(declarations: dict) -> dict[str, int]:
        return {k: len(v.split(",")) for k, v in declarations.items()}

    def __call__(self, parsed: ts.ParsedDict) -> list[DeclareData]:
        deserialised = []
        extracted = extract_directive(parsed["directives"], self.directive_cls)
        for i, directive in enumerate(extracted):
            named_args = parsed["content"]["Declare"][i]
            ident_type = pop_item_from_dict(named_args, "type", DEFAULT_DECLARE_IDENT_TYPE)
            suffix = pop_item_from_dict(named_args, "suffix", DEFAULT_DECLARE_SUFFIX)
            deserialised.append(
                self.dtype(
                    startln=directive.startln,
                    declarations=named_args,
                    ident_type=ident_type,
                    suffix=suffix,
                )
            )
        return deserialised


class StartProfileDataFactory(DataFactoryBase):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.StartProfile
    dtype: Type[StartProfileData] = StartProfileData

    def __call__(self, parsed: ts.ParsedDict) -> list[StartProfileData]:
        deserialised = []
        extracted = extract_directive(parsed["directives"], self.directive_cls)
        for i, directive in enumerate(extracted):
            named_args = parsed["content"]["StartProfile"][i]
            stencil_name = _extract_stencil_name(named_args, directive)
            deserialised.append(self.dtype(name=stencil_name, startln=directive.startln))
        return deserialised


class EndStencilDataFactory(DataFactoryBase):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.EndStencil
    dtype: Type[EndStencilData] = EndStencilData

    def __call__(self, parsed: ts.ParsedDict) -> list[EndStencilData]:
        deserialised = []
        extracted = extract_directive(parsed["directives"], self.directive_cls)
        for i, directive in enumerate(extracted):
            named_args = parsed["content"]["EndStencil"][i]
            stencil_name = _extract_stencil_name(named_args, directive)
            noendif = _extract_boolean_kwarg(directive, named_args, "noendif")
            noprofile = _extract_boolean_kwarg(directive, named_args, "noprofile")
            noaccenddata = _extract_boolean_kwarg(directive, named_args, "noaccenddata")
            deserialised.append(
                self.dtype(
                    name=stencil_name,
                    startln=directive.startln,
                    noendif=noendif,
                    noprofile=noprofile,
                    noaccenddata=noaccenddata,
                )
            )
        return deserialised


class EndFusedStencilDataFactory(DataFactoryBase):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.EndFusedStencil
    dtype: Type[EndFusedStencilData] = EndFusedStencilData

    def __call__(self, parsed: ts.ParsedDict) -> list[EndFusedStencilData]:
        deserialised = []
        extracted = extract_directive(parsed["directives"], self.directive_cls)
        for i, directive in enumerate(extracted):
            named_args = parsed["content"]["EndFusedStencil"][i]
            stencil_name = _extract_stencil_name(named_args, directive)
            deserialised.append(
                self.dtype(
                    name=stencil_name,
                    startln=directive.startln,
                )
            )
        return deserialised


class StartStencilDataFactoryBase(DataFactoryBase):
    directive_cls: Type[ts.ParsedDirective]
    dtype: Type[StartStencilData | StartFusedStencilData]

    def __call__(self, parsed: ts.ParsedDict) -> list[StartStencilData | StartFusedStencilData]:
        field_dimensions = flatten_list_of_dicts(
            [DeclareDataFactory.get_field_dimensions(dim) for dim in parsed["content"]["Declare"]]
        )
        directives = extract_directive(parsed["directives"], self.directive_cls)
        return self.create_stencil_data(
            parsed, field_dimensions, directives, self.directive_cls, self.dtype
        )

    def create_stencil_data(
        self,
        parsed: ts.ParsedDict,
        field_dimensions: dict[str, Any],
        directives: Sequence[ts.ParsedDirective],
        directive_cls: Type[ts.ParsedDirective],
        dtype: Type[StartStencilData | StartFusedStencilData],
    ) -> list[StartStencilData | StartFusedStencilData]:
        """Create and return a list of StartStencilData or StartFusedStencilData objects from parsed directives."""
        deserialised = []
        for i, directive in enumerate(directives):
            named_args = parsed["content"][directive_cls.__name__][i]
            additional_attrs = self._pop_additional_attributes(dtype, named_args)
            acc_present = string_to_bool(pop_item_from_dict(named_args, "accpresent", "true"))
            stencil_name = _extract_stencil_name(named_args, directive)
            bounds = self._make_bounds(named_args)
            fields = self._make_fields(named_args, field_dimensions)
            fields_w_tolerance = self._update_tolerances(named_args, fields)
            deserialised.append(
                dtype(
                    name=stencil_name,
                    fields=fields_w_tolerance,
                    bounds=bounds,
                    startln=directive.startln,
                    acc_present=acc_present,
                    **additional_attrs,
                )
            )
        return deserialised

    def _pop_additional_attributes(
        self,
        dtype: Type[StartStencilData | StartFusedStencilData],
        named_args: dict[str, Any],
    ) -> dict:
        """Pop and return additional attributes specific to StartStencilData."""
        additional_attrs = {}
        if dtype == StartStencilData:
            mergecopy = string_to_bool(pop_item_from_dict(named_args, "mergecopy", "false"))
            copies = string_to_bool(pop_item_from_dict(named_args, "copies", "true"))
            optional_module = pop_item_from_dict(named_args, "optional_module", "None")
            additional_attrs = {"mergecopy": mergecopy, "copies": copies, "optional_module": optional_module}
        return additional_attrs

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


class StartFusedStencilDataFactory(StartStencilDataFactoryBase):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.StartFusedStencil
    dtype: Type[StartFusedStencilData] = StartFusedStencilData


class StartStencilDataFactory(StartStencilDataFactoryBase):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.StartStencil
    dtype: Type[StartStencilData] = StartStencilData


class InsertDataFactory(DataFactoryBase):
    directive_cls: Type[ts.ParsedDirective] = icon4pytools.liskov.parsing.parse.Insert
    dtype: Type[InsertData] = InsertData

    def __call__(self, parsed: ts.ParsedDict) -> list[InsertData]:
        deserialised = []
        extracted = extract_directive(parsed["directives"], self.directive_cls)
        for i, directive in enumerate(extracted):
            content = cast(str, parsed["content"]["Insert"][i])
            deserialised.append(self.dtype(startln=directive.startln, content=content))
        return deserialised


class IntegrationCodeDeserialiser(Deserialiser):
    _FACTORIES = {
        "StartCreate": StartCreateDataFactory(),
        "EndCreate": EndCreateDataFactory(),
        "Imports": ImportsDataFactory(),
        "Declare": DeclareDataFactory(),
        "StartStencil": StartStencilDataFactory(),
        "EndStencil": EndStencilDataFactory(),
        "StartFusedStencil": StartFusedStencilDataFactory(),
        "EndFusedStencil": EndFusedStencilDataFactory(),
        "StartDelete": StartDeleteDataFactory(),
        "EndDelete": EndDeleteDataFactory(),
        "EndIf": EndIfDataFactory(),
        "StartProfile": StartProfileDataFactory(),
        "EndProfile": EndProfileDataFactory(),
        "Insert": InsertDataFactory(),
    }
    _INTERFACE_TYPE = IntegrationCodeInterface
