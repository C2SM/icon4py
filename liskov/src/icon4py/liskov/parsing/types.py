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
from dataclasses import dataclass, field
from importlib import import_module
from inspect import getmembers, isclass
from typing import (
    Any,
    Dict,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    TypedDict,
    TypeVar,
    runtime_checkable,
)


DIRECTIVE_IDENT = "!$DSL"

T = TypeVar("T", bound="CheckForDirectiveClasses")


class CheckForDirectiveClasses(type):
    """Metaclass to be used when defining a new class which implements the ParsedDirective protocol.

    This class checks that the required classes in their respective modules are defined
    according to their naming conventions. When adding a new directive, first a new subclass
    which implements the ParsedDirective protocol must be defined. Then a corresponding
    <directive_name>Data, <directive_name>DataFactory, <directive_name>Statement and
    <directive_name>StatementGenerator class must be defined in their respective module
    which can be seen in _CLS_REQS.
    """

    _CLS_REQS = (
        ("codegen", "interface", "Data"),
        ("parsing", "deserialise", "DataFactory"),
        ("codegen", "f90", "Statement"),
        ("codegen", "f90", "StatementGenerator"),
    )

    def __new__(
        mcs: Type[T],
        name: str,
        bases: Tuple[Type, ...],
        namespace: Dict[str, Any],
        **kwargs: Any,
    ) -> T:
        for subpkg, module_name, cls_suffix in mcs._CLS_REQS:
            module = import_module(f"icon4py.liskov.{subpkg}.{module_name}")
            expected_cls_name = f"{name}{cls_suffix}"
            if expected_cls_name not in [
                cls_name for cls_name, _ in getmembers(module, predicate=isclass)
            ]:
                raise NotImplementedError(
                    f"Required class of name {expected_cls_name} missing in {module}"
                )
        return super().__new__(mcs, name, bases, namespace, **kwargs)


@runtime_checkable
class ParsedDirective(Protocol):
    string: str
    startln: int
    endln: int
    pattern: str
    regex: str

    @property
    def type_name(self) -> str:
        ...

    def get_content(self) -> Any:
        ...


ParsedContent: TypeAlias = dict[str, list[dict[str, str]]]


class ParsedDict(TypedDict):
    directives: Sequence[ParsedDirective]
    content: ParsedContent


@dataclass
class RawDirective:
    string: str
    startln: int
    endln: int


class TypedDirective(RawDirective):
    pattern: str

    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedDirective):
            raise NotImplementedError
        return self.string == other.string


@dataclass(eq=False)
class WithArguments(TypedDirective):
    # regex enforces default Fortran variable naming standard and includes arithmetic operators and pointer access
    # https://gcc.gnu.org/onlinedocs/gfortran/Naming-conventions.html
    regex: str = field(default=r"(.+?)=(.+?)", init=False)

    def get_content(self) -> dict[str, str]:
        args = self.string.replace(f"{self.pattern}", "")
        delimited = args[1:-1].split(";")
        content = {a.split("=")[0].strip(): a.split("=")[1] for a in delimited}
        return content


@dataclass(eq=False)
class WithoutArguments(TypedDirective):
    # matches an empty string at the beginning of a line
    regex: str = field(default=r"^(?![\s\S])", init=False)

    def get_content(self) -> dict:
        return {}


class StartStencil(WithArguments, metaclass=CheckForDirectiveClasses):
    pattern = "START STENCIL"


class EndStencil(WithArguments, metaclass=CheckForDirectiveClasses):
    pattern = "END STENCIL"


class Declare(WithArguments, metaclass=CheckForDirectiveClasses):
    pattern = "DECLARE"


class Imports(WithoutArguments, metaclass=CheckForDirectiveClasses):
    pattern = "IMPORTS"


class StartCreate(WithoutArguments, metaclass=CheckForDirectiveClasses):
    pattern = "START CREATE"


class EndCreate(WithoutArguments, metaclass=CheckForDirectiveClasses):
    pattern = "END CREATE"


class EndIf(WithoutArguments, metaclass=CheckForDirectiveClasses):
    pattern = "ENDIF"


# When adding a new directive this list must be updated.
SUPPORTED_DIRECTIVES: Sequence[Type[ParsedDirective]] = [
    StartStencil,
    EndStencil,
    Imports,
    Declare,
    StartCreate,
    EndCreate,
    EndIf,
]


class UnusedDirective:
    pass
