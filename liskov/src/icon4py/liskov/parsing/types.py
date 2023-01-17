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
from typing import Any, Protocol, Sequence, Type, TypeAlias, TypedDict


DIRECTIVE_IDENT = "!$DSL"


def noeq_dataclass(cls: Any) -> Any:
    return dataclass(cls, eq=False)  # type: ignore


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


@dataclass
class TypedDirective(RawDirective):
    pattern: str
    regex: str

    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedDirective):
            raise NotImplementedError
        return self.string == other.string


@noeq_dataclass
class WithArguments(TypedDirective):
    # regex enforces default Fortran variable naming standard and includes arithmetic operators and pointer access
    # https://gcc.gnu.org/onlinedocs/gfortran/Naming-conventions.html
    regex: str = field(default=r"(.+?)=(.+?)", init=False)

    def get_content(self) -> dict[str, str]:
        args = self.string.replace(f"{self.pattern}", "")
        delimited = args[1:-1].split(";")
        content = {a.split("=")[0].strip(): a.split("=")[1] for a in delimited}
        return content


@noeq_dataclass
class WithoutArguments(TypedDirective):
    # matches an empty string at the beginning of a line
    regex: str = field(default=r"^(?![\s\S])", init=False)

    def get_content(self) -> dict:
        return {}


@noeq_dataclass
class StartStencil(WithArguments):
    pattern: str = field(default="START STENCIL", init=False)


@noeq_dataclass
class EndStencil(WithArguments):
    pattern: str = field(default="END STENCIL", init=False)


@noeq_dataclass
class Declare(WithArguments):
    pattern: str = field(default="DECLARE", init=False)


@noeq_dataclass
class Imports(WithoutArguments):
    pattern: str = field(default="IMPORTS", init=False)


@noeq_dataclass
class StartCreate(WithoutArguments):
    pattern: str = field(default="START CREATE", init=False)


@noeq_dataclass
class EndCreate(WithoutArguments):
    pattern: str = field(default="END CREATE", init=False)


SUPPORTED_DIRECTIVES: Sequence[Type[ParsedDirective]] = [
    StartStencil,
    EndStencil,
    Imports,
    Declare,
    StartCreate,
    EndCreate,
]
