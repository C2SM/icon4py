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
from typing import (
    Any,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeAlias,
    TypedDict,
    runtime_checkable,
)


DIRECTIVE_IDENT = "!$DSL"


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
    regex: str = field(default=r"(.+?)=(.+?)", init=False)

    def get_content(self) -> dict[str, str]:
        args = self.string.replace(f"{self.pattern}", "")
        delimited = args[1:-1].split(";")
        content = {a.split("=")[0].strip(): a.split("=")[1] for a in delimited}
        return content


@dataclass(eq=False)
class WithOptionalArguments(TypedDirective):
    regex: str = field(default=r"(?:.+?=.+?|)", init=False)

    def get_content(self) -> Optional[dict[str, str]]:
        args = self.string.replace(f"{self.pattern}", "")[1:-1]
        if len(args) > 0:
            content = dict([args.split("=")])
            return content
        return None


@dataclass(eq=False)
class WithoutArguments(TypedDirective):
    # matches an empty string at the beginning of a line
    regex: str = field(default=r"^(?![\s\S])", init=False)

    def get_content(self) -> dict:
        return {}


@dataclass(eq=False)
class FreeForm(TypedDirective):
    # matches any string inside brackets
    regex: str = field(default=r"(.+?)", init=False)

    def get_content(self) -> str:
        args = self.string.replace(f"{self.pattern}", "")
        return args[1:-1]


class StartStencil(WithArguments):
    pattern = "START STENCIL"


class EndStencil(WithArguments):
    pattern = "END STENCIL"


class Declare(WithArguments):
    pattern = "DECLARE"


class Imports(WithoutArguments):
    pattern = "IMPORTS"


class StartCreate(WithOptionalArguments):
    pattern = "START CREATE"


class EndCreate(WithoutArguments):
    pattern = "END CREATE"


class EndIf(WithoutArguments):
    pattern = "ENDIF"


class StartProfile(WithArguments):
    pattern = "START PROFILE"


class EndProfile(WithoutArguments):
    pattern = "END PROFILE"


class Insert(FreeForm):
    pattern = "INSERT"


# When adding a new directive this list must be updated.
SUPPORTED_DIRECTIVES: Sequence[Type[ParsedDirective]] = [
    StartStencil,
    EndStencil,
    Imports,
    Declare,
    StartCreate,
    EndCreate,
    EndIf,
    StartProfile,
    EndProfile,
    Insert,
]
