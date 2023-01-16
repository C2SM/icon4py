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
from typing import Protocol, TypeAlias, TypedDict


DIRECTIVE_IDENT = "!$DSL"


class NoDirectivesFound:
    pass


class Directive(Protocol):
    pattern: str

    def __str__(self) -> str:
        return self.pattern


@dataclass
class RawDirective:
    string: str
    startln: int
    endln: int


@dataclass
class DirectiveType:
    @property
    def type_name(self):
        return self.__class__.__name__


@dataclass
class TypedDirective(RawDirective, DirectiveType):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedDirective):
            raise NotImplementedError
        return self.string == other.string


class WithArguments(TypedDirective):
    # matches an empty string at the beginning of a line
    regex = r"(.+?)=(.+?)"

    def get_content(self):
        args = self.string.replace(f"{self.pattern}", "")
        delimited = args[1:-1].split(";")
        content = {a.split("=")[0].strip(): a.split("=")[1] for a in delimited}
        return content


class WithoutArguments(TypedDirective):
    # matches an empty string at the beginning of a line
    regex = r"^(?![\s\S])"

    @staticmethod
    def get_content():
        return {}


class StartStencil(WithArguments):
    pattern = "START STENCIL"


class EndStencil(WithArguments):
    pattern = "END STENCIL"


class Declare(WithArguments):
    pattern = "DECLARE"


class Imports(WithoutArguments):
    pattern = "IMPORTS"


class StartCreate(WithoutArguments):
    pattern = "START CREATE"


class EndCreate(WithoutArguments):
    pattern = "END CREATE"


ParsedContent: TypeAlias = dict[str, list[dict[str, str]]]


class ParsedDict(TypedDict):
    directives: list[TypedDirective]
    content: ParsedContent
