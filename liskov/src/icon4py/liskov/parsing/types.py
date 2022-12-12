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
from typing import Any, Protocol, TypeAlias


IDENTIFIER = "!$DSL"


class NoDirectivesFound:
    pass


ParsedType: TypeAlias = dict[Any, Any] | NoDirectivesFound


class Directive(Protocol):
    pattern: str


@dataclass
class RawDirective:
    string: str
    startln: int
    endln: int


@dataclass
class TypedDirective(RawDirective):
    directive_type: Directive

    def __hash__(self) -> int:
        return hash(self.string)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedDirective):
            raise NotImplementedError
        return self.string == other.string


class DirectiveType:
    pattern: str

    def __str__(self) -> str:
        return self.pattern.capitalize()


class StartStencil(DirectiveType):
    pattern = "START"


class EndStencil(DirectiveType):
    pattern = "END"


class Declare(DirectiveType):
    pattern = "DECLARE"


class Imports(DirectiveType):
    pattern = "IMPORT"
