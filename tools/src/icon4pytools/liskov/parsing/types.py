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
from typing import Any, Protocol, Sequence, TypeAlias, TypedDict, runtime_checkable

DIRECTIVE_IDENT = "!$DSL"


@runtime_checkable
class ParsedDirective(Protocol):
    string: str
    startln: int
    endln: int
    pattern: str
    regex: str

    def __init__(self, string: str, startln: int, endln: int) -> None:
        ...

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
