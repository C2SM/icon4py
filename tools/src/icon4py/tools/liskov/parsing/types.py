# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
