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
from typing import Protocol


IDENTIFIER = "!$DSL"


class NoDirectivesFound:
    pass


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

    def __hash__(self):
        return hash(self.string)

    def __eq__(self, other):
        return self.string == other.string


class StartStencil:
    pattern = f"{IDENTIFIER} START"


class EndStencil:
    pattern = f"{IDENTIFIER} END"


class Declare:
    pattern = f"{IDENTIFIER} DECLARE"


class Create:
    pattern = f"{IDENTIFIER} CREATE"
