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
from typing import Optional, Protocol


class CodeGenInput(Protocol):
    startln: int
    endln: int


@dataclass
class BoundsData:
    hlower: str
    hupper: str
    vlower: str
    vupper: str


@dataclass
class FieldAssociationData:
    variable: str
    association: str
    inp: bool
    out: bool
    abs_tol: Optional[str] = None
    rel_tol: Optional[str] = None


@dataclass
class DeclareData:
    startln: int
    endln: int
    declarations: list[dict[str, str]]


@dataclass
class ImportsData:
    startln: int
    endln: int


class StartCreateData(ImportsData):
    ...


class EndCreateData(ImportsData):
    ...


@dataclass
class StartStencilData:
    name: str
    fields: list[FieldAssociationData]
    bounds: BoundsData
    startln: int
    endln: int


@dataclass
class EndStencilData:
    name: str
    startln: int
    endln: int


@dataclass
class SerialisedDirectives:
    StartStencil: list[StartStencilData]
    EndStencil: list[EndStencilData]
    Declare: DeclareData
    Imports: ImportsData
    StartCreate: StartCreateData
    EndCreate: EndCreateData
