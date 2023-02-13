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
import dataclasses
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable


class UnusedDirective:
    ...


@runtime_checkable
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
    dims: Optional[int]
    abs_tol: Optional[str] = dataclasses.field(kw_only=True, default=None)
    rel_tol: Optional[str] = dataclasses.field(kw_only=True, default=None)
    inp: Optional[bool] = dataclasses.field(kw_only=False, default=None)
    out: Optional[bool] = dataclasses.field(kw_only=False, default=None)


@dataclass
class DeclareData:
    startln: int
    endln: int
    declarations: list[dict[str, str]]
    kind: str


@dataclass
class ImportsData:
    startln: int
    endln: int


@dataclass
class StartCreateData(ImportsData):
    ...


@dataclass
class EndCreateData(ImportsData):
    ...


@dataclass
class EndIfData(ImportsData):
    ...


@dataclass
class StartProfileData(ImportsData):
    name: str


@dataclass
class EndProfileData(ImportsData):
    ...


@dataclass
class StartStencilData:
    name: str
    fields: list[FieldAssociationData]
    bounds: BoundsData
    startln: int
    endln: int
    acc_present: Optional[bool]


@dataclass
class EndStencilData:
    name: str
    startln: int
    endln: int
    noendif: Optional[bool]
    noprofile: Optional[bool]


@dataclass
class DeserialisedDirectives:
    StartStencil: Sequence[StartStencilData]
    EndStencil: Sequence[EndStencilData]
    Declare: DeclareData
    Imports: ImportsData
    StartCreate: StartCreateData
    EndCreate: EndCreateData
    EndIf: Sequence[EndIfData] | UnusedDirective
    StartProfile: Sequence[StartProfileData] | UnusedDirective
    EndProfile: Sequence[EndProfileData] | UnusedDirective
