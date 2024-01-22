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
from typing import Optional, Sequence

from icon4pytools.liskov.codegen.shared.types import CodeGenInput


class UnusedDirective:
    ...


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
class DeclareData(CodeGenInput):
    declarations: dict[str, str]
    ident_type: str
    suffix: str


@dataclass
class ImportsData(CodeGenInput):
    ...


@dataclass
class StartCreateData(CodeGenInput):
    extra_fields: Optional[list[str]]


@dataclass
class EndCreateData(CodeGenInput):
    ...


@dataclass
class EndIfData(CodeGenInput):
    ...


@dataclass
class StartProfileData(CodeGenInput):
    name: str


@dataclass
class EndProfileData(CodeGenInput):
    ...


@dataclass
class BaseStartStencilData(CodeGenInput):
    name: str
    fields: list[FieldAssociationData]
    acc_present: Optional[bool]
    bounds: BoundsData


@dataclass
class StartStencilData(BaseStartStencilData):
    mergecopy: Optional[bool]
    copies: Optional[bool]


@dataclass
class StartFusedStencilData(BaseStartStencilData):
    ...


@dataclass
class StartOptionalStencilData(BaseStartStencilData):
    optional_module: str


@dataclass
class BaseEndStencilData(CodeGenInput):
    name: str


@dataclass
class EndStencilData(BaseEndStencilData):
    noendif: Optional[bool]
    noprofile: Optional[bool]
    noaccenddata: Optional[bool]


@dataclass
class EndFusedStencilData(BaseEndStencilData):
    ...


@dataclass
class EndOptionalStencilData(BaseEndStencilData):
    ...


@dataclass
class StartDeleteData(CodeGenInput):
    startln: int


@dataclass
class EndDeleteData(StartDeleteData):
    ...


@dataclass
class InsertData(CodeGenInput):
    content: str


@dataclass
class IntegrationCodeInterface:
    StartStencil: Sequence[StartStencilData]
    EndStencil: Sequence[EndStencilData]
    StartFusedStencil: Sequence[StartFusedStencilData]
    EndFusedStencil: Sequence[EndFusedStencilData]
    StartOptionalStencil: Sequence[StartOptionalStencilData]
    EndOptionalStencil: Sequence[EndOptionalStencilData]
    StartDelete: Sequence[StartDeleteData] | UnusedDirective
    EndDelete: Sequence[EndDeleteData] | UnusedDirective
    Declare: Sequence[DeclareData]
    Imports: ImportsData
    StartCreate: Sequence[StartCreateData] | UnusedDirective
    EndCreate: Sequence[EndCreateData] | UnusedDirective
    EndIf: Sequence[EndIfData] | UnusedDirective
    StartProfile: Sequence[StartProfileData] | UnusedDirective
    EndProfile: Sequence[EndProfileData] | UnusedDirective
    Insert: Sequence[InsertData] | UnusedDirective
