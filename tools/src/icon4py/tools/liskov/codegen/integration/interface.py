# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
    optional_module: Optional[str]


@dataclass
class StartFusedStencilData(BaseStartStencilData):
    ...


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
