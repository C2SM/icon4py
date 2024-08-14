# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Optional

from icon4pytools.liskov.codegen.shared.types import CodeGenInput


@dataclass
class InitData(CodeGenInput):
    directory: str
    prefix: str


@dataclass
class Metadata:
    key: str
    value: str


@dataclass
class FieldSerialisationData:
    variable: str
    association: str
    decomposed: bool = False
    device: Optional[str] = "cpu"
    dimension: Optional[list[str]] = None
    typespec: Optional[str] = None
    typename: Optional[str] = None
    ptr_var: Optional[str] = None


@dataclass
class SavepointData(CodeGenInput):
    subroutine: str
    intent: str
    fields: list[FieldSerialisationData]
    metadata: Optional[list[Metadata]]


class ImportData(CodeGenInput):
    ...


@dataclass
class SerialisationCodeInterface:
    Import: ImportData
    Init: InitData
    Savepoint: list[SavepointData]
